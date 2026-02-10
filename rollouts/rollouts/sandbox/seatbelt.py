"""macOS Seatbelt sandbox implementation.

Uses /usr/bin/sandbox-exec with SBPL (Seatbelt Profile Language) policies
to restrict process capabilities.

Implementation based on OpenAI Codex CLI (MIT License):
https://github.com/openai/codex/blob/main/codex-rs/core/src/seatbelt.rs
"""

import os
import re
import subprocess
from pathlib import Path

import trio

from rollouts.sandbox.executor import SandboxError, SandboxResult
from rollouts.sandbox.policy import SandboxPolicy

# Hardcoded path to sandbox-exec to prevent PATH injection attacks
SANDBOX_EXEC_PATH = "/usr/bin/sandbox-exec"

# Base policy inspired by Chrome's sandbox and Codex
# Starts with deny-by-default, allows essential operations
SEATBELT_BASE_POLICY = """\
(version 1)

; Start with closed-by-default
(deny default)

; Child processes inherit sandbox policy
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))

; Allow reading user preferences
(allow user-preference-read)

; Process info for same sandbox
(allow process-info* (target same-sandbox))

; Allow writing to /dev/null
(allow file-write-data
  (require-all
    (path "/dev/null")
    (vnode-type CHARACTER-DEVICE)))

; Essential sysctls for system info
(allow sysctl-read
  (sysctl-name "hw.activecpu")
  (sysctl-name "hw.byteorder")
  (sysctl-name "hw.cachelinesize_compat")
  (sysctl-name "hw.cpufamily")
  (sysctl-name "hw.cputype")
  (sysctl-name "hw.logicalcpu_max")
  (sysctl-name "hw.machine")
  (sysctl-name "hw.memsize")
  (sysctl-name "hw.ncpu")
  (sysctl-name "hw.pagesize")
  (sysctl-name "hw.physicalcpu")
  (sysctl-name "hw.physicalcpu_max")
  (sysctl-name "kern.argmax")
  (sysctl-name "kern.hostname")
  (sysctl-name "kern.maxfilesperproc")
  (sysctl-name "kern.osproductversion")
  (sysctl-name "kern.osrelease")
  (sysctl-name "kern.ostype")
  (sysctl-name "kern.osversion")
  (sysctl-name "kern.version")
  (sysctl-name "vm.loadavg")
  (sysctl-name-prefix "hw.optional.arm.")
  (sysctl-name-prefix "hw.perflevel")
  (sysctl-name-prefix "kern.proc.pid.")
)

; IOKit for power management
(allow iokit-open
  (iokit-registry-entry-class "RootDomainUserClient")
)

; IOKit for Metal GPU access
(allow iokit-open
  (iokit-registry-entry-class "AGXDeviceUserClient")
  (iokit-registry-entry-class "AGXSharedUserClient")
  (iokit-registry-entry-class "IOGPUDeviceUserClient")
  (iokit-registry-entry-class-prefix "AGXG")
  (iokit-registry-entry-class-prefix "IOMetal")
  (iokit-registry-entry-class-prefix "IOAccel")
)

; Xcode developer tools access (for xcrun metal, etc.)
(allow file-read*
  (subpath "/Applications/Xcode.app")
  (subpath "/Library/Developer")
)
(allow process-exec
  (subpath "/Applications/Xcode.app")
  (literal "/usr/bin/xcrun")
)

; Directory services lookup
(allow mach-lookup
  (global-name "com.apple.system.opendirectoryd.libinfo")
  (global-name "com.apple.PowerManagement.control")
)

; Python multiprocessing SemLock support
(allow ipc-posix-sem)

; PTY support for interactive shells
(allow pseudo-tty)
(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))
(allow file-read* file-write*
  (require-all
    (regex #"^/dev/ttys[0-9]+")
    (extension "com.apple.sandbox.pty")))
(allow file-ioctl (regex #"^/dev/ttys[0-9]+"))
"""

SEATBELT_NETWORK_POLICY = """\
; Network access (only added when enabled)
(allow network-outbound)
(allow network-inbound)
(allow system-socket)

(allow mach-lookup
    (global-name "com.apple.bsd.dirhelper")
    (global-name "com.apple.system.opendirectoryd.membership")
    (global-name "com.apple.SecurityServer")
    (global-name "com.apple.networkd")
    (global-name "com.apple.ocspd")
    (global-name "com.apple.trustd.agent")
    (global-name "com.apple.SystemConfiguration.DNSConfiguration")
    (global-name "com.apple.SystemConfiguration.configd")
)

(allow file-write*
  (subpath (param "DARWIN_USER_CACHE_DIR"))
)
"""


def _get_darwin_user_cache_dir() -> Path | None:
    """Get the Darwin user cache directory."""
    # This is typically ~/Library/Caches
    cache_dir = os.environ.get("DARWIN_USER_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)

    # Fallback to common location
    home = Path.home()
    caches = home / "Library" / "Caches"
    if caches.exists():
        return caches
    return None


def _canonicalize_path(path: Path) -> Path:
    """Canonicalize path to avoid /var vs /private/var mismatches on macOS."""
    try:
        return path.resolve()
    except OSError:
        return path


def build_seatbelt_policy(policy: SandboxPolicy) -> tuple[str, list[tuple[str, str]]]:
    """Build SBPL policy string and parameter definitions.

    Returns:
        Tuple of (policy_string, [(param_name, param_value), ...])
    """
    params: list[tuple[str, str]] = []

    # Start with base policy
    full_policy = SEATBELT_BASE_POLICY

    # Add read access to entire filesystem
    full_policy += "\n; Allow read-only file operations\n(allow file-read*)\n"

    # Build write policy for writable roots
    writable_roots = policy.get_all_writable_roots()
    if writable_roots:
        write_rules = []

        for idx, root in enumerate(writable_roots):
            canonical_root = _canonicalize_path(root)
            root_param = f"WRITABLE_ROOT_{idx}"
            params.append((root_param, str(canonical_root)))

            # Check for read-only subpaths within this root
            ro_subpaths = [
                ro
                for ro in policy.read_only_paths
                if ro.is_relative_to(root) or root.is_relative_to(ro.parent)
            ]

            if not ro_subpaths:
                # Simple case: allow entire subpath
                write_rules.append(f'(subpath (param "{root_param}"))')
            else:
                # Complex case: allow subpath but exclude read-only paths
                require_parts = [f'(subpath (param "{root_param}"))']
                for ro_idx, ro_path in enumerate(ro_subpaths):
                    canonical_ro = _canonicalize_path(ro_path)
                    ro_param = f"WRITABLE_ROOT_{idx}_RO_{ro_idx}"
                    params.append((ro_param, str(canonical_ro)))
                    require_parts.append(f'(require-not (subpath (param "{ro_param}")))')

                write_rules.append(f"(require-all {' '.join(require_parts)})")

        # Also allow /tmp for temporary files
        tmp_path = _canonicalize_path(Path("/tmp"))
        tmp_param = f"WRITABLE_ROOT_{len(writable_roots)}"
        params.append((tmp_param, str(tmp_path)))
        write_rules.append(f'(subpath (param "{tmp_param}"))')

        full_policy += f"\n(allow file-write*\n  {' '.join(write_rules)}\n)\n"

    # Add network policy if enabled
    if policy.network_access:
        cache_dir = _get_darwin_user_cache_dir()
        if cache_dir:
            params.append(("DARWIN_USER_CACHE_DIR", str(_canonicalize_path(cache_dir))))
        full_policy += f"\n{SEATBELT_NETWORK_POLICY}"

    return full_policy, params


def build_sandbox_exec_args(
    command: str,
    policy: SandboxPolicy,
) -> list[str]:
    """Build arguments for sandbox-exec command.

    Returns list of arguments to pass to sandbox-exec.
    """
    sbpl_policy, params = build_seatbelt_policy(policy)

    args = ["-p", sbpl_policy]

    # Add parameter definitions
    for param_name, param_value in params:
        args.append(f"-D{param_name}={param_value}")

    # End of sandbox-exec args, start of command
    args.append("--")
    args.extend(["sh", "-c", command])

    return args


def is_sandbox_denied(stderr: str) -> bool:
    """Check if the error indicates sandbox denied the operation."""
    denied_indicators = [
        "Operation not permitted",
        "sandbox-exec:",
        "deny",
    ]
    return any(indicator in stderr for indicator in denied_indicators)


async def execute_with_seatbelt(
    command: str,
    policy: SandboxPolicy,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Execute a command under macOS Seatbelt sandbox.

    Args:
        command: Shell command to execute.
        policy: Sandbox policy defining restrictions.
        timeout: Command timeout in seconds.
        env: Optional additional environment variables.

    Returns:
        SandboxResult with execution results.

    Raises:
        SandboxError: If sandbox setup fails.
    """
    if not Path(SANDBOX_EXEC_PATH).exists():
        raise SandboxError(f"sandbox-exec not found at {SANDBOX_EXEC_PATH}")

    args = build_sandbox_exec_args(command, policy)

    # Build environment
    exec_env = os.environ.copy()
    exec_env["ROLLOUTS_SANDBOX"] = "seatbelt"
    if not policy.network_access:
        exec_env["ROLLOUTS_SANDBOX_NETWORK_DISABLED"] = "1"
    if env:
        exec_env.update(env)

    def run_sandboxed() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [SANDBOX_EXEC_PATH, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(policy.working_dir),
            env=exec_env,
        )

    try:
        result = await trio.to_thread.run_sync(run_sandboxed)

        sandbox_denied = result.returncode != 0 and is_sandbox_denied(result.stderr)

        # Extract denial reason from stderr if sandbox blocked
        denied_reason: str | None = None
        if sandbox_denied:
            stderr = result.stderr
            if "network" in stderr.lower():
                denied_reason = f"network access denied (command: {command[:100]})"
            elif "Operation not permitted" in stderr:
                # Try to extract the denied path from PermissionError
                # Pattern: "PermissionError: [Errno 1] Operation not permitted: '<path>'"
                path_match = re.search(
                    r"Operation not permitted:[\s\n]*['\"]([^'\"]+)['\"]", stderr
                )
                if path_match:
                    denied_path = path_match.group(1)
                    denied_reason = f"sandbox blocked write to {denied_path}"
                else:
                    denied_reason = f"filesystem access denied (command: {command[:100]})"
            else:
                denied_reason = f"access denied (command: {command[:100]})"

        return SandboxResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            sandbox_denied=sandbox_denied,
            denied_reason=denied_reason,
        )

    except subprocess.TimeoutExpired:
        return SandboxResult(
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            returncode=-1,
            sandbox_denied=False,
        )
