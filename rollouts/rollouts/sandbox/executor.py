"""Sandbox executor with platform detection.

Routes to the appropriate platform-specific sandbox implementation.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import trio

from rollouts.sandbox.policy import SandboxPolicy


@dataclass
class SandboxResult:
    """Result of a sandboxed command execution."""

    stdout: str
    stderr: str
    returncode: int
    sandbox_denied: bool = False  # True if sandbox blocked the operation
    denied_reason: str | None = None  # Human-readable reason if sandbox_denied


class SandboxError(Exception):
    """Raised when sandbox setup or execution fails."""

    pass


class SandboxUnavailableError(SandboxError):
    """Raised when sandboxing is not available on this platform."""

    pass


def get_platform() -> str:
    """Get the current platform identifier."""
    if sys.platform == "darwin":
        return "macos"
    elif sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform == "win32":
        return "windows"
    else:
        return "unknown"


def is_sandbox_available() -> bool:
    """Check if sandboxing is available on the current platform."""
    platform = get_platform()

    if platform == "macos":
        # Check if sandbox-exec exists
        return Path("/usr/bin/sandbox-exec").exists()

    elif platform == "linux":
        # Check if Landlock is available (kernel 5.13+)
        # We check by looking at /sys/kernel/security/landlock
        return Path("/sys/kernel/security/landlock").exists()

    elif platform == "windows":
        # Windows sandbox not yet implemented
        return False

    return False


def get_sandbox_unavailable_reason() -> str | None:
    """Get a human-readable reason why sandbox is unavailable, or None if available."""
    platform = get_platform()

    if platform == "macos":
        if not Path("/usr/bin/sandbox-exec").exists():
            return "macOS sandbox-exec not found at /usr/bin/sandbox-exec"
        return None

    elif platform == "linux":
        if not Path("/sys/kernel/security/landlock").exists():
            return (
                "Linux Landlock not available. "
                "Requires kernel 5.13+ with CONFIG_SECURITY_LANDLOCK=y"
            )
        return None

    elif platform == "windows":
        return "Windows sandboxing not yet implemented"

    return f"Unknown platform: {sys.platform}"


async def execute_sandboxed(
    command: str,
    policy: SandboxPolicy,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Execute a command inside a sandbox.

    Args:
        command: The shell command to execute.
        policy: Sandbox policy defining restrictions.
        timeout: Command timeout in seconds.
        env: Optional environment variables (merged with current env).

    Returns:
        SandboxResult with stdout, stderr, returncode, and sandbox_denied flag.

    Raises:
        SandboxUnavailableError: If sandboxing is not available on this platform.
        SandboxError: If sandbox setup fails.
    """
    platform = get_platform()

    if platform == "macos":
        from rollouts.sandbox.seatbelt import execute_with_seatbelt

        return await execute_with_seatbelt(command, policy, timeout, env)

    elif platform == "linux":
        from rollouts.sandbox.landlock import execute_with_landlock

        return await execute_with_landlock(command, policy, timeout, env)

    elif platform == "windows":
        raise SandboxUnavailableError(
            "Windows sandboxing not yet implemented. "
            "Use --no-sandbox to run without sandboxing (at your own risk)."
        )

    else:
        raise SandboxUnavailableError(f"Sandboxing not supported on platform: {sys.platform}")


async def execute_unsandboxed(
    command: str,
    working_dir: Path,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Execute a command WITHOUT sandboxing.

    This should only be used when the user explicitly opts out with --no-sandbox.

    Args:
        command: The shell command to execute.
        working_dir: Working directory for execution.
        timeout: Command timeout in seconds.
        env: Optional environment variables.

    Returns:
        SandboxResult with stdout, stderr, and returncode.
    """
    import os

    exec_env = os.environ.copy()
    if env:
        exec_env.update(env)

    def run_command() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(working_dir),
            env=exec_env,
        )

    try:
        result = await trio.to_thread.run_sync(run_command)
        return SandboxResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            sandbox_denied=False,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            returncode=-1,
            sandbox_denied=False,
        )
