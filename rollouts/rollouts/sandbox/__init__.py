"""Sandbox execution for untrusted commands.

OS-level sandboxing for bash command execution, protecting users from
accidental or malicious agent actions.

Supported platforms:
- macOS: Uses Seatbelt (sandbox-exec) with SBPL policies
- Linux: Uses Landlock LSM for filesystem isolation (kernel 5.13+)
- Windows: Not yet supported (will fail with clear error)

Implementation based on OpenAI Codex CLI (MIT License):
https://github.com/openai/codex/tree/main/codex-rs

Usage:
    from rollouts.sandbox import SandboxPolicy, execute_sandboxed

    policy = SandboxPolicy.workspace_write(working_dir)
    result = await execute_sandboxed(command, policy)
"""

from rollouts.sandbox.executor import (
    SandboxError,
    SandboxResult,
    SandboxUnavailableError,
    execute_sandboxed,
    execute_unsandboxed,
    is_sandbox_available,
)
from rollouts.sandbox.policy import SandboxMode, SandboxPolicy

__all__ = [
    "SandboxMode",
    "SandboxPolicy",
    "SandboxResult",
    "SandboxError",
    "SandboxUnavailableError",
    "execute_sandboxed",
    "execute_unsandboxed",
    "is_sandbox_available",
]
