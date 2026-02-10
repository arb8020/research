"""
Sandboxed Git Worktree Environment - isolated git-based coding with OS-level sandboxing.

Combines:
1. GitWorktreeEnvironment - worktree per session, auto-commit history
2. OS-level sandbox - Seatbelt (macOS) / Landlock (Linux)

The agent can only write within its worktree. Bash commands are sandboxed
to prevent escaping to parent directories or sensitive paths.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import trio

from ..dtypes import (
    AgentState,
    RunConfig,
    Tool,
    ToolCall,
    ToolResult,
)
from ..sandbox import SandboxPolicy, execute_sandboxed, is_sandbox_available
from .git_worktree import GitWorktreeEnvironment

SANDBOX_AVAILABLE = is_sandbox_available()


@dataclass
class SandboxedWorktreeEnvironment(GitWorktreeEnvironment):
    """Git worktree environment with OS-level sandboxing for bash commands.

    Inherits all behavior from GitWorktreeEnvironment:
    - Creates .rollouts/worktrees/<session_id>/ for isolation
    - Auto-commits every write/edit/bash
    - Full git history for restore

    Adds:
    - OS-level sandbox for bash (Seatbelt on macOS, Landlock on Linux)
    - Restricts writes to worktree only
    - Optional network access (default: enabled for pip/git)
    """

    network_access: bool = True
    sandbox_enabled: bool = True

    # Extra paths the agent can write to (beyond worktree)
    extra_writable: list[Path] = field(default_factory=list)

    # Extra environment variables to set for bash commands
    extra_env: dict[str, str] = field(default_factory=dict)

    def get_name(self) -> str:
        return "sandboxed-worktree"

    def get_status_info(self) -> dict[str, str] | None:
        info = super().get_status_info() or {}
        if self.sandbox_enabled and SANDBOX_AVAILABLE:
            info["sandbox"] = "on"
        elif self.sandbox_enabled and not SANDBOX_AVAILABLE:
            info["sandbox"] = "unavailable"
        else:
            info["sandbox"] = "off"
        return info

    async def _exec_bash(
        self,
        tool_call: ToolCall,
        work_dir: Path,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Execute bash command with OS-level sandboxing."""
        command = tool_call.args["command"]
        timeout = tool_call.args.get("timeout", 120)

        # If sandbox not available or disabled, fall back to unsandboxed
        if not self.sandbox_enabled or not SANDBOX_AVAILABLE:
            return await super()._exec_bash(tool_call, work_dir, cancel_scope)

        # Build sandbox policy
        policy = SandboxPolicy.workspace_write(
            working_dir=work_dir,
            extra_writable=self.extra_writable or None,
            network_access=self.network_access,
        )

        try:
            result = await execute_sandboxed(
                command, policy, timeout=timeout, env=self.extra_env or None
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += result.stderr

            # Check if sandbox blocked the operation
            if result.sandbox_denied:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content=output or "(no output)",
                    error=f"Sandbox blocked: {result.denied_reason}",
                )

            if result.returncode != 0:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content=output or "(no output)",
                    error=f"Exit code {result.returncode}",
                )

            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=False,
                content=output or "(no output)",
            )

        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Sandbox error: {e}",
            )

    async def serialize(self) -> dict:
        """Capture current state for session persistence."""
        base = await super().serialize()
        base["env_kind"] = "sandboxed_worktree"
        base["network_access"] = self.network_access
        base["sandbox_enabled"] = self.sandbox_enabled
        base["extra_writable"] = [str(p) for p in self.extra_writable]
        base["extra_env"] = self.extra_env
        return base

    @staticmethod
    async def deserialize(data: dict) -> "SandboxedWorktreeEnvironment":
        """Restore environment from serialized state."""
        env = SandboxedWorktreeEnvironment(
            working_dir=Path(data["working_dir"]),
            network_access=data.get("network_access", True),
            sandbox_enabled=data.get("sandbox_enabled", True),
            extra_writable=[Path(p) for p in data.get("extra_writable", [])],
            extra_env=data.get("extra_env", {}),
        )

        session_id = data.get("session_id")
        if session_id:
            await env.on_session_start(session_id)

            head_commit = data.get("head_commit")
            if head_commit and env._worktree_path:
                try:
                    await env._run_git(["checkout", head_commit])
                except Exception:
                    pass

            env._commit_count = data.get("commit_count", 0)

        return env


def create_sandboxed_environment(
    source_dir: Path,
    workspace_base: Path | None = None,
    run_id: str | None = None,
    network_access: bool = True,
    sandbox_enabled: bool = True,
    extra_env: dict[str, str] | None = None,
) -> SandboxedWorktreeEnvironment:
    """Factory to create a sandboxed environment from a source directory.

    Copies source_dir to a fresh workspace, then wraps it in a sandboxed
    git worktree environment.

    Args:
        source_dir: Directory to copy (e.g., tinygrad-nometal/)
        workspace_base: Base directory for workspaces (default: /tmp)
        run_id: Unique identifier for this run (auto-generated if None)
        network_access: Allow network access (default: True for pip/git)
        sandbox_enabled: Enable OS-level sandbox (default: True)
        extra_env: Extra environment variables for bash commands

    Returns:
        SandboxedWorktreeEnvironment ready for use
    """
    import uuid

    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    if workspace_base is None:
        workspace_base = Path("/tmp/rollouts-workspaces")

    workspace_base.mkdir(parents=True, exist_ok=True)
    workspace = workspace_base / f"workspace-{run_id}"

    # Copy source to workspace
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(source_dir, workspace)

    return SandboxedWorktreeEnvironment(
        working_dir=workspace,
        network_access=network_access,
        sandbox_enabled=sandbox_enabled,
        extra_env=extra_env or {},
    )
