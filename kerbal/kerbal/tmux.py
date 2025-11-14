"""Tmux session management for remote execution.

This module handles process management via tmux sessions.
Purely about tmux - no knowledge of Python envs, GPUs, or deployments.

Tiger Style:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bifrost import BifrostClient

logger = logging.getLogger(__name__)


def start_tmux_session(
    client: "BifrostClient",
    session_name: str,
    command: str,
    workspace: str | None = None,
    log_file: str | None = None,
    capture_exit_code: bool = True,
    env_vars: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    """Start a tmux session running a command.

    Casey: Granular operation - just start tmux, nothing else.
    Tiger Style: < 70 lines, tuple return for error.

    Why use 'script' command for logging:
    - Properly captures ALL terminal output (not just stdout/stderr)
    - Guarantees output is flushed before process exit (via -f flag)
    - Preserves actual exit code of command (via -e flag)
    - Prevents log file from missing output on fast-exiting processes
    - Pattern from production systems requiring reliable log capture

    Why append exit code marker:
    - Enables programmatic monitoring of job completion
    - Works even when tmux session dies unexpectedly
    - Pattern from clicker:170a21c for reliable exit detection

    Args:
        client: BifrostClient instance for SSH operations
        session_name: Tmux session name
        command: Command to run in tmux
        workspace: Working directory (optional)
        log_file: Path to log file (optional, for capturing output)
        capture_exit_code: Append "EXIT_CODE: N" marker to log (default: True)
        env_vars: Environment variables to export (e.g., {"HF_TOKEN": "...", "CUDA_VISIBLE_DEVICES": "0,1"})

    Returns:
        (session_name, error_message)
        error_message is None on success

    Example:
        from bifrost import BifrostClient
        client = BifrostClient("root@gpu:22", ssh_key_path="~/.ssh/id_rsa")
        session, err = start_tmux_session(
            client, "training", "python train.py",
            workspace=workspace,
            log_file="train.log",
            env_vars={"CUDA_VISIBLE_DEVICES": "0,1"}
        )
        if err:
            print(f"Failed: {err}")
    """
    # Tiger Style: Assert preconditions
    assert client is not None, "BifrostClient instance required"
    assert session_name, "session name required"
    assert command, "command required"
    assert isinstance(session_name, str), "session_name must be string"
    assert isinstance(command, str), "command must be string"
    if workspace is not None:
        assert isinstance(workspace, str), "workspace must be string"
    if log_file is not None:
        assert isinstance(log_file, str), "log_file must be string"
    if env_vars is not None:
        assert isinstance(env_vars, dict), "env_vars must be dict"
    assert isinstance(capture_exit_code, bool), "capture_exit_code must be bool"

    logger.info(f"🚀 Starting tmux session: {session_name}")

    # Kill existing session if it exists
    client.exec(f"tmux kill-session -t {session_name} 2>/dev/null || true")

    # Build env prefix if needed
    env_prefix = ""
    if env_vars:
        from kerbal.env import build_env_prefix
        env_prefix = build_env_prefix(env_vars)

    # Build tmux command
    tmux_cmd = f"tmux new-session -d -s {session_name}"

    # Add working directory if specified
    if workspace:
        tmux_cmd += f" -c {workspace}"

    # Add the command to run (with env vars if specified)
    full_command = env_prefix + command
    if log_file:
        # Use 'script' command for reliable output capture
        # Why script instead of tee:
        # - script captures ALL terminal output (including terminal control codes)
        # - -c: run command directly
        # - -e: return exit code of child process
        # - -f: flush output immediately (prevents buffering issues)
        # This ensures fast-exiting processes don't lose output
        # Note: util-linux script doesn't have -q flag, uses 'Script started' message
        if capture_exit_code:
            # Full capture with exit code marker
            tmux_cmd += f" 'script -efc \"{full_command}\" {log_file}; echo EXIT_CODE: $? >> {log_file}'"
        else:
            # Casey: Granularity - can run without exit code marker
            tmux_cmd += f" 'script -efc \"{full_command}\" {log_file}'"
    else:
        tmux_cmd += f" '{full_command}'"

    result = client.exec(tmux_cmd)
    if result.exit_code != 0:
        return session_name, f"Failed to start tmux: {result.stderr}"

    logger.info(f"✅ Tmux session started: {session_name}")
    return session_name, None
