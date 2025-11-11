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
    env_vars: dict[str, str] | None = None,
) -> None:
    """Start a tmux session running a command.

    Casey: Granular operation - just start tmux, nothing else.
    Tiger Style: < 70 lines.

    Args:
        client: BifrostClient instance for SSH operations
        session_name: Tmux session name
        command: Command to run in tmux
        workspace: Working directory (optional)
        log_file: Path to log file (optional, for capturing output)
        env_vars: Environment variables to export (e.g., {"HF_TOKEN": "...", "CUDA_VISIBLE_DEVICES": "0,1"})

    Example:
        from bifrost import BifrostClient
        client = BifrostClient("root@gpu:22", ssh_key_path="~/.ssh/id_rsa")
        start_tmux_session(
            client, "training", "python train.py",
            workspace=workspace,
            env_vars={"CUDA_VISIBLE_DEVICES": "0,1"}
        )
    """
    assert client is not None, "BifrostClient instance required"
    assert session_name, "session name required"
    assert command, "command required"

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
        # Capture output to log file
        tmux_cmd += f" '{full_command} 2>&1 | tee {log_file}'"
    else:
        tmux_cmd += f" '{full_command}'"

    result = client.exec(tmux_cmd)
    assert result.exit_code == 0, f"Failed to start tmux session: {result.stderr}"

    logger.info(f"✅ Tmux session started: {session_name}")
