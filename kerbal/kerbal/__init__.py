"""Kerbal package - Script execution orchestration for remote machines.

Like Kerbal Space Program, this package helps you orchestrate complex remote operations
with modular, composable components. Launch your code into production!

This package provides utilities for remote script execution:
- python_env: Python dependency setup and script execution (uv-based)
- tmux: Process management (sessions, detached execution)
- gpu: Hardware checking (nvidia-smi queries, availability)
- env: Environment variable helpers (export building)
- transfer: File transfer (push, sync)
- protocol: Data structures (DependencyConfig, CommandResult)
- job_monitor: Real-time log streaming and job monitoring

These are orthogonal concerns that can be composed together.

Example - Basic deployment:
    from bifrost import BifrostClient
    from kerbal import (
        DependencyConfig,
        setup_script_deps,
        start_tmux_session,
        LogStreamConfig,
        stream_log_until_complete,
    )

    client = BifrostClient("root@gpu:22", ssh_key_path="~/.ssh/id_rsa")

    # Setup Python environment
    deps = DependencyConfig(
        project_name="training",
        dependencies=["torch>=2.0"],
        extras={"training": ["wandb"]},
    )
    setup_script_deps(client, workspace, deps, install_extras=["training"])

    # Run in tmux with real-time monitoring
    session, err = start_tmux_session(
        client, "training-job",
        f"cd {workspace} && .venv/bin/python train.py",
        log_file=f"{workspace}/train.log",
        env_vars={"CUDA_VISIBLE_DEVICES": "0,1"}
    )
    if err:
        print(f"Failed: {err}")
        return

    # Monitor with real-time log streaming
    config = LogStreamConfig(
        session_name=session,
        log_file=f"{workspace}/train.log",
        timeout_sec=7200,  # 2 hours
    )
    success, exit_code, err = stream_log_until_complete(client, config)
    if not success:
        print(f"Job failed: {err}")
"""

from kerbal.protocol import DependencyConfig, CommandResult
from kerbal.python_env import setup_script_deps, run_script
from kerbal.tmux import start_tmux_session
from kerbal.gpu import check_gpus_available, wait_for_gpus
from kerbal.env import build_env_prefix
from kerbal.transfer import push_code, sync_results
from kerbal.job_monitor import (
    LogStreamConfig,
    stream_log_until_complete,
    stream_log_with_condition,
)

__all__ = [
    "DependencyConfig",
    "CommandResult",
    "setup_script_deps",
    "run_script",
    "start_tmux_session",
    "check_gpus_available",
    "wait_for_gpus",
    "build_env_prefix",
    "push_code",
    "sync_results",
    "LogStreamConfig",
    "stream_log_until_complete",
    "stream_log_with_condition",
]
