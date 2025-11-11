"""Deployment and environment setup for remote execution.

This package provides environment setup (getting from SSH to working code)
and deployment orchestration. It sits between bifrost (SSH primitives) and
your application code.

Architecture:
    broker/    - GPU provisioning
    bifrost/   - SSH primitives (connect, exec, push, download)
    deploy/    - Environment setup + orchestration (this package)
    miniray/   - Multi-node coordination (future)

Usage:
    from deploy.api import deploy_project, run_in_project
    from deploy.backends.uv import UvBackend

    workspace = deploy_project(bifrost, "dev/my-project", "dev-my-project")
    run_in_project(bifrost, workspace, "python train.py")
"""

from deploy.api import (
    deploy_project,
    run_in_project,
    start_tmux_session,
    sync_results,
)

__all__ = [
    "deploy_project",
    "run_in_project",
    "start_tmux_session",
    "sync_results",
]
