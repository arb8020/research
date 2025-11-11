"""High-level deployment API - composition layer.

This module composes midas/ (env setup) and kerbal/ (execution helpers)
into convenient high-level deployment operations.

Separation of concerns:
- midas/ - Python environment setup (uv, nix, docker) - "The Midas touch"
- kerbal/ - Remote execution helpers (tmux, gpu, env vars, transfer) - "Launch into production"
- deploy/ - Composition of midas + kerbal (THIS MODULE)

Casey Muratori principles:
- Granular operations (each function does one thing)
- Redundancy (high-level and low-level APIs for same operations)
- No coupling (functions don't depend on hidden state)

Tiger Style principles:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow

Usage patterns:

    # High-level (one function does everything):
    from deploy.api import deploy_and_run
    deploy_and_run(config, "root@host:22", "dev/speedrun")

    # Mid-level (more control):
    from deploy.api import deploy_project, run_in_project
    workspace = deploy_project(bifrost, "dev/speedrun", "dev-speedrun")
    run_in_project(bifrost, workspace, "python train.py")

    # Low-level (full control via kerbal/ and midas/):
    from kerbal import push_code, start_tmux_session
    from midas import UvBackend
    workspace = push_code(bifrost, "dev/speedrun")
    backend = UvBackend()
    backend.bootstrap(bifrost, workspace, "dev-speedrun")
    start_tmux_session(bifrost, "training", "python train.py")
"""

import logging

from midas import EnvBackend, CommandResult, UvBackend
from kerbal import (
    push_code,
    sync_results,
    start_tmux_session,
    check_gpus_available,
    wait_for_gpus,
)

logger = logging.getLogger(__name__)


# === Low-level operations (thin wrappers around midas + kerbal) ===


def bootstrap_env(
    bifrost: "BifrostClient",
    workspace: str,
    extra: str,
    backend: EnvBackend | None = None,
) -> None:
    """Bootstrap environment on remote machine.

    Casey: Granular operation - just setup env, nothing else.
    Tiger Style: < 70 lines, explicit.

    This is a thin wrapper around midas.EnvBackend.bootstrap().

    Args:
        bifrost: Connected bifrost client
        workspace: Remote workspace path (from push_code)
        extra: Python extra group (e.g., "dev-speedrun")
        backend: Environment backend (default: UvBackend())

    Example:
        bootstrap_env(bifrost, workspace, "dev-speedrun")
    """
    assert bifrost is not None, "bifrost client required"
    assert workspace, "workspace path required"
    assert extra, "Python extra required"

    if backend is None:
        backend = UvBackend()

    logger.info(f"🔧 Bootstrapping environment ({backend.__class__.__name__})...")

    backend.bootstrap(bifrost, workspace, extra)

    logger.info("✅ Environment ready")


# === Mid-level operations (convenient combinations) ===


def deploy_project(
    bifrost: "BifrostClient",
    local_path: str,
    extra: str,
    backend: EnvBackend | None = None,
) -> str:
    """Deploy project code and setup environment.

    Casey: Redundant API - bundles push_code + bootstrap_env for convenience.

    This is the "I don't want to think about it" function.
    Takes you from SSH connection to working Python environment.

    Args:
        bifrost: Connected bifrost client
        local_path: Local project path
        extra: Python extra group
        backend: Environment backend (default: UvBackend())

    Returns:
        Remote workspace path

    Example:
        workspace = deploy_project(bifrost, "dev/speedrun", "dev-speedrun")
    """
    # Granular operations composed
    workspace = push_code(bifrost, local_path)
    bootstrap_env(bifrost, workspace, extra, backend)
    return workspace


def run_in_project(
    bifrost: "BifrostClient",
    workspace: str,
    command: str,
    backend: EnvBackend | None = None,
    env_vars: dict[str, str] | None = None,
) -> CommandResult:
    """Run command in deployed project environment.

    Casey: Redundant API - convenience wrapper for backend.run_in_env().

    Args:
        bifrost: Connected bifrost client
        workspace: Remote workspace path
        command: Command to execute
        backend: Environment backend (default: UvBackend())
        env_vars: Environment variables to export (e.g., {"HF_TOKEN": "...", "CUDA_VISIBLE_DEVICES": "0,1"})

    Returns:
        CommandResult with exit_code, stdout, stderr

    Example:
        result = run_in_project(
            bifrost, workspace, "python train.py",
            env_vars={"CUDA_VISIBLE_DEVICES": "0"}
        )
        if result.success:
            print("Training complete!")
    """
    if backend is None:
        backend = UvBackend()

    return backend.run_in_env(bifrost, workspace, command, env_vars)


# === High-level orchestration (coarse-grained) ===


def deploy_and_run(
    bifrost: "BifrostClient",
    local_path: str,
    extra: str,
    command: str,
    backend: EnvBackend | None = None,
    detached: bool = False,
    session_name: str = "deploy",
    env_vars: dict[str, str] | None = None,
) -> CommandResult | None:
    """Deploy project and run command (highest level API).

    Casey: Coarse-grained API - bundles everything for maximum convenience.
    This is like the "update orientation" example from Casey's talk.

    Args:
        bifrost: Connected bifrost client
        local_path: Local project path
        extra: Python extra group
        command: Command to run
        backend: Environment backend (default: UvBackend())
        detached: If True, run in tmux and return immediately
        session_name: Tmux session name (if detached)
        env_vars: Environment variables to export (e.g., {"HF_TOKEN": "...", "CUDA_VISIBLE_DEVICES": "0,1"})

    Returns:
        CommandResult if not detached, None if detached

    Example:
        # Run and wait for completion
        result = deploy_and_run(
            bifrost, "dev/speedrun", "dev-speedrun", "python train.py",
            env_vars={"CUDA_VISIBLE_DEVICES": "0"}
        )

        # Run in background (detached)
        deploy_and_run(
            bifrost, "dev/speedrun", "dev-speedrun", "python train.py",
            detached=True,
            env_vars={"HF_TOKEN": "abc123"}
        )
    """
    # Deploy project
    workspace = deploy_project(bifrost, local_path, extra, backend)

    if detached:
        # Start in tmux and return immediately
        start_tmux_session(bifrost, session_name, command, workspace, env_vars=env_vars)
        logger.info(f"🎯 Running in tmux session: {session_name}")
        logger.info(f"   Attach: tmux attach -t {session_name}")
        return None
    else:
        # Run directly and wait
        logger.info(f"🚀 Running: {command}")
        result = run_in_project(bifrost, workspace, command, backend, env_vars)

        if result.success:
            logger.info("✅ Command completed successfully")
        else:
            logger.error(f"❌ Command failed with exit code {result.exit_code}")

        return result


# Re-export from kerbal/ for convenience
__all__ = [
    # Low-level from midas/
    "bootstrap_env",
    # Low-level from kerbal/
    "push_code",
    "sync_results",
    "start_tmux_session",
    "check_gpus_available",
    "wait_for_gpus",
    # Mid-level composition
    "deploy_project",
    "run_in_project",
    # High-level orchestration
    "deploy_and_run",
    # Types
    "CommandResult",
    "EnvBackend",
]
