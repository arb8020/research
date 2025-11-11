"""File transfer helpers for remote execution.

This module handles pushing code to and syncing results from remote machines.
Purely about file transfer - no knowledge of Python envs or deployments.

Tiger Style:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def push_code(
    bifrost: "BifrostClient",
    local_path: str = ".",
    remote_path: str | None = None,
    exclude: list[str] | None = None,
) -> str:
    """Push code to remote machine (bifrost primitive wrapper).

    Casey: Granular operation - just push code, nothing else.

    Args:
        bifrost: Connected bifrost client
        local_path: Local path to push (default: current directory)
        remote_path: Remote path (default: ~/.bifrost/workspace)
        exclude: Patterns to exclude from sync (e.g., ["*.pyc", ".git/**"])

    Returns:
        Absolute remote workspace path

    Example:
        workspace = push_code(bifrost, "dev/integration_training")
    """
    assert bifrost is not None, "bifrost client required"

    logger.info(f"📦 Pushing code from {local_path}...")

    # Use bifrost's push with excludes if provided
    if exclude:
        # TODO: bifrost.push needs to support exclude parameter
        # For now, just use default push
        logger.warning("Exclude patterns not yet supported by bifrost.push()")

    workspace = bifrost.push(local_path=local_path, remote_path=remote_path)

    logger.info(f"✅ Code pushed to {workspace}")
    return workspace


def sync_results(
    bifrost: "BifrostClient",
    remote_path: str,
    local_path: str,
) -> None:
    """Download results from remote to local.

    Casey: Granular operation - just sync files, nothing else.
    Tiger Style: < 70 lines.

    Args:
        bifrost: Connected bifrost client
        remote_path: Remote path to download from
        local_path: Local path to download to

    Example:
        sync_results(bifrost, "results/exp_123", "./local_results")
    """
    assert bifrost is not None, "bifrost client required"
    assert remote_path, "remote path required"
    assert local_path, "local path required"

    logger.info(f"💾 Syncing results from {remote_path}...")

    # Create local directory
    Path(local_path).mkdir(parents=True, exist_ok=True)

    # Download using bifrost
    result = bifrost.download_files(
        remote_path=remote_path,
        local_path=local_path,
        recursive=True,
    )

    if result and result.success:
        logger.info(f"✅ Results synced to {local_path}")
    else:
        logger.warning("⚠️  Some files may not have synced")
