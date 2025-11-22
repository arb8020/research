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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bifrost import BifrostClient

logger = logging.getLogger(__name__)


def push_code(
    client: "BifrostClient",
    local_path: str = ".",
    remote_path: str | None = None,
    exclude: list[str] | None = None,
) -> str:
    """Push code to remote machine (client primitive wrapper).

    Casey: Granular operation - just push code, nothing else.

    Args:
        client: BifrostClient instance for SSH operations
        local_path: Local path to push (default: current directory)
        remote_path: Remote path (default: ~/.bifrost/workspace)
        exclude: Patterns to exclude from sync (e.g., ["*.pyc", ".git/**"])

    Returns:
        Absolute remote workspace path

    Example:
        workspace = push_code(client, "dev/integration_training")
    """
    assert bifrost is not None, "BifrostClient instance required"

    logger.debug(f"pushing code from {local_path}...")

    # Use bifrost's push with excludes if provided
    if exclude:
        # TODO: client.push needs to support exclude parameter
        # For now, just use default push
        logger.warning("Exclude patterns not yet supported by client.push()")

    workspace = client.push(local_path=local_path, remote_path=remote_path)

    logger.info(f"code pushed to {workspace}")
    return workspace


def sync_results(
    client: "BifrostClient",
    remote_path: str,
    local_path: str,
) -> None:
    """Download results from remote to local.

    Casey: Granular operation - just sync files, nothing else.
    Tiger Style: < 70 lines.

    Args:
        client: BifrostClient instance for SSH operations
        remote_path: Remote path to download from
        local_path: Local path to download to

    Example:
        sync_results(client, "results/exp_123", "./local_results")
    """
    assert bifrost is not None, "BifrostClient instance required"
    assert remote_path, "remote path required"
    assert local_path, "local path required"

    logger.debug(f"syncing results from {remote_path}...")

    # Create local directory
    Path(local_path).mkdir(parents=True, exist_ok=True)

    # Download using bifrost
    result = client.download_files(
        remote_path=remote_path,
        local_path=local_path,
        recursive=True,
    )

    if result and result.success:
        logger.info(f"results synced to {local_path}")
    else:
        logger.warning("⚠️  Some files may not have synced")
