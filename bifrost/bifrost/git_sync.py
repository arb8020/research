"""Pure functions for Git-based code deployment

No classes, just stateless functions that take inputs and return outputs.
All state managed by caller (BifrostClient).
"""

import paramiko
import logging
from typing import Optional, Union, List

from .types import RemoteConfig

logger = logging.getLogger(__name__)


def deploy_code(ssh_client: paramiko.SSHClient,
                config: RemoteConfig,
                workspace_path: str) -> str:
    """Deploy code via git to remote workspace.

    Args:
        ssh_client: Active SSH client connection
        config: Remote connection configuration
        workspace_path: Path to workspace on remote (e.g., ~/.bifrost/workspace)

    Returns:
        Path to deployed workspace

    Raises:
        RuntimeError: If deployment fails
    """
    # Assert inputs (Tiger Style)
    assert ssh_client is not None, "ssh_client cannot be None"
    assert isinstance(config, RemoteConfig), "config must be RemoteConfig"
    assert isinstance(workspace_path, str) and len(workspace_path) > 0, \
        "workspace_path must be non-empty string"

    logger.info(f"Deploying code to {workspace_path}")

    # Check if workspace exists
    stdin, stdout, stderr = ssh_client.exec_command(f"test -d {workspace_path}")
    workspace_exists = stdout.channel.recv_exit_status() == 0

    if workspace_exists:
        logger.debug(f"Workspace exists, updating...")
        # Update existing workspace via git pull
        _update_workspace(ssh_client, workspace_path)
    else:
        logger.debug(f"Workspace doesn't exist, creating...")
        # Create new workspace via git clone
        _create_workspace(ssh_client, workspace_path)

    # Assert output
    assert workspace_path, "Failed to deploy code"
    logger.info(f"Code deployed to {workspace_path}")
    return workspace_path


def run_bootstrap(ssh_client: paramiko.SSHClient,
                  config: RemoteConfig,
                  workspace_path: str,
                  bootstrap_cmd: Union[str, List[str]]) -> None:
    """Run bootstrap command(s) to prepare environment.

    Args:
        ssh_client: Active SSH client connection
        config: Remote connection configuration
        workspace_path: Path to workspace on remote
        bootstrap_cmd: Command(s) to run - either single string or list of commands
                      Each command runs in sequence, fails fast if any fails

    Raises:
        RuntimeError: If any bootstrap step fails
    """
    # Assert inputs (Tiger Style)
    assert ssh_client is not None, "ssh_client cannot be None"
    assert isinstance(config, RemoteConfig), "config must be RemoteConfig"
    assert isinstance(workspace_path, str) and len(workspace_path) > 0, \
        "workspace_path must be non-empty string"
    assert isinstance(bootstrap_cmd, (str, list)), \
        "bootstrap_cmd must be string or list of strings"

    # Normalize to list for uniform processing
    if isinstance(bootstrap_cmd, str):
        commands = [bootstrap_cmd]
    else:
        commands = bootstrap_cmd
        # Assert all items are strings
        assert all(isinstance(cmd, str) and len(cmd) > 0 for cmd in commands), \
            "All bootstrap commands must be non-empty strings"

    logger.info(f"Running {len(commands)} bootstrap step(s)...")

    # Execute each command in sequence
    for i, cmd in enumerate(commands, 1):
        # Log what we're doing (truncate long commands)
        cmd_preview = cmd[:60] + "..." if len(cmd) > 60 else cmd
        logger.info(f"Step {i}/{len(commands)}: {cmd_preview}")

        # Run command in workspace
        full_cmd = f"cd {workspace_path} && {cmd}"
        stdin, stdout, stderr = ssh_client.exec_command(full_cmd)

        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            error_output = stderr.read().decode()
            raise RuntimeError(
                f"Bootstrap step {i}/{len(commands)} failed with exit code {exit_code}: {error_output}"
            )

        logger.debug(f"Step {i}/{len(commands)} completed successfully")

    logger.info("All bootstrap steps completed successfully")


def _create_workspace(ssh_client: paramiko.SSHClient, workspace_path: str) -> None:
    """Create new workspace by cloning current git repo.

    Uses git bundle to transfer code without remote repo setup.
    """
    import subprocess
    import tempfile
    import os

    logger.debug("Creating git bundle of current repo...")

    # Create git bundle locally
    with tempfile.NamedTemporaryFile(suffix='.bundle', delete=False) as bundle_file:
        bundle_path = bundle_file.name

    try:
        # Create bundle from current HEAD
        result = subprocess.run(
            ['git', 'bundle', 'create', bundle_path, 'HEAD'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git bundle create failed: {result.stderr}")

        logger.debug("Uploading bundle to remote...")

        # Upload bundle to remote
        sftp = ssh_client.open_sftp()
        try:
            remote_bundle = f"/tmp/bifrost-bundle-{os.getpid()}.bundle"
            sftp.put(bundle_path, remote_bundle)
        finally:
            sftp.close()

        logger.debug("Cloning from bundle...")

        # Clone from bundle on remote
        stdin, stdout, stderr = ssh_client.exec_command(
            f"git clone {remote_bundle} {workspace_path} && rm {remote_bundle}"
        )

        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            error_output = stderr.read().decode()
            raise RuntimeError(f"Git clone from bundle failed: {error_output}")

        logger.debug("Workspace created successfully")

    finally:
        # Clean up local bundle
        if os.path.exists(bundle_path):
            os.unlink(bundle_path)


def _update_workspace(ssh_client: paramiko.SSHClient, workspace_path: str) -> None:
    """Update existing workspace with latest code.

    Uses git bundle to transfer changes.
    """
    import subprocess
    import tempfile
    import os

    logger.debug("Creating git bundle of current repo...")

    # Create git bundle locally
    with tempfile.NamedTemporaryFile(suffix='.bundle', delete=False) as bundle_file:
        bundle_path = bundle_file.name

    try:
        # Create bundle from current HEAD
        result = subprocess.run(
            ['git', 'bundle', 'create', bundle_path, 'HEAD'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git bundle create failed: {result.stderr}")

        logger.debug("Uploading bundle to remote...")

        # Upload bundle to remote
        sftp = ssh_client.open_sftp()
        try:
            remote_bundle = f"/tmp/bifrost-bundle-{os.getpid()}.bundle"
            sftp.put(bundle_path, remote_bundle)
        finally:
            sftp.close()

        logger.debug("Updating workspace from bundle...")

        # Fetch and reset from bundle
        update_cmd = f"""
cd {workspace_path} &&
git fetch {remote_bundle} HEAD &&
git reset --hard FETCH_HEAD &&
rm {remote_bundle}
"""
        stdin, stdout, stderr = ssh_client.exec_command(update_cmd)

        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            error_output = stderr.read().decode()
            raise RuntimeError(f"Git update from bundle failed: {error_output}")

        logger.debug("Workspace updated successfully")

    finally:
        # Clean up local bundle
        if os.path.exists(bundle_path):
            os.unlink(bundle_path)
