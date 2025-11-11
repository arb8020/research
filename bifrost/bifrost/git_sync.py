"""Pure functions for Git-based code deployment

No classes, just stateless functions that take inputs and return outputs.
All state managed by caller (BifrostClient).
"""

import paramiko
import logging
from typing import Optional, Union, List

from .types import RemoteConfig
from shared.retry import retry

logger = logging.getLogger(__name__)


@retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,))
def _upload_bundle_with_retry(sftp, local_bundle_path: str, remote_bundle_path: str) -> None:
    """Upload git bundle with retry logic.

    This is at an external boundary (network I/O over SFTP) so retry is appropriate.
    Retries up to 3 times with exponential backoff (1s, 2s, 4s) on any exception.

    Args:
        sftp: Active SFTP client
        local_bundle_path: Path to local bundle file
        remote_bundle_path: Path to remote bundle file

    Raises:
        Exception: If upload fails after all retry attempts
    """
    import os

    # Tiger Style: Assert inputs
    assert sftp is not None, "sftp client cannot be None"
    assert os.path.exists(local_bundle_path), f"Local bundle not found: {local_bundle_path}"
    assert isinstance(remote_bundle_path, str) and len(remote_bundle_path) > 0, \
        "remote_bundle_path must be non-empty string"

    # Upload the bundle
    sftp.put(local_bundle_path, remote_bundle_path)

    logger.debug(f"Bundle uploaded successfully to {remote_bundle_path}")


def _check_untracked_files() -> Optional[List[str]]:
    """Check for untracked files in the git repo.

    Returns:
        List of untracked files, or None if not in a git repo
    """
    import subprocess

    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            return None  # Not a git repo

        # Parse git status output
        # Untracked files start with '??'
        untracked = []
        for line in result.stdout.splitlines():
            if line.startswith('??'):
                # Extract filename (remove '?? ' prefix)
                filename = line[3:].strip()
                untracked.append(filename)

        return untracked if untracked else None

    except Exception:
        return None  # Git command failed


def _check_uncommitted_changes() -> Optional[List[str]]:
    """Check for uncommitted changes (modified, staged, or deleted files).

    Returns:
        List of uncommitted files, or None if not in a git repo
    """
    import subprocess

    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            return None  # Not a git repo

        # Parse git status output
        # Modified/staged files have status codes in first 2 columns (not '??')
        # Examples: ' M' (modified), 'M ' (staged), 'MM' (staged + modified)
        uncommitted = []
        for line in result.stdout.splitlines():
            if line and not line.startswith('??'):
                # Extract filename (skip first 3 chars: status codes + space)
                filename = line[3:].strip()
                uncommitted.append(filename)

        return uncommitted if uncommitted else None

    except Exception:
        return None  # Git command failed


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

    # Check for untracked files and warn user
    untracked = _check_untracked_files()
    if untracked:
        logger.warning(f"⚠️  Found {len(untracked)} untracked file(s) that will NOT be deployed:")
        for file in untracked[:5]:  # Show first 5
            logger.warning(f"  - {file}")
        if len(untracked) > 5:
            logger.warning(f"  ... and {len(untracked) - 5} more")
        logger.warning("💡 Tip: Use 'git add' to track these files, or add them to .gitignore")

    # Check for uncommitted changes and warn user
    uncommitted = _check_uncommitted_changes()
    if uncommitted:
        logger.warning(f"⚠️  Found {len(uncommitted)} uncommitted change(s) that will NOT be deployed:")
        for file in uncommitted[:5]:  # Show first 5
            logger.warning(f"  - {file}")
        if len(uncommitted) > 5:
            logger.warning(f"  ... and {len(uncommitted) - 5} more")
        logger.warning("💡 Tip: Use 'git commit' to include these changes in deployment")

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

        # Upload bundle to remote with retry logic
        sftp = ssh_client.open_sftp()
        try:
            remote_bundle = f"/tmp/bifrost-bundle-{os.getpid()}.bundle"
            _upload_bundle_with_retry(sftp, bundle_path, remote_bundle)
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

    # Check for untracked files and warn user
    untracked = _check_untracked_files()
    if untracked:
        logger.warning(f"⚠️  Found {len(untracked)} untracked file(s) that will NOT be deployed:")
        for file in untracked[:5]:  # Show first 5
            logger.warning(f"  - {file}")
        if len(untracked) > 5:
            logger.warning(f"  ... and {len(untracked) - 5} more")
        logger.warning("💡 Tip: Use 'git add' to track these files, or add them to .gitignore")

    # Check for uncommitted changes and warn user
    uncommitted = _check_uncommitted_changes()
    if uncommitted:
        logger.warning(f"⚠️  Found {len(uncommitted)} uncommitted change(s) that will NOT be deployed:")
        for file in uncommitted[:5]:  # Show first 5
            logger.warning(f"  - {file}")
        if len(uncommitted) > 5:
            logger.warning(f"  ... and {len(uncommitted) - 5} more")
        logger.warning("💡 Tip: Use 'git commit' to include these changes in deployment")

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

        # Upload bundle to remote with retry logic
        sftp = ssh_client.open_sftp()
        try:
            remote_bundle = f"/tmp/bifrost-bundle-{os.getpid()}.bundle"
            _upload_bundle_with_retry(sftp, bundle_path, remote_bundle)
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
