"""Pure functions for Git-based code deployment

No classes, just stateless functions that take inputs and return outputs.
All state managed by caller (BifrostClient).
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import paramiko
from infra_utils.retry import retry

from .types import RemoteConfig

logger = logging.getLogger(__name__)

# Default timeout for the entire deploy_code() call (seconds)
DEPLOY_TIMEOUT_SECONDS = 300  # 5 minutes


@retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,))
def _upload_bundle_with_retry(
    sftp: paramiko.SFTPClient, local_bundle_path: str, remote_bundle_path: str
) -> None:
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
    assert isinstance(remote_bundle_path, str) and len(remote_bundle_path) > 0, (
        "remote_bundle_path must be non-empty string"
    )

    bundle_size = os.path.getsize(local_bundle_path)
    logger.info(f"uploading bundle ({bundle_size:,} bytes) to {remote_bundle_path}")

    t0 = time.monotonic()
    bytes_transferred = [0]

    def _progress(transferred: int, total: int) -> None:
        bytes_transferred[0] = transferred
        # Log every ~10% or every 5MB, whichever comes first
        if total > 0 and (
            transferred == total or transferred % max(total // 10, 5 * 1024 * 1024) < 32768
        ):
            elapsed = time.monotonic() - t0
            pct = 100 * transferred / total
            rate_mbps = (transferred / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            logger.debug(
                f"  upload: {transferred:,}/{total:,} bytes ({pct:.0f}%) {rate_mbps:.1f} MB/s"
            )

    sftp.put(local_bundle_path, remote_bundle_path, callback=_progress)

    elapsed = time.monotonic() - t0
    rate_mbps = (bundle_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
    logger.info(f"bundle uploaded: {bundle_size:,} bytes in {elapsed:.1f}s ({rate_mbps:.1f} MB/s)")


def _check_untracked_files(repo_root: str | None = None) -> list[str] | None:
    """Check for untracked files in the git repo.

    Returns:
        List of untracked files, or None if not in a git repo
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            cwd=repo_root,
        )

        if result.returncode != 0:
            return None  # Not a git repo

        # Parse git status output
        # Untracked files start with '??'
        untracked = []
        for line in result.stdout.splitlines():
            if line.startswith("??"):
                # Extract filename (remove '?? ' prefix)
                filename = line[3:].strip()
                untracked.append(filename)
    except Exception:
        return None  # Git command failed
    else:
        return untracked if untracked else None


def _check_uncommitted_changes(repo_root: str | None = None) -> list[str] | None:
    """Check for uncommitted changes (modified, staged, or deleted files).

    Returns:
        List of uncommitted files, or None if not in a git repo
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            cwd=repo_root,
        )

        if result.returncode != 0:
            return None  # Not a git repo

        # Parse git status output
        # Modified/staged files have status codes in first 2 columns (not '??')
        # Examples: ' M' (modified), 'M ' (staged), 'MM' (staged + modified)
        uncommitted = []
        for line in result.stdout.splitlines():
            if line and not line.startswith("??"):
                # Extract filename (skip first 3 chars: status codes + space)
                filename = line[3:].strip()
                uncommitted.append(filename)
    except Exception:
        return None  # Git command failed
    else:
        return uncommitted if uncommitted else None


def ensure_clean_git_repo(*, repo_root: str, allow_dirty: bool) -> None:
    """Fail fast if a local git repo is dirty and allow_dirty is false."""

    assert repo_root, "repo_root must be non-empty string"
    if allow_dirty:
        return

    untracked = _check_untracked_files(repo_root)
    uncommitted = _check_uncommitted_changes(repo_root)
    if untracked or uncommitted:
        dirty_files = []
        if untracked:
            dirty_files.extend(untracked)
        if uncommitted:
            dirty_files.extend(uncommitted)
        raise RuntimeError(
            f"Workspace has {len(dirty_files)} uncommitted/untracked file(s). "
            f"Deploy with allow_dirty=True to proceed anyway. "
            f"Files: {', '.join(dirty_files[:5])}"
            + (f" and {len(dirty_files) - 5} more" if len(dirty_files) > 5 else "")
        )


def create_git_archive(repo_root: str) -> tuple[str, str]:
    """Create a committed-only tarball snapshot of HEAD for a git repo."""

    import subprocess

    resolved_root = str(Path(repo_root).expanduser().resolve())
    assert resolved_root, "repo_root must resolve to a path"

    hash_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=resolved_root,
    )
    commit_hash = hash_result.stdout.strip()
    if hash_result.returncode != 0 or not commit_hash:
        raise RuntimeError(f"Git rev-parse HEAD failed for {resolved_root}")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as archive_file:
        archive_path = archive_file.name

    archive_result = subprocess.run(
        ["git", "archive", "--format=tar.gz", f"--output={archive_path}", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=resolved_root,
    )
    if archive_result.returncode != 0:
        os.unlink(archive_path)
        raise RuntimeError(
            f"git archive failed for {resolved_root}: {archive_result.stderr.strip()}"
        )
    return archive_path, commit_hash


def deploy_code(
    ssh_client: paramiko.SSHClient,
    config: RemoteConfig,
    workspace_path: str,
    timeout: float = DEPLOY_TIMEOUT_SECONDS,
    allow_dirty: bool = False,
) -> str:
    """Deploy code via git to remote workspace.

    Args:
        ssh_client: Active SSH client connection
        config: Remote connection configuration
        workspace_path: Path to workspace on remote (e.g., ~/.bifrost/workspace)
        timeout: Maximum seconds for the entire deploy (default 5 min)
        allow_dirty: If True, proceed despite uncommitted/untracked changes.
                    If False (default), raise RuntimeError if dirty.

    Returns:
        Path to deployed workspace

    Raises:
        RuntimeError: If deployment fails or workspace is dirty (unless allow_dirty=True)
        TimeoutError: If deployment exceeds timeout
    """
    # Assert inputs (Tiger Style)
    assert ssh_client is not None, "ssh_client cannot be None"
    assert isinstance(config, RemoteConfig), "config must be RemoteConfig"
    assert isinstance(workspace_path, str) and len(workspace_path) > 0, (
        "workspace_path must be non-empty string"
    )
    assert isinstance(allow_dirty, bool), "allow_dirty must be boolean"

    # Check for dirty workspace (uncommitted/untracked changes)
    if not allow_dirty:
        ensure_clean_git_repo(repo_root=os.getcwd(), allow_dirty=allow_dirty)

    deploy_start = time.monotonic()
    logger.info(
        f"deploy_code: starting (workspace={workspace_path}, timeout={timeout}s, allow_dirty={allow_dirty})"
    )

    def _check_timeout(step: str) -> None:
        elapsed = time.monotonic() - deploy_start
        if elapsed > timeout:
            raise TimeoutError(
                f"deploy_code timed out after {elapsed:.1f}s during '{step}' (timeout={timeout}s)"
            )

    # Check if workspace exists
    t0 = time.monotonic()
    stdin, stdout, stderr = ssh_client.exec_command(f"test -d {workspace_path}")
    workspace_exists = stdout.channel.recv_exit_status() == 0
    logger.debug(
        f"deploy_code: workspace exists check: {workspace_exists} ({time.monotonic() - t0:.1f}s)"
    )
    _check_timeout("workspace_exists_check")

    if workspace_exists:
        logger.info("deploy_code: workspace exists, updating...")
        _update_workspace(ssh_client, workspace_path, _check_timeout)
    else:
        logger.info("deploy_code: workspace doesn't exist, creating...")
        _create_workspace(ssh_client, workspace_path, _check_timeout)

    _check_timeout("post_deploy")

    # Get deployed commit hash for logging
    t0 = time.monotonic()
    stdin, stdout, stderr = ssh_client.exec_command(f"cd {workspace_path} && git rev-parse HEAD")
    deployed_hash = stdout.read().decode().strip()
    short_hash = deployed_hash[:7] if deployed_hash else "unknown"
    logger.debug(f"deploy_code: rev-parse took {time.monotonic() - t0:.1f}s")

    total_elapsed = time.monotonic() - deploy_start
    # Assert output
    assert workspace_path, "Failed to deploy code"
    logger.info(f"deploy_code: done in {total_elapsed:.1f}s — {workspace_path} @ {short_hash}")
    return workspace_path


def run_bootstrap(
    ssh_client: paramiko.SSHClient,
    config: RemoteConfig,
    workspace_path: str,
    bootstrap_cmd: str | list[str],
    on_step: Callable[[str, int, int], None] | None = None,
) -> None:
    """Run bootstrap command(s) to prepare environment.

    Args:
        ssh_client: Active SSH client connection
        config: Remote connection configuration
        workspace_path: Path to workspace on remote
        bootstrap_cmd: Command(s) to run - either single string or list of commands
                      Each command runs in sequence, fails fast if any fails
        on_step: Optional callback called before each step with (cmd, index, total).
                 Use for progress reporting (e.g., updating a spinner).

    Raises:
        RuntimeError: If any bootstrap step fails
    """
    # Assert inputs (Tiger Style)
    assert ssh_client is not None, "ssh_client cannot be None"
    assert isinstance(config, RemoteConfig), "config must be RemoteConfig"
    assert isinstance(workspace_path, str) and len(workspace_path) > 0, (
        "workspace_path must be non-empty string"
    )
    assert isinstance(bootstrap_cmd, (str, list)), "bootstrap_cmd must be string or list of strings"

    # Normalize to list for uniform processing
    if isinstance(bootstrap_cmd, str):
        commands = [bootstrap_cmd]
    else:
        commands = bootstrap_cmd
        # Assert all items are strings
        assert all(isinstance(cmd, str) and len(cmd) > 0 for cmd in commands), (
            "All bootstrap commands must be non-empty strings"
        )

    logger.info(f"running {len(commands)} bootstrap step(s)...")

    # Execute each command in sequence
    for i, cmd in enumerate(commands):
        # Fire callback before step (0-indexed)
        if on_step is not None:
            on_step(cmd, i, len(commands))

        # Log what we're doing (truncate long commands)
        cmd_preview = cmd[:60] + "..." if len(cmd) > 60 else cmd
        logger.debug(f"step {i + 1}/{len(commands)}: {cmd_preview}")

        # Run command in workspace
        full_cmd = f"cd {workspace_path} && {cmd}"
        stdin, stdout, stderr = ssh_client.exec_command(full_cmd)

        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            error_output = stderr.read().decode()
            raise RuntimeError(
                f"Bootstrap step {i}/{len(commands)} failed with exit code {exit_code}: {error_output}"
            )

        logger.debug(f"Step {i + 1}/{len(commands)} completed successfully")

    logger.info("all bootstrap steps completed successfully")


def _warn_dirty_files() -> None:
    """Print loud warnings for untracked/uncommitted files that won't be deployed."""
    untracked = _check_untracked_files()
    if untracked:
        print(
            f"\n⚠️  WARNING: {len(untracked)} untracked file(s) will NOT be deployed (not in git):"
        )
        for file in untracked[:5]:
            print(f"   - {file}")
        if len(untracked) > 5:
            print(f"   ... and {len(untracked) - 5} more")
        print("   Run 'git add <file>' to include them.\n")

    uncommitted = _check_uncommitted_changes()
    if uncommitted:
        print(f"\n⚠️  WARNING: {len(uncommitted)} uncommitted change(s) will NOT be deployed:")
        for file in uncommitted[:5]:
            print(f"   - {file}")
        if len(uncommitted) > 5:
            print(f"   ... and {len(uncommitted) - 5} more")
        print("   Run 'git commit' to include them.\n")


def _create_workspace(
    ssh_client: paramiko.SSHClient,
    workspace_path: str,
    check_timeout: Callable[[str], None] = lambda _: None,
) -> None:
    """Create new workspace by cloning current git repo.

    Uses git bundle to transfer code without remote repo setup.
    """
    import os
    import subprocess
    import tempfile

    _warn_dirty_files()

    # Get current HEAD commit hash for logging
    hash_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else "unknown"
    short_hash = commit_hash[:7] if commit_hash != "unknown" else "unknown"

    logger.info(f"_create_workspace: bundling HEAD {short_hash}")

    # Create git bundle locally
    with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as bundle_file:
        bundle_path = bundle_file.name

    try:
        # Create bundle from current HEAD
        t0 = time.monotonic()
        result = subprocess.run(
            ["git", "bundle", "create", bundle_path, "HEAD"], capture_output=True, text=True
        )
        bundle_elapsed = time.monotonic() - t0
        assert result.returncode == 0, (
            f"Git bundle create failed: {result.stderr}\n\nNot in a git repository. Run 'git init' first."
        )

        bundle_size = os.path.getsize(bundle_path)
        logger.info(
            f"_create_workspace: bundle created ({bundle_size:,} bytes) in {bundle_elapsed:.1f}s"
        )
        check_timeout("bundle_create")

        # Upload bundle to remote with retry logic
        t0 = time.monotonic()
        sftp = ssh_client.open_sftp()
        try:
            remote_bundle = f"/tmp/bifrost-bundle-{os.getpid()}.bundle"
            _upload_bundle_with_retry(sftp, bundle_path, remote_bundle)
        finally:
            sftp.close()
        logger.info(f"_create_workspace: sftp upload took {time.monotonic() - t0:.1f}s")
        check_timeout("sftp_upload")

        # Clone from bundle on remote
        t0 = time.monotonic()
        logger.info("_create_workspace: cloning from bundle on remote...")
        stdin, stdout, stderr = ssh_client.exec_command(
            f"git clone {remote_bundle} {workspace_path} && rm {remote_bundle}"
        )

        exit_code = stdout.channel.recv_exit_status()
        clone_elapsed = time.monotonic() - t0
        logger.info(f"_create_workspace: remote clone took {clone_elapsed:.1f}s (exit={exit_code})")
        if exit_code != 0:
            error_output = stderr.read().decode()
            raise RuntimeError(f"Git clone from bundle failed: {error_output}")
        check_timeout("remote_clone")

        # Verify what commit was deployed
        verify_cmd = f"cd {workspace_path} && git rev-parse HEAD"
        stdin, stdout, stderr = ssh_client.exec_command(verify_cmd)
        deployed_hash = stdout.read().decode().strip()
        deployed_short = deployed_hash[:7] if deployed_hash else "unknown"

        logger.info(f"_create_workspace: done — deployed {deployed_short}")
        if deployed_hash != commit_hash:
            logger.warning("Deployed hash doesn't match local HEAD!")
            logger.warning(f"   Local:  {short_hash} ({commit_hash})")
            logger.warning(f"   Remote: {deployed_short} ({deployed_hash})")

    finally:
        # Clean up local bundle
        if os.path.exists(bundle_path):
            os.unlink(bundle_path)


def _update_workspace(
    ssh_client: paramiko.SSHClient,
    workspace_path: str,
    check_timeout: Callable[[str], None] = lambda _: None,
) -> None:
    """Update existing workspace with latest code.

    Uses git bundle to transfer changes.
    """
    import os
    import subprocess
    import tempfile

    _warn_dirty_files()

    # Get current HEAD commit hash for logging
    hash_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else "unknown"
    short_hash = commit_hash[:7] if commit_hash != "unknown" else "unknown"

    logger.info(f"_update_workspace: bundling HEAD {short_hash}")

    # Create git bundle locally
    with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as bundle_file:
        bundle_path = bundle_file.name

    try:
        # Create bundle from current HEAD
        t0 = time.monotonic()
        result = subprocess.run(
            ["git", "bundle", "create", bundle_path, "HEAD"], capture_output=True, text=True
        )
        bundle_elapsed = time.monotonic() - t0
        assert result.returncode == 0, (
            f"Git bundle create failed: {result.stderr}\n\nNot in a git repository. Run 'git init' first."
        )

        bundle_size = os.path.getsize(bundle_path)
        logger.info(
            f"_update_workspace: bundle created ({bundle_size:,} bytes) in {bundle_elapsed:.1f}s"
        )
        check_timeout("bundle_create")

        # Upload bundle to remote with retry logic
        t0 = time.monotonic()
        sftp = ssh_client.open_sftp()
        try:
            remote_bundle = f"/tmp/bifrost-bundle-{os.getpid()}.bundle"
            _upload_bundle_with_retry(sftp, bundle_path, remote_bundle)
        finally:
            sftp.close()
        logger.info(f"_update_workspace: sftp upload took {time.monotonic() - t0:.1f}s")
        check_timeout("sftp_upload")

        # Fetch and reset from bundle
        t0 = time.monotonic()
        logger.info("_update_workspace: fetching and resetting on remote...")
        update_cmd = f"""
cd {workspace_path} &&
git fetch {remote_bundle} HEAD &&
git reset --hard FETCH_HEAD &&
rm {remote_bundle}
"""
        stdin, stdout, stderr = ssh_client.exec_command(update_cmd)

        exit_code = stdout.channel.recv_exit_status()
        fetch_elapsed = time.monotonic() - t0
        logger.info(
            f"_update_workspace: remote fetch+reset took {fetch_elapsed:.1f}s (exit={exit_code})"
        )
        if exit_code != 0:
            error_output = stderr.read().decode()
            raise RuntimeError(f"Git update from bundle failed: {error_output}")
        check_timeout("remote_fetch_reset")

        # Verify what commit was deployed
        verify_cmd = f"cd {workspace_path} && git rev-parse HEAD"
        stdin, stdout, stderr = ssh_client.exec_command(verify_cmd)
        deployed_hash = stdout.read().decode().strip()
        deployed_short = deployed_hash[:7] if deployed_hash else "unknown"

        logger.info(f"_update_workspace: done — deployed {deployed_short}")
        if deployed_hash != commit_hash:
            logger.warning("Deployed hash doesn't match local HEAD!")
            logger.warning(f"   Local:  {short_hash} ({commit_hash})")
            logger.warning(f"   Remote: {deployed_short} ({deployed_hash})")

    finally:
        # Clean up local bundle
        if os.path.exists(bundle_path):
            os.unlink(bundle_path)
