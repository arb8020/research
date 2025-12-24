"""Bifrost Async SDK - Trio-based async client for remote GPU execution."""

import logging
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import asyncssh
import trio
import trio_asyncio
from shared.validation import validate_ssh_key_path, validate_timeout

from .types import (
    CopyResult,
    EnvironmentVariables,
    ExecResult,
    RemoteConfig,
    SSHConnection,
    SSHConnectionError,
    TransferError,
)
from .validation import validate_bootstrap_cmd

logger = logging.getLogger(__name__)


def _trio_wrap(coro_func: Callable) -> Callable:
    """Helper to wrap asyncio coroutines for trio-asyncio.

    Usage: await _trio_wrap(conn.run)(args, kwargs)
    """
    return trio_asyncio.aio_as_trio(coro_func)


class AsyncBifrostClient:
    """
    Async Bifrost SDK client for remote GPU execution and job management.

    Provides async access to all Bifrost functionality using Trio:
    - Remote code execution (detached and synchronous)
    - Job monitoring and log streaming
    - File transfer operations
    - Git-based code deployment

    Example:
        async with AsyncBifrostClient("root@gpu.example.com:22", ssh_key_path="~/.ssh/id_rsa") as client:
            job = await client.run_detached("python train_model.py")
            await client.wait_for_completion(job.job_id)
            await client.copy_files("/remote/outputs/", "./results/", recursive=True)
    """

    def __init__(
        self,
        ssh_connection: str,
        ssh_key_path: str,
        timeout: int = 30,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """
        Initialize Async Bifrost client.

        Args:
            ssh_connection: SSH connection string like 'user@host:port'
            ssh_key_path: Path to SSH private key (required)
            timeout: SSH connection timeout in seconds
            progress_callback: Optional callback for file transfer progress
        """
        # Validate and parse SSH connection
        self.ssh = SSHConnection.from_string(ssh_connection)

        # Validate inputs
        validated_ssh_key_path = validate_ssh_key_path(ssh_key_path)
        validated_timeout = validate_timeout(timeout, min_value=1, max_value=300)

        # Create RemoteConfig
        self._remote_config = RemoteConfig(
            host=self.ssh.host,
            port=self.ssh.port,
            user=self.ssh.user,
            key_path=validated_ssh_key_path,
        )

        self.timeout = validated_timeout
        self.ssh_key_path = validated_ssh_key_path
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)

        # Connection will be established on-demand
        self._ssh_conn: asyncssh.SSHClientConnection | None = None

        # Track last deployed workspace for smart working_dir defaults
        self._last_workspace: str | None = None

    async def _establish_connection(self) -> asyncssh.SSHClientConnection:
        """Establish SSH connection with retry logic using Trio.

        Uses trio_asyncio to bridge between Trio and asyncssh (which is asyncio-based).
        Trio-style retry with exponential backoff.

        Returns:
            asyncssh.SSHClientConnection

        Raises:
            SSHConnectionError: If connection fails after all retry attempts
        """
        # Retry 3 times with exponential backoff (2s, 4s, 8s = 14s total)
        # Enough to handle transient network issues but fail fast on real problems
        max_attempts = 3
        delay = 2
        backoff = 2

        for attempt in range(max_attempts):
            try:
                # Use trio_asyncio.aio_as_trio to call asyncio code from trio
                conn = await trio_asyncio.aio_as_trio(asyncssh.connect)(
                    host=self.ssh.host,
                    port=self.ssh.port,
                    username=self.ssh.user,
                    client_keys=[self.ssh_key_path] if self.ssh_key_path else None,
                    connect_timeout=self.timeout,
                    # Send keepalive every 30s to prevent idle timeout
                    # Most SSH servers drop idle connections after 60s, so 30s = 2x safety margin
                    keepalive_interval=30,
                    known_hosts=None,  # Accept any host key (like paramiko.AutoAddPolicy)
                )
            except Exception as e:
                if attempt < max_attempts - 1:
                    wait_time = delay * (backoff**attempt)
                    self.logger.debug(
                        f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s..."
                    )
                    await trio.sleep(wait_time)
                else:
                    raise SSHConnectionError(
                        f"Failed to connect to {self.ssh} after {max_attempts} attempts: {e}"
                    ) from e
            else:
                self.logger.debug(f"Connected to {self.ssh}")
                return conn

        raise SSHConnectionError(f"Failed to connect to {self.ssh}")

    async def _get_connection(self) -> asyncssh.SSHClientConnection:
        """Get or create SSH connection.

        Checks if connection is active, reconnects if needed.
        """
        # Check if we need to establish a new connection
        if self._ssh_conn is None:
            self._ssh_conn = await self._establish_connection()
            assert self._ssh_conn is not None, "SSH connection must be initialized"
            return self._ssh_conn

        # Check if existing connection is still alive
        transport = self._ssh_conn._transport
        if transport is None or transport.is_closing():
            self.logger.debug("SSH connection inactive, reconnecting...")
            self._ssh_conn = await self._establish_connection()
            assert self._ssh_conn is not None, "SSH connection must be initialized"
            return self._ssh_conn

        # Connection is active, return it
        assert self._ssh_conn is not None, "SSH connection must be initialized"
        return self._ssh_conn

    def _build_command_with_env(
        self, command: str, working_dir: str, env: EnvironmentVariables | None
    ) -> str:
        """Build command with environment variables and working directory.

        Args:
            command: Command to execute
            working_dir: Directory to run in
            env: Environment variables (EnvironmentVariables dataclass)

        Returns:
            Full command string with cd and exports
        """
        import shlex

        parts = []

        # Change directory
        parts.append(f"cd {working_dir}")

        # Export environment variables
        if env:
            env_dict = env.to_dict()
            for key, value in env_dict.items():
                # Use shell quoting for safety
                parts.append(f"export {key}={shlex.quote(value)}")

        # Execute command
        parts.append(command)

        return " && ".join(parts)

    async def exec(
        self,
        command: str,
        env: EnvironmentVariables | dict[str, str] | None = None,
        working_dir: str | None = None,
    ) -> ExecResult:
        """
        Execute command in remote environment.

        This method:
        1. Executes command directly on remote instance
        2. Runs in context of working directory (defaults to ~/.bifrost/workspace/)
        3. Applies environment variables if provided
        4. Returns ExecResult (never raises on non-zero exit)

        Mental model: Like `docker exec` - run command in remote environment

        Args:
            command: Command to execute
            env: Environment variables to set (dict or EnvironmentVariables)
            working_dir: Working directory (defaults to ~/.bifrost/workspace/ if deployed)

        Returns:
            ExecResult with stdout, stderr, exit_code

        Raises:
            SSHConnectionError: SSH connection failed
        """
        try:
            conn = await self._get_connection()

            # Default to last deployed workspace, or home directory if nothing deployed
            if working_dir is None:
                working_dir = self._last_workspace or "~"
                if self._last_workspace:
                    self.logger.debug(f"Using workspace from last push(): {working_dir}")
                else:
                    self.logger.debug(
                        f"No workspace deployed yet, using home directory: {working_dir}"
                    )

            # Convert dict to EnvironmentVariables if needed
            if env is None:
                env_vars = None
            elif isinstance(env, EnvironmentVariables):
                env_vars = env
            elif isinstance(env, dict):
                env_vars = EnvironmentVariables.from_dict(env)
            else:
                env_vars = None

            # Build command with environment and working directory
            full_command = self._build_command_with_env(command, working_dir, env_vars)

            # Execute command
            result = await _trio_wrap(conn.run)(full_command, check=False)

            return ExecResult(
                stdout=result.stdout, stderr=result.stderr, exit_code=result.exit_status or 0
            )

        except Exception as e:
            if isinstance(e, SSHConnectionError):
                raise
            raise SSHConnectionError(f"Execution failed: {e}") from e

    async def exec_stream(
        self,
        command: str,
        env: EnvironmentVariables | dict[str, str] | None = None,
        working_dir: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Execute command and stream output line-by-line in real-time.

        Like exec() but yields output as it's produced instead of waiting for completion.
        Useful for long-running commands like package installations.

        Args:
            command: Command to execute
            env: Environment variables to set (dict or EnvironmentVariables)
            working_dir: Working directory (defaults to ~/.bifrost/workspace/ if deployed)

        Yields:
            Lines of output (stdout and stderr interleaved) as they're produced

        Raises:
            SSHConnectionError: SSH connection failed
        """
        try:
            conn = await self._get_connection()

            # Default to last deployed workspace, or home directory if nothing deployed (same logic as exec)
            if working_dir is None:
                working_dir = self._last_workspace or "~"
                if self._last_workspace:
                    self.logger.debug(f"Using workspace from last push(): {working_dir}")
                else:
                    self.logger.debug(
                        f"No workspace deployed yet, using home directory: {working_dir}"
                    )

            # Convert dict to EnvironmentVariables if needed
            if env is None:
                env_vars = None
            elif isinstance(env, EnvironmentVariables):
                env_vars = env
            elif isinstance(env, dict):
                env_vars = EnvironmentVariables.from_dict(env)
            else:
                env_vars = None

            # Build command with environment and working directory
            full_command = self._build_command_with_env(command, working_dir, env_vars)

            # Create async process - asyncssh returns AsyncIterator for stdout
            # Using term_type='ansi' to get a PTY which combines stdout/stderr
            process = await _trio_wrap(conn.create_process)(full_command, term_type="ansi")
            try:
                # Stream output line by line
                # Can't use 'async for' because asyncssh's async iterator doesn't work with
                # trio-asyncio's event loop shim. Manual readline() calls work correctly.
                while True:
                    try:
                        line = await _trio_wrap(process.stdout.readline)()
                        if not line:
                            break
                        yield line.rstrip("\r\n")
                    except EOFError:
                        break
            finally:
                process.close()

        except Exception as e:
            if isinstance(e, SSHConnectionError):
                raise
            raise SSHConnectionError(f"Streaming execution failed: {e}") from e

    async def push(self, workspace_path: str, bootstrap_cmd: str | list[str] | None = None) -> str:
        """Deploy code to remote workspace.

        Args:
            workspace_path: Remote workspace path (REQUIRED).
                          Must be explicit to prevent accidental collisions.

                          Recommended convention: ~/.bifrost/workspaces/{project-name}

                          Examples:
                            push(workspace_path="~/.bifrost/workspaces/clicker")
                            push(workspace_path="~/.bifrost/workspaces/integration_training")
                            push(workspace_path="~/projects/my-custom-project")
                            push(workspace_path="/opt/production/deployment")

            bootstrap_cmd: Optional bootstrap command(s) - either single string or list of commands
                          (e.g., "uv sync --frozen" or ["pip install uv", "uv sync --frozen"])

        Returns:
            Path to deployed workspace (absolute, tilde-expanded)

        Raises:
            SSHConnectionError: SSH connection failed
            RuntimeError: Deployment failed

        Note:
            Each project should use a unique workspace_path to prevent
            collisions when running multiple projects on the same remote node.
        """
        # Validate input
        assert workspace_path, "workspace_path is required and cannot be empty"
        if bootstrap_cmd is not None:
            bootstrap_cmd = validate_bootstrap_cmd(bootstrap_cmd)

        self.logger.debug(f"📁 Deploying to workspace: {workspace_path}")

        # TODO: Implement async git deployment
        # For now, we'll create a minimal implementation that:
        # 1. Creates the workspace directory
        # 2. Syncs code (simplified version)
        # 3. Runs bootstrap if specified

        conn = await self._get_connection()

        # Create workspace directory
        await _trio_wrap(conn.run)(f"mkdir -p {workspace_path}", check=True)

        # Expand path to absolute (resolves ~ and env vars)
        expanded_result = await _trio_wrap(conn.run)(f"echo {workspace_path}", check=True)
        expanded_workspace_path = expanded_result.stdout.strip()

        # Run bootstrap if specified
        if bootstrap_cmd:
            # Handle both single string and list of commands
            commands = [bootstrap_cmd] if isinstance(bootstrap_cmd, str) else bootstrap_cmd
            for cmd in commands:
                self.logger.debug(f"Running bootstrap: {cmd}")
                result = await _trio_wrap(conn.run)(
                    f"cd {expanded_workspace_path} && {cmd}", check=False
                )
                if result.exit_status != 0:
                    raise RuntimeError(f"Bootstrap command failed: {cmd}\n{result.stderr}")

        # Assert output
        assert expanded_workspace_path, "push() returned empty workspace_path"

        # Track last deployed workspace for smart working_dir defaults
        self._last_workspace = expanded_workspace_path

        return expanded_workspace_path

    async def expand_path(self, path: str) -> str:
        """Expand ~ and environment variables in path to absolute path.

        This is a convenience helper to expand paths on the remote machine.

        Args:
            path: Path to expand (may contain ~ or env vars)

        Returns:
            Absolute expanded path on remote machine

        Raises:
            SSHConnectionError: SSH connection failed

        Example:
            workspace = await client.push(workspace_path="~/.bifrost/workspaces/foo")
            workspace = await client.expand_path(workspace)  # /home/user/.bifrost/workspaces/foo
        """
        assert path, "path must be non-empty string"

        result = await self.exec(f"echo {path}")
        assert result.exit_code == 0, f"Failed to expand path: {result.stderr}"

        expanded = result.stdout.strip()
        assert expanded, f"Path expansion returned empty string for: {path}"

        return expanded

    async def copy_files(
        self, remote_path: str, local_path: str, recursive: bool = False
    ) -> CopyResult:
        """
        Copy files from remote to local machine.

        Args:
            remote_path: Remote file or directory path
            local_path: Local destination path
            recursive: Copy directories recursively

        Returns:
            CopyResult with transfer statistics

        Raises:
            SSHConnectionError: SSH connection failed
            TransferError: File transfer failed
        """
        import time

        start_time = time.time()

        try:
            conn = await self._get_connection()

            # Check if remote path exists
            result = await _trio_wrap(conn.run)(f"test -e {remote_path}", check=False)
            if result.exit_status != 0:
                raise TransferError(f"Remote path not found: {remote_path}")

            # Check if remote path is directory
            result = await _trio_wrap(conn.run)(f"test -d {remote_path}", check=False)
            is_directory = result.exit_status == 0

            if is_directory and not recursive:
                raise TransferError(f"{remote_path} is a directory. Use recursive=True")

            # Start SFTP session
            sftp = await _trio_wrap(conn.start_sftp_client)()
            try:
                files_copied = 0
                total_bytes = 0

                if is_directory:
                    files_copied, total_bytes = await self._copy_directory(
                        sftp, conn, remote_path, local_path
                    )
                else:
                    total_bytes = await self._copy_file(sftp, remote_path, local_path)
                    files_copied = 1

                duration = time.time() - start_time

                return CopyResult(
                    success=True,
                    files_copied=files_copied,
                    total_bytes=total_bytes,
                    duration_seconds=duration,
                )
            finally:
                sftp.close()

        except Exception as e:
            if isinstance(e, (SSHConnectionError, TransferError)):
                raise
            duration = time.time() - start_time
            return CopyResult(
                success=False,
                files_copied=0,
                total_bytes=0,
                duration_seconds=duration,
                error_message=str(e),
            )

    async def _copy_file(self, sftp: asyncssh.SFTPClient, remote_path: str, local_path: str) -> int:
        """Copy single file and return bytes transferred."""
        # Ensure local directory exists
        local_dir = Path(local_path).parent
        local_dir.mkdir(parents=True, exist_ok=True)

        # Get file attributes
        attrs = await _trio_wrap(sftp.stat)(remote_path)
        file_size = attrs.size

        # Copy file
        await _trio_wrap(sftp.get)(remote_path, local_path)

        return file_size

    async def _copy_directory(
        self,
        sftp: asyncssh.SFTPClient,
        conn: asyncssh.SSHClientConnection,
        remote_path: str,
        local_path: str,
    ) -> tuple[int, int]:
        """Copy directory recursively and return (files_copied, total_bytes).

        Uses Trio's structured concurrency to copy files in parallel.
        """
        # Get directory listing
        result = await _trio_wrap(conn.run)(f"find {remote_path} -type f", check=True)
        file_list = [f.strip() for f in result.stdout.split("\n") if f.strip()]

        files_copied = 0
        total_bytes = 0

        # Use Trio nursery for parallel file transfers
        async def copy_one_file(remote_file: str) -> None:
            nonlocal files_copied, total_bytes

            # Calculate relative path and local destination (string ops only, no fs access)
            rel_path = os.path.relpath(remote_file, remote_path)  # noqa: ASYNC240
            local_file = os.path.join(local_path, rel_path)

            # Copy file
            try:
                file_bytes = await self._copy_file(sftp, remote_file, local_file)
                files_copied += 1
                total_bytes += file_bytes
            except Exception as e:
                self.logger.warning(f"Failed to copy {rel_path}: {e}")

        # Copy all files in parallel using Trio nursery
        async with trio.open_nursery() as nursery:
            for remote_file in file_list:
                nursery.start_soon(copy_one_file, remote_file)

        return files_copied, total_bytes

    async def upload_files(
        self, local_path: str, remote_path: str, recursive: bool = False
    ) -> CopyResult:
        """
        Upload files from local to remote machine.

        Args:
            local_path: Local file or directory path
            remote_path: Remote destination path
            recursive: Upload directories recursively

        Returns:
            CopyResult with transfer statistics

        Raises:
            SSHConnectionError: SSH connection failed
            TransferError: File transfer failed
        """
        import time

        start_time = time.time()

        try:
            conn = await self._get_connection()

            # Check if local path exists
            local_path_obj = trio.Path(local_path)
            if not await local_path_obj.exists():
                raise TransferError(f"Local path not found: {local_path}")

            is_directory = await local_path_obj.is_dir()

            if is_directory and not recursive:
                raise TransferError(f"{local_path} is a directory. Use recursive=True")

            # Start SFTP session
            sftp = await _trio_wrap(conn.start_sftp_client)()
            try:
                files_uploaded = 0
                total_bytes = 0

                if is_directory:
                    files_uploaded, total_bytes = await self._upload_directory(
                        sftp, local_path, remote_path
                    )
                else:
                    total_bytes = await self._upload_file(sftp, local_path, remote_path)
                    files_uploaded = 1

                duration = time.time() - start_time

                return CopyResult(
                    success=True,
                    files_copied=files_uploaded,
                    total_bytes=total_bytes,
                    duration_seconds=duration,
                )
            finally:
                sftp.close()

        except Exception as e:
            if isinstance(e, (SSHConnectionError, TransferError)):
                raise
            duration = time.time() - start_time
            return CopyResult(
                success=False,
                files_copied=0,
                total_bytes=0,
                duration_seconds=duration,
                error_message=str(e),
            )

    async def _upload_file(
        self, sftp: asyncssh.SFTPClient, local_path: str, remote_path: str
    ) -> int:
        """Upload single file and return bytes transferred."""
        # Create remote directory if needed
        remote_dir = os.path.dirname(remote_path)
        if remote_dir and remote_dir != ".":
            await self._create_remote_dir(sftp, remote_dir)

        # Get file size
        local_stat = await trio.Path(local_path).stat()
        file_size = local_stat.st_size

        # Upload file
        await _trio_wrap(sftp.put)(local_path, remote_path)

        return file_size

    async def _create_remote_dir(self, sftp: asyncssh.SFTPClient, remote_dir: str) -> None:
        """Create remote directory recursively."""
        try:
            await _trio_wrap(sftp.stat)(remote_dir)  # Check if directory exists
        except FileNotFoundError:
            # Directory doesn't exist, create it
            parent_dir = os.path.dirname(remote_dir)
            if parent_dir and parent_dir != remote_dir:  # Avoid infinite recursion
                await self._create_remote_dir(sftp, parent_dir)
            try:
                await _trio_wrap(sftp.mkdir)(remote_dir)
            except OSError:
                # Directory might have been created by another process
                pass

    async def _upload_directory(
        self, sftp: asyncssh.SFTPClient, local_path: str, remote_path: str
    ) -> tuple[int, int]:
        """Upload directory recursively and return (files_uploaded, total_bytes).

        Uses Trio's structured concurrency to upload files in parallel.
        """
        local_path_obj = trio.Path(local_path)

        files_uploaded = 0
        total_bytes = 0

        # Collect all files first
        all_files = []
        for f in await local_path_obj.rglob("*"):
            if await f.is_file():
                all_files.append(f)

        # Use Trio nursery for parallel file uploads
        async def upload_one_file(local_file: trio.Path) -> None:
            nonlocal files_uploaded, total_bytes

            # Calculate relative path and remote destination
            rel_path = local_file.relative_to(local_path_obj)
            remote_file = f"{remote_path}/{rel_path}".replace("\\", "/")

            # Upload file
            try:
                file_bytes = await self._upload_file(sftp, str(local_file), remote_file)
                files_uploaded += 1
                total_bytes += file_bytes
            except Exception as e:
                self.logger.warning(f"Failed to upload {rel_path}: {e}")

        # Upload all files in parallel using Trio nursery
        async with trio.open_nursery() as nursery:
            for local_file in all_files:
                nursery.start_soon(upload_one_file, local_file)

        return files_uploaded, total_bytes

    async def download_files(
        self, remote_path: str, local_path: str, recursive: bool = False
    ) -> CopyResult:
        """
        Download files from remote to local machine.

        This is an alias for copy_files() with clearer naming.

        Args:
            remote_path: Remote file or directory path
            local_path: Local destination path
            recursive: Download directories recursively

        Returns:
            CopyResult with transfer statistics
        """
        return await self.copy_files(remote_path, local_path, recursive)

    async def close(self) -> None:
        """Close SSH connection."""
        if self._ssh_conn:
            self._ssh_conn.close()
            await _trio_wrap(self._ssh_conn.wait_closed)()
            self._ssh_conn = None

    async def __aenter__(self) -> "AsyncBifrostClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.close()
