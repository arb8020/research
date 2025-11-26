"""Bifrost Async SDK - Trio-based async client for remote GPU execution."""

import asyncssh
import trio
import trio_asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, AsyncIterator, List, Dict, Union
from contextlib import asynccontextmanager
import logging

from shared.validation import validate_ssh_key_path, validate_timeout
from .types import (
    SSHConnection, JobInfo, JobStatus, CopyResult, ConnectionError, JobError, TransferError,
    RemoteConfig, ExecResult, EnvironmentVariables, SessionInfo, JobMetadata
)
from .validation import (
    generate_job_id, validate_bootstrap_cmd, validate_command,
    validate_environment_variables, validate_poll_interval
)


logger = logging.getLogger(__name__)


def _trio_wrap(coro_func):
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
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
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
            key_path=validated_ssh_key_path
        )

        self.timeout = validated_timeout
        self.ssh_key_path = validated_ssh_key_path
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)

        # Connection will be established on-demand
        self._ssh_conn: Optional[asyncssh.SSHClientConnection] = None

    async def _establish_connection(self) -> asyncssh.SSHClientConnection:
        """Establish SSH connection with retry logic using Trio.

        Uses trio_asyncio to bridge between Trio and asyncssh (which is asyncio-based).
        Trio-style retry with exponential backoff.

        Returns:
            asyncssh.SSHClientConnection

        Raises:
            ConnectionError: If connection fails after all retry attempts
        """
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
                    keepalive_interval=30,  # Send keepalive every 30 seconds
                    known_hosts=None,  # Accept any host key (like paramiko.AutoAddPolicy)
                )

                self.logger.debug(f"Connected to {self.ssh}")
                return conn

            except Exception as e:
                if attempt < max_attempts - 1:
                    wait_time = delay * (backoff ** attempt)
                    self.logger.debug(f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    await trio.sleep(wait_time)
                else:
                    raise ConnectionError(f"Failed to connect to {self.ssh} after {max_attempts} attempts: {e}")

        raise ConnectionError(f"Failed to connect to {self.ssh}")

    async def _get_connection(self) -> asyncssh.SSHClientConnection:
        """Get or create SSH connection.

        Checks if connection is active, reconnects if needed.
        """
        needs_reconnect = False

        if self._ssh_conn is None:
            needs_reconnect = True
        elif self._ssh_conn._transport.is_closing():
            self.logger.debug("SSH connection inactive, reconnecting...")
            needs_reconnect = True

        if needs_reconnect:
            self._ssh_conn = await self._establish_connection()

        # Assert post-condition
        assert self._ssh_conn is not None, "SSH connection must be initialized"
        return self._ssh_conn

    def _build_command_with_env(self, command: str, working_dir: str,
                                env: Optional[EnvironmentVariables]) -> str:
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
        env: Optional[Union[EnvironmentVariables, Dict[str, str]]] = None,
        working_dir: Optional[str] = None
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
            ConnectionError: SSH connection failed
        """
        try:
            conn = await self._get_connection()

            # Default to workspace if it exists
            if working_dir is None:
                result = await _trio_wrap(conn.run)("test -d ~/.bifrost/workspace", check=False)
                if result.exit_status == 0:
                    working_dir = "~/.bifrost/workspace"
                    self.logger.debug(f"Using default working directory: {working_dir}")
                else:
                    working_dir = "~"
                    self.logger.warning("No code deployed yet. Running from home directory.")

            # Convert dict to EnvironmentVariables if needed
            env_vars = None
            if env is not None:
                if isinstance(env, EnvironmentVariables):
                    env_vars = env
                elif isinstance(env, dict):
                    env_vars = EnvironmentVariables.from_dict(env)

            # Build command with environment and working directory
            full_command = self._build_command_with_env(command, working_dir, env_vars)

            # Execute command
            result = await _trio_wrap(conn.run)(full_command, check=False)

            return ExecResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_status or 0
            )

        except Exception as e:
            if isinstance(e, ConnectionError):
                raise
            raise ConnectionError(f"Execution failed: {e}")

    async def exec_stream(
        self,
        command: str,
        env: Optional[Union[EnvironmentVariables, Dict[str, str]]] = None,
        working_dir: Optional[str] = None
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
            ConnectionError: SSH connection failed
        """
        try:
            conn = await self._get_connection()

            # Default to workspace if it exists (same logic as exec)
            if working_dir is None:
                result = await _trio_wrap(conn.run)("test -d ~/.bifrost/workspace", check=False)
                if result.exit_status == 0:
                    working_dir = "~/.bifrost/workspace"
                    self.logger.debug(f"Using default working directory: {working_dir}")
                else:
                    working_dir = "~"
                    self.logger.warning("No code deployed yet. Running from home directory.")

            # Convert dict to EnvironmentVariables if needed
            env_vars = None
            if env is not None:
                if isinstance(env, EnvironmentVariables):
                    env_vars = env
                elif isinstance(env, dict):
                    env_vars = EnvironmentVariables.from_dict(env)

            # Build command with environment and working directory
            full_command = self._build_command_with_env(command, working_dir, env_vars)

            # Create async process - asyncssh returns AsyncIterator for stdout
            # Using term_type='ansi' to get a PTY which combines stdout/stderr
            process = await _trio_wrap(conn.create_process)(full_command, term_type='ansi')
            try:
                # Stream output line by line
                # We need to manually read lines using wrapped readline calls
                while True:
                    try:
                        line = await _trio_wrap(process.stdout.readline)()
                        if not line:
                            break
                        yield line.rstrip('\r\n')
                    except EOFError:
                        break
            finally:
                process.close()

        except Exception as e:
            if isinstance(e, ConnectionError):
                raise
            raise ConnectionError(f"Streaming execution failed: {e}")

    async def push(
        self,
        workspace_path: str,
        bootstrap_cmd: Optional[Union[str, List[str]]] = None
    ) -> str:
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
            ConnectionError: SSH connection failed
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

        # Expand path to absolute
        expanded_result = await _trio_wrap(conn.run)(f"echo {workspace_path}", check=True)
        workspace_path = expanded_result.stdout.strip()

        # Run bootstrap if specified
        if bootstrap_cmd:
            # Handle both single string and list of commands
            commands = [bootstrap_cmd] if isinstance(bootstrap_cmd, str) else bootstrap_cmd
            for cmd in commands:
                self.logger.debug(f"Running bootstrap: {cmd}")
                result = await _trio_wrap(conn.run)(f"cd {workspace_path} && {cmd}", check=False)
                if result.exit_status != 0:
                    raise RuntimeError(f"Bootstrap command failed: {cmd}\n{result.stderr}")

        # Assert output
        assert workspace_path, "push() returned empty workspace_path"
        return workspace_path

    async def deploy(
        self,
        command: str,
        bootstrap_cmd: Optional[Union[str, List[str]]] = None,
        env: Optional[Union[EnvironmentVariables, Dict[str, str]]] = None
    ) -> ExecResult:
        """Deploy code and execute command.

        Equivalent to: push(bootstrap_cmd) + exec(command, env)

        Args:
            command: Command to execute
            bootstrap_cmd: Optional bootstrap command(s) - either single string or list of commands
            env: Environment variables

        Returns:
            ExecResult with stdout, stderr, exit_code

        Raises:
            ConnectionError: SSH connection failed
            RuntimeError: Deployment failed
        """
        workspace_path = await self.push(workspace_path="~/.bifrost/workspace", bootstrap_cmd=bootstrap_cmd)
        return await self.exec(command, env=env, working_dir=workspace_path)

    async def expand_path(self, path: str) -> str:
        """Expand ~ and environment variables in path to absolute path.

        This is a convenience helper to expand paths on the remote machine.

        Args:
            path: Path to expand (may contain ~ or env vars)

        Returns:
            Absolute expanded path on remote machine

        Raises:
            ConnectionError: SSH connection failed

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

    async def get_all_jobs(self) -> List[JobInfo]:
        """
        Get status of all jobs on the remote instance.

        Returns:
            List of JobInfo objects for all jobs

        Raises:
            ConnectionError: SSH connection failed
        """
        try:
            conn = await self._get_connection()

            # Get list of job directories
            result = await _trio_wrap(conn.run)("ls -1 ~/.bifrost/jobs/ 2>/dev/null || echo ''", check=False)
            job_dirs = [d.strip() for d in result.stdout.split('\n') if d.strip()]

            jobs = []
            for job_id in job_dirs:
                try:
                    job_info = await self.get_job_status(job_id)
                    jobs.append(job_info)
                except JobError:
                    # Skip jobs that can't be read
                    continue

            return jobs

        except Exception as e:
            if isinstance(e, ConnectionError):
                raise
            raise ConnectionError(f"Failed to list jobs: {e}")

    async def get_job_status(self, job_id: str) -> JobInfo:
        """
        Get current status of a detached job.

        Args:
            job_id: Job identifier

        Returns:
            JobInfo with current status

        Raises:
            ConnectionError: SSH connection failed
            JobError: Job not found or status check failed
        """
        try:
            conn = await self._get_connection()

            # Get job metadata
            metadata_result = await _trio_wrap(conn.run)(
                f"cat ~/.bifrost/jobs/{job_id}/metadata.json 2>/dev/null",
                check=False
            )

            if metadata_result.exit_status != 0:
                raise JobError(f"Job {job_id} not found")

            metadata_dict = json.loads(metadata_result.stdout)
            # Parse metadata using frozen dataclass for validation
            metadata = JobMetadata.from_dict(metadata_dict)

            # Get current status (may be updated from metadata)
            status_result = await _trio_wrap(conn.run)(
                f"cat ~/.bifrost/jobs/{job_id}/status 2>/dev/null",
                check=False
            )
            status_str = status_result.stdout.strip()

            # Parse times from metadata
            start_time = datetime.fromisoformat(metadata.start_time.replace('Z', '+00:00'))
            end_time = None
            if metadata.end_time:
                end_time = datetime.fromisoformat(metadata.end_time.replace('Z', '+00:00'))

            # Calculate runtime
            runtime_seconds = None
            if end_time:
                runtime_seconds = (end_time - start_time).total_seconds()
            else:
                runtime_seconds = (datetime.now().astimezone() - start_time).total_seconds()

            return JobInfo(
                job_id=job_id,
                status=JobStatus(status_str) if status_str else JobStatus(metadata.status),
                command=metadata.command,
                start_time=start_time,
                end_time=end_time,
                exit_code=metadata.exit_code,
                runtime_seconds=runtime_seconds
            )

        except json.JSONDecodeError as e:
            raise JobError(f"Invalid job metadata for {job_id}: {e}")
        except Exception as e:
            if isinstance(e, (ConnectionError, JobError)):
                raise
            raise JobError(f"Failed to get job status: {e}")

    async def get_logs(self, job_id: str, lines: int = 100, log_type: str = "command") -> str:
        """Get recent logs from a job.

        Args:
            job_id: Job identifier
            lines: Number of lines to retrieve (default: 100)
            log_type: "command" or "bootstrap"

        Returns:
            Log content

        Raises:
            ConnectionError: SSH connection failed
            JobError: Job not found or logs unavailable
        """
        try:
            conn = await self._get_connection()

            if log_type == "bootstrap":
                log_file = f"~/.bifrost/jobs/{job_id}/bootstrap.log"
            else:
                log_file = f"~/.bifrost/jobs/{job_id}/job.log"

            # Check if log file exists
            check_result = await _trio_wrap(conn.run)(f"test -f {log_file}", check=False)
            if check_result.exit_status != 0:
                raise JobError(f"No {log_type} log found for job {job_id}")

            # Get last N lines
            result = await _trio_wrap(conn.run)(f"tail -n {lines} {log_file}", check=False)

            if result.exit_status != 0:
                raise JobError(f"Failed to read logs: {result.stderr}")

            return result.stdout

        except Exception as e:
            if isinstance(e, (ConnectionError, JobError)):
                raise
            raise JobError(f"Failed to get job logs: {e}")

    async def follow_job_logs(self, job_id: str) -> AsyncIterator[str]:
        """
        Stream job logs in real-time (like tail -f).

        Args:
            job_id: Job identifier

        Yields:
            Log lines as they are written

        Raises:
            ConnectionError: SSH connection failed
            JobError: Job not found or logs unavailable
        """
        try:
            conn = await self._get_connection()

            log_file = f"~/.bifrost/jobs/{job_id}/job.log"

            # Use tail -f to follow the log file
            process = await _trio_wrap(conn.create_process)(f"tail -f {log_file}")
            try:
                while True:
                    try:
                        line = await _trio_wrap(process.stdout.readline)()
                        if not line:
                            break
                        yield line.rstrip('\n')
                    except EOFError:
                        break
            finally:
                process.close()

        except Exception as e:
            if isinstance(e, ConnectionError):
                raise
            raise JobError(f"Failed to follow job logs: {e}")

    async def list_sessions(self) -> List[str]:
        """List all bifrost tmux sessions on remote.

        Returns:
            List of tmux session names
        """
        conn = await self._get_connection()

        result = await _trio_wrap(conn.run)("tmux list-sessions -F '#{session_name}' 2>/dev/null || echo ''", check=False)
        if result.exit_status != 0:
            return []

        sessions = result.stdout.strip().split('\n')
        # Filter to bifrost sessions only
        return [s for s in sessions if s.startswith('bifrost-') and s]

    async def get_session_info(self, job_id: str) -> SessionInfo:
        """Get tmux session information for a job.

        Returns:
            SessionInfo with session names and attach commands
        """
        job = await self.get_job_status(job_id)

        main_session = job.tmux_session or f"bifrost-{job_id}"
        attach_main = f"ssh {self.ssh.user}@{self.ssh.host} -p {self.ssh.port} -t 'tmux attach -t {main_session}'"

        if job.bootstrap_session:
            bootstrap_session = job.bootstrap_session
            attach_bootstrap = f"ssh {self.ssh.user}@{self.ssh.host} -p {self.ssh.port} -t 'tmux attach -t {bootstrap_session}'"
            return SessionInfo(
                job_id=job_id,
                main_session=main_session,
                attach_main=attach_main,
                bootstrap_session=bootstrap_session,
                attach_bootstrap=attach_bootstrap
            )
        else:
            return SessionInfo(
                job_id=job_id,
                main_session=main_session,
                attach_main=attach_main
            )

    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None
    ) -> JobInfo:
        """
        Wait for a job to complete.

        Args:
            job_id: Job identifier
            poll_interval: Seconds between status checks
            timeout: Optional timeout in seconds

        Returns:
            Final JobInfo when job is complete

        Raises:
            JobError: Job failed or timeout exceeded
            ConnectionError: SSH connection failed
        """
        # Validate inputs
        poll_interval = validate_poll_interval(poll_interval)
        if timeout is not None:
            timeout = validate_timeout(int(timeout), min_value=1, max_value=86400)

        # Use trio's structured concurrency for timeout
        async def _wait_loop():
            while True:
                job_info = await self.get_job_status(job_id)

                if job_info.is_complete:
                    return job_info

                await trio.sleep(poll_interval)

        if timeout:
            # Use Trio's timeout context manager
            with trio.move_on_after(timeout) as cancel_scope:
                result = await _wait_loop()

            if cancel_scope.cancelled_caught:
                raise JobError(f"Timeout waiting for job {job_id} to complete")

            return result
        else:
            return await _wait_loop()

    async def copy_files(
        self,
        remote_path: str,
        local_path: str,
        recursive: bool = False
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
            ConnectionError: SSH connection failed
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
                    duration_seconds=duration
                )
            finally:
                sftp.close()

        except Exception as e:
            if isinstance(e, (ConnectionError, TransferError)):
                raise
            duration = time.time() - start_time
            return CopyResult(
                success=False,
                files_copied=0,
                total_bytes=0,
                duration_seconds=duration,
                error_message=str(e)
            )

    async def _copy_file(self, sftp, remote_path: str, local_path: str) -> int:
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

    async def _copy_directory(self, sftp, conn, remote_path: str, local_path: str) -> tuple[int, int]:
        """Copy directory recursively and return (files_copied, total_bytes).

        Uses Trio's structured concurrency to copy files in parallel.
        """
        # Get directory listing
        result = await _trio_wrap(conn.run)(f"find {remote_path} -type f", check=True)
        file_list = [f.strip() for f in result.stdout.split('\n') if f.strip()]

        files_copied = 0
        total_bytes = 0

        # Use Trio nursery for parallel file transfers
        async def copy_one_file(remote_file: str):
            nonlocal files_copied, total_bytes

            # Calculate relative path and local destination
            rel_path = os.path.relpath(remote_file, remote_path)
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
        self,
        local_path: str,
        remote_path: str,
        recursive: bool = False
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
            ConnectionError: SSH connection failed
            TransferError: File transfer failed
        """
        import time

        start_time = time.time()

        try:
            conn = await self._get_connection()

            # Check if local path exists
            local_path_obj = Path(local_path)
            if not local_path_obj.exists():
                raise TransferError(f"Local path not found: {local_path}")

            is_directory = local_path_obj.is_dir()

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
                    duration_seconds=duration
                )
            finally:
                sftp.close()

        except Exception as e:
            if isinstance(e, (ConnectionError, TransferError)):
                raise
            duration = time.time() - start_time
            return CopyResult(
                success=False,
                files_copied=0,
                total_bytes=0,
                duration_seconds=duration,
                error_message=str(e)
            )

    async def _upload_file(self, sftp, local_path: str, remote_path: str) -> int:
        """Upload single file and return bytes transferred."""
        # Create remote directory if needed
        remote_dir = os.path.dirname(remote_path)
        if remote_dir and remote_dir != '.':
            await self._create_remote_dir(sftp, remote_dir)

        # Get file size
        file_size = os.path.getsize(local_path)

        # Upload file
        await _trio_wrap(sftp.put)(local_path, remote_path)

        return file_size

    async def _create_remote_dir(self, sftp, remote_dir: str):
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

    async def _upload_directory(self, sftp, local_path: str, remote_path: str) -> tuple[int, int]:
        """Upload directory recursively and return (files_uploaded, total_bytes).

        Uses Trio's structured concurrency to upload files in parallel.
        """
        local_path_obj = Path(local_path)

        files_uploaded = 0
        total_bytes = 0

        # Collect all files first
        all_files = [f for f in local_path_obj.rglob('*') if f.is_file()]

        # Use Trio nursery for parallel file uploads
        async def upload_one_file(local_file: Path):
            nonlocal files_uploaded, total_bytes

            # Calculate relative path and remote destination
            rel_path = local_file.relative_to(local_path_obj)
            remote_file = f"{remote_path}/{rel_path}".replace('\\', '/')

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
        self,
        remote_path: str,
        local_path: str,
        recursive: bool = False
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

    async def close(self):
        """Close SSH connection."""
        if self._ssh_conn:
            self._ssh_conn.close()
            await _trio_wrap(self._ssh_conn.wait_closed)()
            self._ssh_conn = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
