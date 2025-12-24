"""Bifrost SDK - Python client for remote GPU execution and job management."""

import logging
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import paramiko
from shared.retry import retry
from shared.validation import validate_ssh_key_path, validate_timeout

from . import git_sync
from .types import (
    CopyResult,
    EnvironmentVariables,
    ExecResult,
    JobError,
    JobInfo,
    ProcessSpec,
    RemoteConfig,
    ServerInfo,
    SSHConnection,
    SSHConnectionError,
    TransferError,
)
from .validation import generate_job_id, validate_bootstrap_cmd

logger = logging.getLogger(__name__)


class BifrostClient:
    """
    Bifrost SDK client for remote GPU execution and job management.

    Provides programmatic access to all Bifrost functionality:
    - Remote code execution (detached and synchronous)
    - Job monitoring and log streaming
    - File transfer operations
    - Git-based code deployment

    Example:
        client = BifrostClient("root@gpu.example.com:22")
        job = client.run_detached("python train_model.py")
        client.wait_for_completion(job.job_id)
        client.copy_files("/remote/outputs/", "./results/", recursive=True)
    """

    def __init__(
        self,
        ssh_connection: str,
        ssh_key_path: str,
        timeout: int = 30,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """
        Initialize Bifrost client.

        Args:
            ssh_connection: SSH connection string like 'user@host:port'
            ssh_key_path: Path to SSH private key (required)
            timeout: SSH connection timeout in seconds
            progress_callback: Optional callback for file transfer progress
        """
        # Validate and parse SSH connection
        self.ssh = SSHConnection.from_string(ssh_connection)

        # Validate inputs (validation helpers contain all assertions)
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
        self._ssh_client: paramiko.SSHClient | None = None

        # Track last deployed workspace for smart working_dir defaults
        self._last_workspace: str | None = None

    @retry(max_attempts=3, delay=2, backoff=2, exceptions=(Exception,))
    def _establish_connection(
        self, ssh_client: paramiko.SSHClient, private_key: paramiko.PKey | None = None
    ) -> None:
        """Establish SSH connection with retry logic and keepalive.

        This is at an external boundary (network I/O) so retry is appropriate.
        Retries up to 3 times with exponential backoff (2s, 4s, 8s) on connection errors.

        Enables SSH keepalive (30s interval) to prevent connection timeouts during
        long-running operations where no data flows over the SSH channel.

        Args:
            ssh_client: Paramiko SSH client
            private_key: Optional private key object

        Raises:
            Exception: If connection fails after all retry attempts
        """
        if private_key:
            ssh_client.connect(
                hostname=self.ssh.host,
                port=self.ssh.port,
                username=self.ssh.user,
                pkey=private_key,
                timeout=self.timeout,
            )
        else:
            # Use SSH agent or default keys
            ssh_client.connect(
                hostname=self.ssh.host,
                port=self.ssh.port,
                username=self.ssh.user,
                timeout=self.timeout,
            )

        # Tiger Style: Assert post-condition and enable keepalive
        transport = ssh_client.get_transport()
        assert transport is not None, "SSH transport must exist after connection"
        transport.set_keepalive(30)  # Send keepalive every 30 seconds

    def _get_ssh_client(self) -> paramiko.SSHClient:
        """Get or create SSH client connection.

        Tiger Style: Check transport is not just present but ACTIVE.
        Detects stale connections and reconnects automatically.
        """
        # Check if we need to (re)connect
        needs_reconnect = False
        if self._ssh_client is None:
            needs_reconnect = True
        else:
            transport = self._ssh_client.get_transport()
            if transport is None:
                needs_reconnect = True
            elif not transport.is_active():
                self.logger.debug("SSH transport inactive, reconnecting...")
                needs_reconnect = True

        if needs_reconnect:
            self._ssh_client = paramiko.SSHClient()
            self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            try:
                # Load SSH key if provided
                key_content = self._load_ssh_key()

                if key_content:
                    # Use provided key
                    import io

                    from paramiko import ECDSAKey, Ed25519Key, RSAKey

                    key_file = io.StringIO(key_content)
                    # Try different key types
                    private_key = None
                    for key_class in [RSAKey, Ed25519Key, ECDSAKey]:
                        try:
                            key_file.seek(0)
                            private_key = key_class.from_private_key(key_file)
                            break
                        except Exception:
                            continue

                    if not private_key:
                        raise SSHConnectionError(f"Could not parse SSH key at {self.ssh_key_path}")

                    # Connect with retry logic
                    self._establish_connection(self._ssh_client, private_key)
                else:
                    # Connect with retry logic (no key)
                    self._establish_connection(self._ssh_client, None)

                self.logger.debug(f"Connected to {self.ssh}")
            except Exception as e:
                raise SSHConnectionError(f"Failed to connect to {self.ssh}: {e}") from e

        # Tiger Style: Assert post-conditions
        assert self._ssh_client is not None, "SSH client must be initialized"
        transport = self._ssh_client.get_transport()
        assert transport is not None, "SSH transport must exist"
        assert transport.is_active(), "SSH transport must be active"

        return self._ssh_client

    def _load_ssh_key(self) -> str | None:
        """Load SSH private key content from file path."""
        if not self.ssh_key_path:
            return None

        import os

        key_path = os.path.expanduser(self.ssh_key_path)
        try:
            with open(key_path) as f:
                return f.read()
        except Exception as e:
            raise SSHConnectionError(f"Failed to load SSH key from {key_path}: {e}") from e

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

    def _generate_job_id(self, session_name: str | None) -> str:
        """Generate job ID with optional human-readable component.

        Uses validation helper which handles all assertions.

        Args:
            session_name: Optional human-readable session name

        Returns:
            Job ID string
        """
        return generate_job_id(session_name)

    def push(self, workspace_path: str, bootstrap_cmd: str | list[str] | None = None) -> str:
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

        ssh_client = self._get_ssh_client()
        self.logger.debug(f"📁 Deploying to workspace: {workspace_path}")

        # Deploy code (pure function)
        workspace_path = git_sync.deploy_code(ssh_client, self._remote_config, workspace_path)

        # Run bootstrap if specified (pure function)
        if bootstrap_cmd:
            git_sync.run_bootstrap(ssh_client, self._remote_config, workspace_path, bootstrap_cmd)

        # Assert output
        assert workspace_path, "push() returned empty workspace_path"

        # Track last deployed workspace for smart working_dir defaults
        self._last_workspace = workspace_path

        return workspace_path

    def exec(
        self,
        command: str,
        env: EnvironmentVariables | dict[str, str] | None = None,
        working_dir: str | None = None,
        timeout: float | None = None,
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
            timeout: Optional timeout in seconds for command execution

        Returns:
            ExecResult with stdout, stderr, exit_code

        Raises:
            SSHConnectionError: SSH connection failed or timeout exceeded
        """
        try:
            ssh_client = self._get_ssh_client()

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
            env_vars = None
            if env is not None:
                if isinstance(env, EnvironmentVariables):
                    env_vars = env
                elif isinstance(env, dict):
                    env_vars = EnvironmentVariables.from_dict(env)

            # Build command with environment and working directory
            full_command = self._build_command_with_env(command, working_dir, env_vars)

            # Execute command
            stdin, stdout, stderr = ssh_client.exec_command(full_command)

            # Set timeout on channel if specified
            if timeout is not None:
                stdout.channel.settimeout(timeout)

            try:
                exit_code = stdout.channel.recv_exit_status()
                return ExecResult(
                    stdout=stdout.read().decode(),
                    stderr=stderr.read().decode(),
                    exit_code=exit_code,
                )
            except TimeoutError as e:
                raise SSHConnectionError(f"Command timed out after {timeout}s: {command}") from e

        except Exception as e:
            if isinstance(e, SSHConnectionError):
                raise
            raise SSHConnectionError(f"Execution failed: {e}") from e

    def exec_stream(  # noqa: PLR1702 - streaming requires nested try/while/if for proper cleanup
        self,
        command: str,
        env: EnvironmentVariables | dict[str, str] | None = None,
        working_dir: str | None = None,
        timeout: float | None = None,
    ) -> Iterator[str]:
        """
        Execute command and stream output line-by-line in real-time.

        Like exec() but yields output as it's produced instead of waiting for completion.
        Useful for long-running commands like package installations.

        Args:
            command: Command to execute
            env: Environment variables to set (dict or EnvironmentVariables)
            working_dir: Working directory (defaults to ~/.bifrost/workspace/ if deployed)
            timeout: Optional timeout in seconds for command execution

        Yields:
            Lines of output (stdout and stderr interleaved) as they're produced

        Raises:
            SSHConnectionError: SSH connection failed or timeout exceeded
        """
        try:
            ssh_client = self._get_ssh_client()

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
            env_vars = None
            if env is not None:
                if isinstance(env, EnvironmentVariables):
                    env_vars = env
                elif isinstance(env, dict):
                    env_vars = EnvironmentVariables.from_dict(env)

            # Build command with environment and working directory
            full_command = self._build_command_with_env(command, working_dir, env_vars)

            # Open interactive channel to stream combined stdout/stderr in real-time
            transport = ssh_client.get_transport()
            if transport is None:
                raise SSHConnectionError("SSH transport is not available")

            channel = None
            try:
                channel = transport.open_session()
                channel.set_combine_stderr(True)
                channel.get_pty()

                # Set timeout on channel if specified
                if timeout is not None:
                    channel.settimeout(timeout)

                channel.exec_command(full_command)

                buffer = ""
                while True:
                    try:
                        if channel.recv_ready():
                            chunk = channel.recv(4096)
                            if not chunk:
                                break

                            text = chunk.decode(errors="replace")
                            buffer += text

                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                yield line.rstrip("\r")
                            continue

                        if channel.exit_status_ready():
                            break

                        time.sleep(0.1)
                    except TimeoutError as e:
                        raise SSHConnectionError(
                            f"Command timed out after {timeout}s: {command}"
                        ) from e

                # Drain any remaining data after command exit
                while channel.recv_ready():
                    chunk = channel.recv(4096)
                    if not chunk:
                        break
                    text = chunk.decode(errors="replace")
                    buffer += text
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        yield line.rstrip("\r")

                if buffer:
                    yield buffer.rstrip("\r")

                # Wait for command exit to propagate errors to caller
                channel.recv_exit_status()
            finally:
                if channel is not None:
                    channel.close()

        except Exception as e:
            if isinstance(e, SSHConnectionError):
                raise
            raise SSHConnectionError(f"Streaming execution failed: {e}") from e

    def expand_path(self, path: str) -> str:
        """Expand ~ and environment variables in path to absolute path.

        This is a convenience helper to expand paths on the remote machine.
        Eliminates the common pattern: client.exec(f"echo {path}").stdout.strip()

        Args:
            path: Path to expand (may contain ~ or env vars)

        Returns:
            Absolute expanded path on remote machine

        Raises:
            SSHConnectionError: SSH connection failed

        Example:
            workspace = client.push(workspace_path="~/.bifrost/workspaces/foo")
            workspace = client.expand_path(workspace)  # /home/user/.bifrost/workspaces/foo
        """
        assert path, "path must be non-empty string"

        result = self.exec(f"echo {path}")
        assert result.exit_code == 0, f"Failed to expand path: {result.stderr}"

        expanded = result.stdout.strip()
        assert expanded, f"Path expansion returned empty string for: {path}"

        return expanded

    # ========================================================================
    # v2 API: submit() and serve() - functions-over-classes pattern
    # ========================================================================

    def submit(
        self,
        spec: ProcessSpec,
        name: str,
        log_file: str | None = None,
        workspace: str | None = None,
    ) -> JobInfo:
        """Submit a job for execution in a tmux session.

        This is the new v2 API that returns a frozen JobInfo.
        Use with job_status(), job_wait(), job_logs(), job_kill() functions.

        Args:
            spec: ProcessSpec defining what to run
            name: Job name (used for tmux session name)
            log_file: Path to log file (default: ~/.bifrost/logs/{name}.log)
            workspace: Working directory (default: last pushed workspace or ~)

        Returns:
            JobInfo identifier (frozen dataclass)

        Raises:
            JobError: Job creation failed

        Example:
            from bifrost import BifrostClient
            from bifrost.types import ProcessSpec
            from bifrost.job import job_status, job_wait, job_stream_logs

            client = BifrostClient("root@gpu:22", ssh_key_path="~/.ssh/id_ed25519")
            job = client.submit(
                ProcessSpec(command="python", args=("train.py",)),
                name="training",
            )

            # Stream logs
            for line in job_stream_logs(client, job):
                print(line)

            # Wait for completion
            exit_code = job_wait(client, job)
        """
        assert spec is not None, "ProcessSpec required"
        assert name, "job name required"

        # Default workspace to last pushed or home
        if workspace is None:
            workspace = self._last_workspace or "~"

        # Expand tilde in workspace path - tmux -c doesn't expand ~ like bash does
        if workspace.startswith("~"):
            workspace = self.expand_path(workspace)

        # Default log file
        if log_file is None:
            log_file = f"~/.bifrost/logs/{name}.log"

        # Expand tilde in log_file for the same reason
        if log_file.startswith("~"):
            log_file = self.expand_path(log_file)

        # Ensure log directory exists
        self.exec(f"mkdir -p $(dirname {log_file})")

        # Build tmux session name
        session_name = f"bifrost-job-{name}"

        # Kill existing session if present
        self.exec(f"tmux kill-session -t {session_name} 2>/dev/null || true")

        # Expand tilde in ProcessSpec.cwd if present (ProcessSpec is frozen, so create new one)
        if spec.cwd and spec.cwd.startswith("~"):
            expanded_cwd = self.expand_path(spec.cwd)
            spec = ProcessSpec(
                command=spec.command,
                args=spec.args,
                cwd=expanded_cwd,
                env=spec.env,
                cuda_device_ids=spec.cuda_device_ids,
            )

        # Build command from ProcessSpec
        full_cmd = spec.build_command()

        # Build tmux command with script for reliable logging
        # - script -efc: run command with exit code capture
        # - EXIT_CODE marker: for programmatic exit code detection
        escaped_cmd = full_cmd.replace("'", "'\\''")
        tmux_cmd = f"tmux new-session -d -s {session_name}"
        if workspace:
            tmux_cmd += f" -c {workspace}"
        tmux_cmd += f" 'script -efc \"{escaped_cmd}\" {log_file}; echo EXIT_CODE: $? >> {log_file}'"

        result = self.exec(tmux_cmd)
        if result.exit_code != 0:
            raise JobError(f"Failed to start job {name}: {result.stderr}")

        self.logger.info(f"Job started: {name} (session: {session_name})")

        return JobInfo(
            name=name,
            tmux_session=session_name,
            log_file=log_file,
            workspace=workspace,
        )

    def serve(
        self,
        spec: ProcessSpec,
        name: str,
        port: int,
        health_endpoint: str = "/health",
        log_file: str | None = None,
        workspace: str | None = None,
    ) -> ServerInfo:
        """Start a server process in a tmux session.

        This is the new v2 API that returns a frozen ServerInfo.
        Use with server_is_healthy(), server_wait_until_healthy(), server_stop() functions.

        Args:
            spec: ProcessSpec defining the server command
            name: Server name (used for tmux session name)
            port: Port the server listens on
            health_endpoint: Health check endpoint path (default: /health)
            log_file: Path to log file (default: ~/.bifrost/logs/{name}.log)
            workspace: Working directory (default: last pushed workspace or ~)

        Returns:
            ServerInfo identifier (frozen dataclass)

        Raises:
            JobError: Server creation failed

        Example:
            from bifrost import BifrostClient
            from bifrost.types import ProcessSpec
            from bifrost.server import server_wait_until_healthy, server_stop

            client = BifrostClient("root@gpu:22", ssh_key_path="~/.ssh/id_ed25519")
            server = client.serve(
                ProcessSpec(
                    command="python",
                    args=("-m", "sglang.launch_server", "--model", "meta-llama/Llama-3.1-8B"),
                ),
                name="sglang",
                port=30000,
            )

            # Wait for server to be healthy
            if server_wait_until_healthy(client, server, timeout=300):
                print(f"Server ready at localhost:{server.port}")

            # Stop server when done
            server_stop(client, server)
        """
        assert spec is not None, "ProcessSpec required"
        assert name, "server name required"
        assert port > 0, "port must be positive"

        # Default workspace to last pushed or home
        if workspace is None:
            workspace = self._last_workspace or "~"

        # Expand tilde in workspace path - tmux -c doesn't expand ~ like bash does
        if workspace.startswith("~"):
            workspace = self.expand_path(workspace)

        # Default log file
        if log_file is None:
            log_file = f"~/.bifrost/logs/{name}.log"

        # Expand tilde in log_file for the same reason
        if log_file.startswith("~"):
            log_file = self.expand_path(log_file)

        # Ensure log directory exists
        self.exec(f"mkdir -p $(dirname {log_file})")

        # Build tmux session name
        session_name = f"bifrost-server-{name}"

        # Kill existing session if present
        self.exec(f"tmux kill-session -t {session_name} 2>/dev/null || true")

        # Expand tilde in ProcessSpec.cwd if present (ProcessSpec is frozen, so create new one)
        if spec.cwd and spec.cwd.startswith("~"):
            expanded_cwd = self.expand_path(spec.cwd)
            spec = ProcessSpec(
                command=spec.command,
                args=spec.args,
                cwd=expanded_cwd,
                env=spec.env,
                cuda_device_ids=spec.cuda_device_ids,
            )

        # Build command from ProcessSpec
        full_cmd = spec.build_command()

        # Build tmux command with script for reliable logging
        escaped_cmd = full_cmd.replace("'", "'\\''")
        tmux_cmd = f"tmux new-session -d -s {session_name}"
        if workspace:
            tmux_cmd += f" -c {workspace}"
        tmux_cmd += f" 'script -efc \"{escaped_cmd}\" {log_file}'"

        result = self.exec(tmux_cmd)
        if result.exit_code != 0:
            raise JobError(f"Failed to start server {name}: {result.stderr}")

        self.logger.info(f"Server started: {name} (session: {session_name}, port: {port})")

        return ServerInfo(
            name=name,
            tmux_session=session_name,
            log_file=log_file,
            port=port,
            health_endpoint=health_endpoint,
            workspace=workspace,
        )

    def copy_files(self, remote_path: str, local_path: str, recursive: bool = False) -> CopyResult:
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
        start_time = time.time()

        try:
            ssh_client = self._get_ssh_client()

            # FOOTGUN WARNING: Tilde expansion behavior differs between shell and SFTP!
            # - Shell commands (exec_command) expand ~ automatically via bash
            # - SFTP operations (stat, get, put) treat ~ as a LITERAL directory name
            #
            # This method uses shell commands to check file existence (works with ~),
            # but then calls _copy_file() which uses SFTP (must expand ~ manually).
            # See _copy_file() for the tilde expansion logic.

            # Check if remote path exists (expand tilde if present)
            if remote_path.startswith("~/"):
                # For tilde paths, don't use quotes so bash can expand ~
                stdin, stdout, stderr = ssh_client.exec_command(f"test -e {remote_path}")
            else:
                # For other paths, use quotes for safety
                stdin, stdout, stderr = ssh_client.exec_command(f"test -e '{remote_path}'")
            if stdout.channel.recv_exit_status() != 0:
                raise TransferError(f"Remote path not found: {remote_path}")

            # Check if remote path is directory (expand tilde if present)
            if remote_path.startswith("~/"):
                stdin, stdout, stderr = ssh_client.exec_command(f"test -d {remote_path}")
            else:
                stdin, stdout, stderr = ssh_client.exec_command(f"test -d '{remote_path}'")
            is_directory = stdout.channel.recv_exit_status() == 0

            if is_directory and not recursive:
                raise TransferError(f"{remote_path} is a directory. Use recursive=True")

            # Create SFTP client
            sftp = ssh_client.open_sftp()

            try:
                files_copied = 0
                total_bytes = 0

                if is_directory:
                    files_copied, total_bytes = self._copy_directory(
                        sftp, ssh_client, remote_path, local_path
                    )
                else:
                    total_bytes = self._copy_file(sftp, remote_path, local_path)
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

    def _copy_file(self, sftp: paramiko.SFTPClient, remote_path: str, local_path: str) -> int:
        """Copy single file and return bytes transferred.

        IMPORTANT: This method receives paths that may contain tilde (~).
        SFTP protocol does NOT expand tilde - it treats it as a literal directory.
        We must manually expand ~ to /root before calling sftp.stat() and sftp.get().
        """
        # Ensure local directory exists
        local_dir = Path(local_path).parent
        local_dir.mkdir(parents=True, exist_ok=True)

        # CRITICAL: Expand tilde for SFTP operations
        # SFTP treats "~/foo" as literal path with directory named "~"
        # Shell commands expand it to "/root/foo" (or appropriate home dir)
        if remote_path.startswith("~/"):
            remote_path = remote_path.replace("~", "/root", 1)

        # Get file size
        file_size = sftp.stat(remote_path).st_size

        # Define progress callback wrapper
        def progress_wrapper(transferred: int, total: int) -> None:
            if self.progress_callback:
                self.progress_callback("file", transferred, total)

        # Copy file with optional progress reporting
        if file_size > 1024 * 1024 and self.progress_callback:  # Files >1MB
            sftp.get(remote_path, local_path, callback=progress_wrapper)
        else:
            sftp.get(remote_path, local_path)

        return file_size

    def _copy_directory(
        self,
        sftp: paramiko.SFTPClient,
        ssh_client: paramiko.SSHClient,
        remote_path: str,
        local_path: str,
    ) -> tuple[int, int]:
        """Copy directory recursively and return (files_copied, total_bytes)."""
        # Get directory listing (expand tilde if present)
        if remote_path.startswith("~/"):
            stdin, stdout, stderr = ssh_client.exec_command(f"find {remote_path} -type f")
        else:
            stdin, stdout, stderr = ssh_client.exec_command(f"find '{remote_path}' -type f")
        if stdout.channel.recv_exit_status() != 0:
            error = stderr.read().decode()
            raise TransferError(f"Failed to list directory contents: {error}")

        file_list = [f.strip() for f in stdout.read().decode().split("\n") if f.strip()]

        files_copied = 0
        total_bytes = 0

        # Convert remote_path to absolute form for proper relative path calculation
        if remote_path.startswith("~/"):
            abs_remote_path = remote_path.replace("~", "/root", 1)
        else:
            abs_remote_path = remote_path

        for remote_file in file_list:
            # Calculate relative path and local destination
            rel_path = os.path.relpath(remote_file, abs_remote_path)
            local_file = os.path.join(local_path, rel_path)

            # Copy file
            try:
                file_bytes = self._copy_file(sftp, remote_file, local_file)
                files_copied += 1
                total_bytes += file_bytes
            except Exception as e:
                self.logger.warning(f"Failed to copy {rel_path}: {e}")

        return files_copied, total_bytes

    def upload_files(
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
        start_time = time.time()

        try:
            ssh_client = self._get_ssh_client()

            # Check if local path exists
            local_path_obj = Path(local_path)
            if not local_path_obj.exists():
                raise TransferError(f"Local path not found: {local_path}")

            is_directory = local_path_obj.is_dir()

            if is_directory and not recursive:
                raise TransferError(f"{local_path} is a directory. Use recursive=True")

            # Create SFTP client
            sftp = ssh_client.open_sftp()

            try:
                files_uploaded = 0
                total_bytes = 0

                if is_directory:
                    files_uploaded, total_bytes = self._upload_directory(
                        sftp, ssh_client, local_path, remote_path
                    )
                else:
                    total_bytes = self._upload_file(sftp, local_path, remote_path)
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

    def _create_remote_dir(self, sftp: paramiko.SFTPClient, remote_dir: str) -> None:
        """Create remote directory recursively."""
        try:
            sftp.stat(remote_dir)  # Check if directory exists
        except FileNotFoundError:
            # Directory doesn't exist, create it
            parent_dir = os.path.dirname(remote_dir)
            if parent_dir and parent_dir != remote_dir:  # Avoid infinite recursion
                self._create_remote_dir(sftp, parent_dir)
            try:
                sftp.mkdir(remote_dir)
            except OSError:
                # Directory might have been created by another process
                pass

    def _upload_file(self, sftp: paramiko.SFTPClient, local_path: str, remote_path: str) -> int:
        """Upload single file and return bytes transferred."""
        # Create remote directory if needed
        remote_dir = os.path.dirname(remote_path)
        if remote_dir and remote_dir != ".":
            # Create directory structure recursively
            self._create_remote_dir(sftp, remote_dir)

        # Get file size
        file_size = os.path.getsize(local_path)

        # Define progress callback wrapper
        def progress_wrapper(transferred: int, total: int) -> None:
            if self.progress_callback:
                self.progress_callback("file", transferred, total)

        # Upload file with optional progress reporting
        if file_size > 1024 * 1024 and self.progress_callback:  # Files >1MB
            sftp.put(local_path, remote_path, callback=progress_wrapper)
        else:
            sftp.put(local_path, remote_path)

        return file_size

    def _upload_directory(
        self,
        sftp: paramiko.SFTPClient,
        ssh_client: paramiko.SSHClient,
        local_path: str,
        remote_path: str,
    ) -> tuple[int, int]:
        """Upload directory recursively and return (files_uploaded, total_bytes)."""
        local_path_obj = Path(local_path)

        files_uploaded = 0
        total_bytes = 0

        # Walk through local directory
        for local_file in local_path_obj.rglob("*"):
            if local_file.is_file():
                # Calculate relative path and remote destination
                rel_path = local_file.relative_to(local_path_obj)
                remote_file = f"{remote_path}/{rel_path}".replace("\\", "/")

                # Upload file
                try:
                    file_bytes = self._upload_file(sftp, str(local_file), remote_file)
                    files_uploaded += 1
                    total_bytes += file_bytes
                except Exception as e:
                    self.logger.warning(f"Failed to upload {rel_path}: {e}")

        return files_uploaded, total_bytes

    def download_files(
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
        return self.copy_files(remote_path, local_path, recursive)

    def close(self) -> None:
        """Close SSH connection."""
        if self._ssh_client:
            self._ssh_client.close()
            self._ssh_client = None

    def __enter__(self) -> "BifrostClient":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.close()
