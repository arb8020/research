"""Bifrost SDK - Python client for remote GPU execution and job management."""

import paramiko
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Iterator, List, Dict, Union
import logging

from shared.validation import validate_ssh_key_path, validate_timeout
from shared.retry import retry
from .types import (
    SSHConnection, JobInfo, JobStatus, CopyResult, ConnectionError, JobError, TransferError,
    RemoteConfig, ExecResult, EnvironmentVariables, SessionInfo, JobMetadata
)
from .deploy import GitDeployment
from . import git_sync
from .validation import (
    generate_job_id, validate_bootstrap_cmd, validate_command,
    validate_environment_variables, validate_poll_interval
)


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
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
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
            key_path=validated_ssh_key_path
        )

        self.timeout = validated_timeout
        self.ssh_key_path = validated_ssh_key_path
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)

        # Connection will be established on-demand
        self._ssh_client: Optional[paramiko.SSHClient] = None
    
    @retry(max_attempts=3, delay=2, backoff=2, exceptions=(Exception,))
    def _establish_connection(self, ssh_client: paramiko.SSHClient, private_key=None) -> None:
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
                timeout=self.timeout
            )
        else:
            # Use SSH agent or default keys
            ssh_client.connect(
                hostname=self.ssh.host,
                port=self.ssh.port,
                username=self.ssh.user,
                timeout=self.timeout
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
                    from paramiko import RSAKey, Ed25519Key, ECDSAKey

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
                        raise ConnectionError(f"Could not parse SSH key at {self.ssh_key_path}")

                    # Connect with retry logic
                    self._establish_connection(self._ssh_client, private_key)
                else:
                    # Connect with retry logic (no key)
                    self._establish_connection(self._ssh_client, None)

                self.logger.debug(f"Connected to {self.ssh}")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to {self.ssh}: {e}")

        # Tiger Style: Assert post-conditions
        assert self._ssh_client is not None, "SSH client must be initialized"
        transport = self._ssh_client.get_transport()
        assert transport is not None, "SSH transport must exist"
        assert transport.is_active(), "SSH transport must be active"

        return self._ssh_client
    
    def _load_ssh_key(self) -> Optional[str]:
        """Load SSH private key content from file path."""
        if not self.ssh_key_path:
            return None

        import os
        key_path = os.path.expanduser(self.ssh_key_path)
        try:
            with open(key_path, 'r') as f:
                return f.read()
        except Exception as e:
            raise ConnectionError(f"Failed to load SSH key from {key_path}: {e}")

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

    def _generate_job_id(self, session_name: Optional[str]) -> str:
        """Generate job ID with optional human-readable component.

        Uses validation helper which handles all assertions.

        Args:
            session_name: Optional human-readable session name

        Returns:
            Job ID string
        """
        return generate_job_id(session_name)

    def push(
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

        ssh_client = self._get_ssh_client()
        self.logger.info(f"📁 Deploying to workspace: {workspace_path}")

        # Deploy code (pure function)
        workspace_path = git_sync.deploy_code(ssh_client, self._remote_config, workspace_path)

        # Run bootstrap if specified (pure function)
        if bootstrap_cmd:
            git_sync.run_bootstrap(ssh_client, self._remote_config, workspace_path, bootstrap_cmd)

        # Assert output
        assert workspace_path, "push() returned empty workspace_path"
        return workspace_path
    
    def exec(self, command: str, env: Optional[Union[EnvironmentVariables, Dict[str, str]]] = None, working_dir: Optional[str] = None) -> ExecResult:
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
            ssh_client = self._get_ssh_client()

            # Default to workspace if it exists
            if working_dir is None:
                default_dir = "~/.bifrost/workspace"
                stdin, stdout, stderr = ssh_client.exec_command(f"test -d {default_dir}")
                if stdout.channel.recv_exit_status() == 0:
                    working_dir = default_dir
                    self.logger.debug(f"Using default working directory: {default_dir}")
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
            stdin, stdout, stderr = ssh_client.exec_command(full_command)
            exit_code = stdout.channel.recv_exit_status()

            return ExecResult(
                stdout=stdout.read().decode(),
                stderr=stderr.read().decode(),
                exit_code=exit_code
            )

        except Exception as e:
            if isinstance(e, ConnectionError):
                raise
            raise ConnectionError(f"Execution failed: {e}")

    def exec_stream(self, command: str, env: Optional[Union[EnvironmentVariables, Dict[str, str]]] = None, working_dir: Optional[str] = None) -> Iterator[str]:
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
            ssh_client = self._get_ssh_client()

            # Default to workspace if it exists (same logic as exec)
            if working_dir is None:
                default_dir = "~/.bifrost/workspace"
                stdin, stdout, stderr = ssh_client.exec_command(f"test -d {default_dir}")
                if stdout.channel.recv_exit_status() == 0:
                    working_dir = default_dir
                    self.logger.debug(f"Using default working directory: {default_dir}")
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

            # Open interactive channel to stream combined stdout/stderr in real-time
            transport = ssh_client.get_transport()
            if transport is None:
                raise ConnectionError("SSH transport is not available")

            channel = None
            try:
                channel = transport.open_session()
                channel.set_combine_stderr(True)
                channel.get_pty()
                channel.exec_command(full_command)

                buffer = ""
                while True:
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
            if isinstance(e, ConnectionError):
                raise
            raise ConnectionError(f"Streaming execution failed: {e}")

    def expand_path(self, path: str) -> str:
        """Expand ~ and environment variables in path to absolute path.

        This is a convenience helper to expand paths on the remote machine.
        Eliminates the common pattern: client.exec(f"echo {path}").stdout.strip()

        Args:
            path: Path to expand (may contain ~ or env vars)

        Returns:
            Absolute expanded path on remote machine

        Raises:
            ConnectionError: SSH connection failed

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

    def deploy(self, command: str, bootstrap_cmd: Optional[Union[str, List[str]]] = None,
               env: Optional[Union[EnvironmentVariables, Dict[str, str]]] = None) -> ExecResult:
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
        workspace_path = self.push(bootstrap_cmd)
        return self.exec(command, env=env, working_dir=workspace_path)
    
    def run_detached(
        self,
        command: str,
        bootstrap_cmd: Optional[Union[str, List[str]]] = None,
        bootstrap_timeout: int = 600,
        env: Optional[Union[EnvironmentVariables, Dict[str, str]]] = None,
        session_name: Optional[str] = None,
        no_deploy: bool = False
    ) -> JobInfo:
        """Execute command as detached background job.

        Args:
            command: Command to execute
            bootstrap_cmd: Optional bootstrap command(s) - either single string or list of commands
            bootstrap_timeout: Max seconds to wait for bootstrap (default: 600 = 10 min)
            env: Environment variables
            session_name: Optional human-readable session name
            no_deploy: Skip git deployment (legacy)

        Returns:
            JobInfo with job details and session names

        Raises:
            ConnectionError: SSH connection failed
            JobError: Job creation failed
        """
        # Validate inputs (validation helpers contain all assertions)
        command = validate_command(command)
        validate_timeout(bootstrap_timeout, min_value=1, max_value=3600)
        if bootstrap_cmd is not None:
            bootstrap_cmd = validate_bootstrap_cmd(bootstrap_cmd)

        # Convert dict to EnvironmentVariables if needed
        env_vars = None
        if env is not None:
            if isinstance(env, EnvironmentVariables):
                env_vars = env
            elif isinstance(env, dict):
                env_vars = EnvironmentVariables.from_dict(env)

        # Generate job ID
        job_id = self._generate_job_id(session_name)

        try:
            if not no_deploy:
                # Get configured SSH client (with credentials)
                client = self._get_ssh_client()

                # Use GitDeployment for detached execution
                deployment = GitDeployment(self.ssh.user, self.ssh.host, self.ssh.port)
                # Convert EnvironmentVariables to dict for GitDeployment
                env_dict = env_vars.to_dict() if env_vars else None
                actual_job_id = deployment.deploy_and_execute_detached(client, command, env_dict)

                # Note: GitDeployment generates its own job_id, we use that one
                # In Phase 3 full implementation, we'd pass our job_id to it
                job_id = actual_job_id
            else:
                # TODO: Implement legacy detached mode
                raise JobError("Legacy detached mode not yet implemented in SDK")

            # Construct session names (for Phase 3, approximate them)
            main_session = f"bifrost-{job_id}"
            bootstrap_session = f"bifrost-{job_id}-bootstrap" if bootstrap_cmd else None

            # Assert output
            assert main_session, "Failed to start main session"
            assert job_id, "Failed to generate job_id"

            # Return job info
            return JobInfo(
                job_id=job_id,
                status=JobStatus.STARTING,
                command=command,
                tmux_session=main_session,
                bootstrap_session=bootstrap_session,
                start_time=datetime.now()
            )

        except Exception as e:
            raise JobError(f"Failed to create detached job: {e}")
    
    def get_job_status(self, job_id: str) -> JobInfo:
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
            ssh_client = self._get_ssh_client()

            # Get job metadata
            metadata_cmd = f"cat ~/.bifrost/jobs/{job_id}/metadata.json 2>/dev/null"
            stdin, stdout, stderr = ssh_client.exec_command(metadata_cmd)

            if stdout.channel.recv_exit_status() != 0:
                raise JobError(f"Job {job_id} not found")

            metadata_dict = json.loads(stdout.read().decode())
            # Parse metadata using frozen dataclass for validation
            metadata = JobMetadata.from_dict(metadata_dict)
            
            # Get current status (may be updated from metadata)
            status_cmd = f"cat ~/.bifrost/jobs/{job_id}/status 2>/dev/null"
            stdin, stdout, stderr = ssh_client.exec_command(status_cmd)
            status_str = stdout.read().decode().strip()

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
    
    def get_all_jobs(self) -> List[JobInfo]:
        """
        Get status of all jobs on the remote instance.
        
        Returns:
            List of JobInfo objects for all jobs
            
        Raises:
            ConnectionError: SSH connection failed
        """
        try:
            ssh_client = self._get_ssh_client()
            
            # Get list of job directories
            stdin, stdout, stderr = ssh_client.exec_command("ls -1 ~/.bifrost/jobs/ 2>/dev/null || echo ''")
            job_dirs = [d.strip() for d in stdout.read().decode().split('\n') if d.strip()]
            
            jobs = []
            for job_id in job_dirs:
                try:
                    job_info = self.get_job_status(job_id)
                    jobs.append(job_info)
                except JobError:
                    # Skip jobs that can't be read
                    continue
            
            return jobs
            
        except Exception as e:
            if isinstance(e, ConnectionError):
                raise
            raise ConnectionError(f"Failed to list jobs: {e}")
    
    def get_logs(self, job_id: str, lines: int = 100, log_type: str = "command") -> str:
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
            ssh_client = self._get_ssh_client()

            if log_type == "bootstrap":
                log_file = f"~/.bifrost/jobs/{job_id}/bootstrap.log"
            else:
                log_file = f"~/.bifrost/jobs/{job_id}/job.log"

            # Check if log file exists
            stdin, stdout, stderr = ssh_client.exec_command(f"test -f {log_file}")
            if stdout.channel.recv_exit_status() != 0:
                raise JobError(f"No {log_type} log found for job {job_id}")

            # Get last N lines
            tail_cmd = f"tail -n {lines} {log_file}"
            stdin, stdout, stderr = ssh_client.exec_command(tail_cmd)

            if stdout.channel.recv_exit_status() != 0:
                error = stderr.read().decode()
                raise JobError(f"Failed to read logs: {error}")

            return stdout.read().decode()

        except Exception as e:
            if isinstance(e, (ConnectionError, JobError)):
                raise
            raise JobError(f"Failed to get job logs: {e}")

    
    def follow_job_logs(self, job_id: str) -> Iterator[str]:
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
            ssh_client = self._get_ssh_client()
            
            log_file = f"~/.bifrost/jobs/{job_id}/job.log"
            
            # Use tail -f to follow the log file
            tail_cmd = f"tail -f {log_file}"
            stdin, stdout, stderr = ssh_client.exec_command(tail_cmd)
            
            # Stream output line by line
            for line in iter(stdout.readline, ""):
                yield line.rstrip('\n')
                
        except Exception as e:
            if isinstance(e, ConnectionError):
                raise
            raise JobError(f"Failed to follow job logs: {e}")
    
    def list_sessions(self) -> List[str]:
        """List all bifrost tmux sessions on remote.

        Returns:
            List of tmux session names
        """
        ssh_client = self._get_ssh_client()

        stdin, stdout, stderr = ssh_client.exec_command("tmux list-sessions -F '#{session_name}' 2>/dev/null || echo ''")
        if stdout.channel.recv_exit_status() != 0:
            return []

        sessions = stdout.read().decode().strip().split('\n')
        # Filter to bifrost sessions only
        return [s for s in sessions if s.startswith('bifrost-') and s]

    def get_session_info(self, job_id: str) -> SessionInfo:
        """Get tmux session information for a job.

        Returns:
            SessionInfo with session names and attach commands
        """
        job = self.get_job_status(job_id)

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

    def wait_for_completion(self, job_id: str, poll_interval: float = 5.0, timeout: Optional[float] = None) -> JobInfo:
        """
        Wait for a job to complete.

        Args:
            job_id: Job identifier
            poll_interval: Seconds between status checks
            timeout: Optional timeout in seconds

        Returns:
            Final JobInfo when job_info is complete

        Raises:
            JobError: Job failed or timeout exceeded
            ConnectionError: SSH connection failed
        """
        # Validate inputs (validation helpers contain all assertions)
        poll_interval = validate_poll_interval(poll_interval)
        if timeout is not None:
            timeout = validate_timeout(int(timeout), min_value=1, max_value=86400)

        start_time = time.time()

        while True:
            job_info = self.get_job_status(job_id)

            if job_info.is_complete:
                return job_info

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise JobError(f"Timeout waiting for job {job_id} to complete")

            time.sleep(poll_interval)
    
    def copy_files(
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
            if remote_path.startswith('~/'):
                # For tilde paths, don't use quotes so bash can expand ~
                stdin, stdout, stderr = ssh_client.exec_command(f"test -e {remote_path}")
            else:
                # For other paths, use quotes for safety
                stdin, stdout, stderr = ssh_client.exec_command(f"test -e '{remote_path}'")
            if stdout.channel.recv_exit_status() != 0:
                raise TransferError(f"Remote path not found: {remote_path}")
            
            # Check if remote path is directory (expand tilde if present)
            if remote_path.startswith('~/'):
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
                    files_copied, total_bytes = self._copy_directory(sftp, ssh_client, remote_path, local_path)
                else:
                    total_bytes = self._copy_file(sftp, remote_path, local_path)
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
    
    def _copy_file(self, sftp, remote_path: str, local_path: str) -> int:
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
        if remote_path.startswith('~/'):
            remote_path = remote_path.replace('~', '/root', 1)

        # Get file size
        file_size = sftp.stat(remote_path).st_size
        
        # Define progress callback wrapper
        def progress_wrapper(transferred, total):
            if self.progress_callback:
                self.progress_callback("file", transferred, total)
        
        # Copy file with optional progress reporting
        if file_size > 1024 * 1024 and self.progress_callback:  # Files >1MB
            sftp.get(remote_path, local_path, callback=progress_wrapper)
        else:
            sftp.get(remote_path, local_path)
        
        return file_size
    
    def _copy_directory(self, sftp, ssh_client, remote_path: str, local_path: str) -> tuple[int, int]:
        """Copy directory recursively and return (files_copied, total_bytes)."""
        # Get directory listing (expand tilde if present)
        if remote_path.startswith('~/'):
            stdin, stdout, stderr = ssh_client.exec_command(f"find {remote_path} -type f")
        else:
            stdin, stdout, stderr = ssh_client.exec_command(f"find '{remote_path}' -type f")
        if stdout.channel.recv_exit_status() != 0:
            error = stderr.read().decode()
            raise TransferError(f"Failed to list directory contents: {error}")
        
        file_list = [f.strip() for f in stdout.read().decode().split('\n') if f.strip()]
        
        files_copied = 0
        total_bytes = 0
        
        # Convert remote_path to absolute form for proper relative path calculation
        if remote_path.startswith('~/'):
            abs_remote_path = remote_path.replace('~', '/root', 1)
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
                    files_uploaded, total_bytes = self._upload_directory(sftp, ssh_client, local_path, remote_path)
                else:
                    total_bytes = self._upload_file(sftp, local_path, remote_path)
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
    
    def _create_remote_dir(self, sftp, remote_dir: str):
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
    
    def _upload_file(self, sftp, local_path: str, remote_path: str) -> int:
        """Upload single file and return bytes transferred."""
        # Create remote directory if needed
        remote_dir = os.path.dirname(remote_path)
        if remote_dir and remote_dir != '.':
            # Create directory structure recursively
            self._create_remote_dir(sftp, remote_dir)
        
        # Get file size
        file_size = os.path.getsize(local_path)
        
        # Define progress callback wrapper
        def progress_wrapper(transferred, total):
            if self.progress_callback:
                self.progress_callback("file", transferred, total)
        
        # Upload file with optional progress reporting
        if file_size > 1024 * 1024 and self.progress_callback:  # Files >1MB
            sftp.put(local_path, remote_path, callback=progress_wrapper)
        else:
            sftp.put(local_path, remote_path)
        
        return file_size
    
    def _upload_directory(self, sftp, ssh_client, local_path: str, remote_path: str) -> tuple[int, int]:
        """Upload directory recursively and return (files_uploaded, total_bytes)."""
        local_path_obj = Path(local_path)
        
        files_uploaded = 0
        total_bytes = 0
        
        # Walk through local directory
        for local_file in local_path_obj.rglob('*'):
            if local_file.is_file():
                # Calculate relative path and remote destination
                rel_path = local_file.relative_to(local_path_obj)
                remote_file = f"{remote_path}/{rel_path}".replace('\\', '/')
                
                # Upload file
                try:
                    file_bytes = self._upload_file(sftp, str(local_file), remote_file)
                    files_uploaded += 1
                    total_bytes += file_bytes
                except Exception as e:
                    self.logger.warning(f"Failed to upload {rel_path}: {e}")
        
        return files_uploaded, total_bytes
    
    def download_files(
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
        return self.copy_files(remote_path, local_path, recursive)
    
    def close(self):
        """Close SSH connection."""
        if self._ssh_client:
            self._ssh_client.close()
            self._ssh_client = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
