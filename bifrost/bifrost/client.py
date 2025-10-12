"""Bifrost SDK - Python client for remote GPU execution and job management."""

import paramiko
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Iterator, List, Dict
import logging

from .types import (
    SSHConnection, JobInfo, JobStatus, CopyResult, ConnectionError, JobError, TransferError,
    RemoteConfig, ExecResult
)
from .deploy import GitDeployment
from . import git_sync


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
        # Parse SSH connection
        self.ssh = SSHConnection.from_string(ssh_connection)

        # Create RemoteConfig
        import os
        self._remote_config = RemoteConfig(
            host=self.ssh.host,
            port=self.ssh.port,
            user=self.ssh.user,
            key_path=os.path.expanduser(ssh_key_path)
        )

        self.timeout = timeout
        self.ssh_key_path = ssh_key_path
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)

        # Connection will be established on-demand
        self._ssh_client: Optional[paramiko.SSHClient] = None
    
    def _get_ssh_client(self) -> paramiko.SSHClient:
        """Get or create SSH client connection."""
        if self._ssh_client is None or self._ssh_client.get_transport() is None:
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
                    
                    self._ssh_client.connect(
                        hostname=self.ssh.host,
                        port=self.ssh.port, 
                        username=self.ssh.user,
                        pkey=private_key,
                        timeout=self.timeout
                    )
                else:
                    # Use SSH agent or default keys
                    self._ssh_client.connect(
                        hostname=self.ssh.host,
                        port=self.ssh.port, 
                        username=self.ssh.user,
                        timeout=self.timeout
                    )
                    
                self.logger.info(f"Connected to {self.ssh}")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to {self.ssh}: {e}")
        
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
                                env: Optional[Dict[str, str]]) -> str:
        """Build command with environment variables and working directory.

        Args:
            command: Command to execute
            working_dir: Directory to run in
            env: Environment variables

        Returns:
            Full command string with cd and exports
        """
        import shlex

        parts = []

        # Change directory
        parts.append(f"cd {working_dir}")

        # Export environment variables
        if env:
            for key, value in env.items():
                # Basic validation
                if not key.isidentifier():
                    raise ValueError(f"Invalid env var name: {key}")
                # Use shell quoting for safety
                parts.append(f"export {key}={shlex.quote(value)}")

        # Execute command
        parts.append(command)

        return " && ".join(parts)

    def _generate_job_id(self, session_name: Optional[str]) -> str:
        """Generate job ID with optional human-readable component.

        Uses timestamp + random suffix to avoid collisions.

        Args:
            session_name: Optional human-readable session name

        Returns:
            Job ID string
        """
        import secrets
        from datetime import datetime

        # Assert input
        if session_name is not None:
            assert isinstance(session_name, str), "session_name must be string"
            assert len(session_name) > 0, "session_name cannot be empty"
            assert len(session_name) < 100, f"session_name too long: {len(session_name)}"

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = secrets.token_hex(4)  # 8 hex chars, ~4B combinations

        if session_name:
            # Sanitize session name
            safe_name = "".join(c if c.isalnum() or c == "-" else "_"
                               for c in session_name)
            assert len(safe_name) > 0, "Sanitized session_name is empty"
            job_id = f"{safe_name}-{timestamp}-{random_suffix}"
        else:
            # Auto-generate from timestamp
            job_id = f"job-{timestamp}-{random_suffix}"

        # Assert output
        assert len(job_id) > 0, "Generated empty job_id"
        assert len(job_id) < 256, f"Job ID too long: {len(job_id)} chars"
        assert "-" in job_id, "Job ID missing separators"

        return job_id

    def push(self, bootstrap_cmd: Optional[str] = None) -> str:
        """Deploy code to remote workspace.

        Args:
            bootstrap_cmd: Optional explicit command to install dependencies
                          (e.g., "uv sync --frozen" or "pip install -r requirements.txt")

        Returns:
            Path to deployed workspace

        Raises:
            ConnectionError: SSH connection failed
            RuntimeError: Deployment failed
        """
        # Assert input
        if bootstrap_cmd is not None:
            assert isinstance(bootstrap_cmd, str) and len(bootstrap_cmd) > 0, \
                "bootstrap_cmd must be non-empty string"

        ssh_client = self._get_ssh_client()
        workspace_path = "~/.bifrost/workspace"

        # Deploy code (pure function)
        workspace_path = git_sync.deploy_code(ssh_client, self._remote_config, workspace_path)

        # Install dependencies if specified (pure function)
        if bootstrap_cmd:
            git_sync.install_dependencies(ssh_client, self._remote_config, workspace_path, bootstrap_cmd)

        # Assert output
        assert workspace_path, "push() returned empty workspace_path"
        return workspace_path
    
    def exec(self, command: str, env: Optional[Dict[str, str]] = None, working_dir: Optional[str] = None) -> ExecResult:
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
            env: Environment variables to set
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

            # Build command with environment and working directory
            full_command = self._build_command_with_env(command, working_dir, env)

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
    
    def deploy(self, command: str, bootstrap_cmd: Optional[str] = None,
               env: Optional[Dict[str, str]] = None) -> ExecResult:
        """Deploy code and execute command.

        Equivalent to: push(bootstrap_cmd) + exec(command, env)

        Args:
            command: Command to execute
            bootstrap_cmd: Optional dependency installation command
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
        bootstrap_cmd: Optional[str] = None,
        bootstrap_timeout: int = 600,
        env: Optional[Dict[str, str]] = None,
        session_name: Optional[str] = None,
        no_deploy: bool = False
    ) -> JobInfo:
        """Execute command as detached background job.

        Args:
            command: Command to execute
            bootstrap_cmd: Optional bootstrap command (runs in separate session)
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
        # Assert inputs
        assert isinstance(command, str) and len(command) > 0, "command must be non-empty string"
        assert isinstance(bootstrap_timeout, int) and bootstrap_timeout > 0, \
            f"bootstrap_timeout must be positive int, got {bootstrap_timeout}"

        # Generate job ID
        job_id = self._generate_job_id(session_name)

        try:
            if not no_deploy:
                # Get configured SSH client (with credentials)
                client = self._get_ssh_client()

                # Use GitDeployment for detached execution
                deployment = GitDeployment(self.ssh.user, self.ssh.host, self.ssh.port)
                actual_job_id = deployment.deploy_and_execute_detached(client, command, env)

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
            
            metadata = json.loads(stdout.read().decode())
            
            # Get current status
            status_cmd = f"cat ~/.bifrost/jobs/{job_id}/status 2>/dev/null"
            stdin, stdout, stderr = ssh_client.exec_command(status_cmd) 
            status_str = stdout.read().decode().strip()
            
            # Get end time if available
            end_time = None
            end_time_cmd = f"cat ~/.bifrost/jobs/{job_id}/end_time 2>/dev/null"
            stdin, stdout, stderr = ssh_client.exec_command(end_time_cmd)
            if stdout.channel.recv_exit_status() == 0:
                end_time_str = stdout.read().decode().strip()
                if end_time_str:
                    end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
            
            # Calculate runtime
            start_time = datetime.fromisoformat(metadata['start_time'].replace('Z', '+00:00'))
            runtime_seconds = None
            if end_time:
                runtime_seconds = (end_time - start_time).total_seconds()
            else:
                runtime_seconds = (datetime.now().astimezone() - start_time).total_seconds()
            
            return JobInfo(
                job_id=job_id,
                status=JobStatus(status_str) if status_str else JobStatus.PENDING,
                command=metadata.get('command', ''),
                start_time=start_time,
                end_time=end_time,
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

    def get_session_info(self, job_id: str) -> Dict[str, str]:
        """Get tmux session information for a job.

        Returns:
            Dict with session names and attach commands
        """
        job = self.get_job_status(job_id)

        info = {
            "job_id": job_id,
            "main_session": job.tmux_session or f"bifrost-{job_id}",
            "attach_main": f"ssh {self.ssh.user}@{self.ssh.host} -p {self.ssh.port} -t 'tmux attach -t {job.tmux_session or f'bifrost-{job_id}'}'"
        }

        if job.bootstrap_session:
            info["bootstrap_session"] = job.bootstrap_session
            info["attach_bootstrap"] = f"ssh {self.ssh.user}@{self.ssh.host} -p {self.ssh.port} -t 'tmux attach -t {job.bootstrap_session}'"

        return info

    def wait_for_completion(self, job_id: str, poll_interval: float = 5.0, timeout: Optional[float] = None) -> JobInfo:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job identifier
            poll_interval: Seconds between status checks
            timeout: Optional timeout in seconds
            
        Returns:
            Final JobInfo when job completes
            
        Raises:
            JobError: Job failed or timeout exceeded
            ConnectionError: SSH connection failed
        """
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
        """Copy single file and return bytes transferred."""
        # Ensure local directory exists
        local_dir = Path(local_path).parent
        local_dir.mkdir(parents=True, exist_ok=True)
        
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