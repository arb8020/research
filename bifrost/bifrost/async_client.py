"""Bifrost Async SDK - Trio-based async client for remote GPU execution."""

import logging
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Literal

import asyncssh
import trio
import trio_asyncio
from infra_utils.validation import validate_ssh_key_path, validate_timeout

from . import git_sync
from .service_launch import build_detached_service_launch_command
from .types import (
    CopyResult,
    EnvironmentVariables,
    ExecResult,
    ObservedProcessHandle,
    OutputSink,
    ProcessOutputLine,
    ProcessSpec,
    ProcessState,
    PythonProjectMaterialization,
    ReadinessProbe,
    RemoteConfig,
    ServerInfo,
    ServiceHandle,
    ServiceSpec,
    ServiceState,
    SSHConnection,
    SSHConnectionError,
    TransferError,
    WorkspaceHandle,
    WorkspaceMaterializationSpec,
    append_lifecycle_event,
    create_jsonl_event_stream,
)
from .validation import validate_bootstrap_cmd

logger = logging.getLogger(__name__)


def _trio_wrap(coro_func: Callable) -> Callable:
    """Helper to wrap asyncio coroutines for trio-asyncio.

    Usage: await _trio_wrap(conn.run)(args, kwargs)
    """
    return trio_asyncio.aio_as_trio(coro_func)


async def _close_sftp_client(sftp: object) -> None:
    """Best-effort close for Paramiko-like and AsyncSSH-like SFTP clients."""
    close = getattr(sftp, "close", None)
    if callable(close):
        result = close()
        if hasattr(result, "__await__"):
            await result
        return

    exit_method = getattr(sftp, "exit", None)
    if callable(exit_method):
        result = exit_method()
        if hasattr(result, "__await__"):
            await result


def _check_dirty_workspace_sync(*, allow_dirty: bool) -> None:
    """Fail fast if the local git workspace is dirty and allow_dirty is false."""

    if allow_dirty:
        return

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        return

    dirty_files = [line[3:].strip() for line in status.stdout.splitlines() if line.strip()]
    if dirty_files:
        raise RuntimeError(
            f"Workspace has {len(dirty_files)} uncommitted/untracked file(s). "
            f"Deploy with allow_dirty=True to proceed anyway. "
            f"Files: {', '.join(dirty_files[:5])}"
            + (f" and {len(dirty_files) - 5} more" if len(dirty_files) > 5 else "")
        )


def _create_git_bundle_sync() -> tuple[str, str]:
    """Create a git bundle for HEAD and return (bundle_path, commit_hash)."""

    hash_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit_hash = hash_result.stdout.strip()
    if hash_result.returncode != 0 or not commit_hash:
        raise RuntimeError("Git rev-parse HEAD failed; are you in a git repository?")

    with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as bundle_file:
        bundle_path = bundle_file.name

    result = subprocess.run(
        ["git", "bundle", "create", bundle_path, "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        os.unlink(bundle_path)
        raise RuntimeError(
            f"Git bundle create failed: {result.stderr}\n\nNot in a git repository. Run 'git init' first."
        )
    return bundle_path, commit_hash


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
        self._owned_asyncio_loop_cm: Any | None = None

        # Track last deployed workspace for smart working_dir defaults
        self._last_workspace: str | None = None

    @property
    def backend(self) -> str:
        return "ssh"

    def current_workspace(self) -> WorkspaceHandle | None:
        if self._last_workspace is None:
            return None
        return WorkspaceHandle(root=self._last_workspace, backend=self.backend)

    async def _ensure_asyncio_loop(self) -> None:
        """Ensure a trio-asyncio loop exists for asyncssh bridging.

        AsyncBifrostClient is used from plain Trio callers in `rollouts`, not
        only from `trio_asyncio.run(...)`. Own a loop when there isn't already
        one in context so asyncssh operations have a live asyncio bridge.
        """
        if trio_asyncio.current_loop.get() is not None:
            return
        if self._owned_asyncio_loop_cm is None:
            loop_cm = trio_asyncio.open_loop()
            await loop_cm.__aenter__()
            self._owned_asyncio_loop_cm = loop_cm

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
        await self._ensure_asyncio_loop()

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

    async def materialize(self, spec: WorkspaceMaterializationSpec) -> WorkspaceHandle:
        workspace_root = spec.requested_root or self._last_workspace
        assert workspace_root is not None, "SSH materialization requires a requested_root"
        bootstrap_cmd: str | list[str] | None = None
        if spec.bootstrap_commands:
            bootstrap_cmd = list(spec.bootstrap_commands)
        return await self.materialize_workspace(
            workspace_path=workspace_root,
            bootstrap_cmd=bootstrap_cmd,
            allow_dirty=spec.allow_dirty,
            extra_python_projects=spec.extra_python_projects,
        )

    async def _materialize_extra_python_projects(
        self,
        *,
        conn: asyncssh.SSHClientConnection,
        workspace_root: str,
        extra_python_projects: tuple[PythonProjectMaterialization, ...],
        allow_dirty: bool,
    ) -> None:
        if not extra_python_projects:
            return

        sftp = await _trio_wrap(conn.start_sftp_client)()
        try:
            for project in extra_python_projects:
                local_root = str(Path(project.local_root).expanduser().resolve())
                await trio.to_thread.run_sync(
                    lambda repo_root=local_root: git_sync.ensure_clean_git_repo(
                        repo_root=repo_root, allow_dirty=allow_dirty
                    )
                )
                archive_path, commit_hash = await trio.to_thread.run_sync(
                    git_sync.create_git_archive,
                    local_root,
                )
                remote_archive = f"/tmp/bifrost-extra-{project.resolved_name}-{os.getpid()}-{int(time.time())}.tar.gz"
                try:
                    await _trio_wrap(sftp.put)(archive_path, remote_archive)
                finally:
                    await trio.to_thread.run_sync(os.unlink, archive_path)

                remote_source_root = project.remote_source_root(workspace_root)
                result = await self.exec(
                    " && ".join((
                        f"rm -rf {shlex.quote(remote_source_root)}",
                        f"mkdir -p {shlex.quote(remote_source_root)}",
                        f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_source_root)}",
                        f"rm -f {shlex.quote(remote_archive)}",
                        f"test -f {shlex.quote(remote_source_root + '/pyproject.toml')}",
                    )),
                    working_dir="~",
                )
                if result.exit_code != 0:
                    raise RuntimeError(
                        f"Failed to materialize extra Python project {project.resolved_name}: "
                        f"{result.stderr}"
                    )
                self.logger.info(
                    "materialized extra Python project %s @ %s into %s",
                    project.resolved_name,
                    commit_hash[:7],
                    remote_source_root,
                )
        finally:
            await _close_sftp_client(sftp)

    async def exec(
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
            run_kwargs: dict[str, Any] = {"check": False}
            if timeout is not None:
                run_kwargs["timeout"] = timeout
            result = await _trio_wrap(conn.run)(full_command, **run_kwargs)

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
    ) -> AsyncIterator[ProcessOutputLine]:
        """
        Execute command and stream output line-by-line in real-time.

        Like exec() but yields output as it's produced instead of waiting for completion.
        Useful for long-running commands like package installations.

        Args:
            command: Command to execute
            env: Environment variables to set (dict or EnvironmentVariables)
            working_dir: Working directory (defaults to ~/.bifrost/workspace/ if deployed)

        Yields:
            Typed stdout/stderr lines as they're produced

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

            process = await _trio_wrap(conn.create_process)(full_command)
            send_channel, receive_channel = trio.open_memory_channel[ProcessOutputLine](256)

            async def _forward_stream(
                reader: Any, stream_name: Literal["stdout", "stderr"]
            ) -> None:
                async with send_channel.clone() as stream_send:
                    while True:
                        try:
                            line = await _trio_wrap(reader.readline)()
                        except EOFError:
                            break
                        if not line:
                            break
                        await stream_send.send(
                            ProcessOutputLine(stream=stream_name, text=line.rstrip("\r\n"))
                        )

            try:
                async with trio.open_nursery() as nursery:
                    nursery.start_soon(_forward_stream, process.stdout, "stdout")
                    nursery.start_soon(_forward_stream, process.stderr, "stderr")
                    send_channel.close()
                    async with receive_channel:
                        async for line in receive_channel:
                            yield line
            finally:
                process.close()

        except Exception as e:
            if isinstance(e, SSHConnectionError):
                raise
            raise SSHConnectionError(f"Streaming execution failed: {e}") from e

    async def stream_exec(
        self,
        command: str,
        env: EnvironmentVariables | dict[str, str] | None = None,
        working_dir: str | None = None,
    ) -> AsyncIterator[ProcessOutputLine]:
        async for line in self.exec_stream(command, env=env, working_dir=working_dir):
            yield line

    async def push(
        self,
        workspace_path: str,
        bootstrap_cmd: str | list[str] | None = None,
        on_bootstrap_step: Callable[[str, int, int], None] | None = None,
        allow_dirty: bool = False,
        extra_python_projects: tuple[PythonProjectMaterialization, ...] = (),
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

        await trio.to_thread.run_sync(_check_dirty_workspace_sync, allow_dirty)
        bundle_path, commit_hash = await trio.to_thread.run_sync(_create_git_bundle_sync)
        remote_bundle = f"/tmp/bifrost-bundle-{os.getpid()}-{int(time.time())}.bundle"

        conn = await self._get_connection()
        sftp = await _trio_wrap(conn.start_sftp_client)()
        try:
            if self.progress_callback is None:
                await _trio_wrap(sftp.put)(bundle_path, remote_bundle)
            else:
                await _trio_wrap(sftp.put)(
                    bundle_path,
                    remote_bundle,
                    progress_handler=lambda _src, _dst, transferred, total: self.progress_callback(
                        remote_bundle, transferred, total
                    ),
                )
        finally:
            await _close_sftp_client(sftp)
            await trio.to_thread.run_sync(os.unlink, bundle_path)

        workspace_exists = (
            await self.exec(f"test -d {shlex.quote(workspace_path)}", working_dir="~")
        ).exit_code == 0
        if workspace_exists:
            update_cmd = (
                f"cd {shlex.quote(workspace_path)} && "
                f"git fetch {shlex.quote(remote_bundle)} HEAD && "
                "git reset --hard FETCH_HEAD && "
                f"rm {shlex.quote(remote_bundle)}"
            )
            update_result = await self.exec(update_cmd, working_dir="~")
            if update_result.exit_code != 0:
                raise RuntimeError(f"Git update from bundle failed: {update_result.stderr}")
        else:
            create_cmd = (
                f"git clone {shlex.quote(remote_bundle)} {shlex.quote(workspace_path)} && "
                f"rm {shlex.quote(remote_bundle)}"
            )
            create_result = await self.exec(create_cmd, working_dir="~")
            if create_result.exit_code != 0:
                raise RuntimeError(f"Git clone from bundle failed: {create_result.stderr}")

        verify_result = await self.exec(
            f"cd {shlex.quote(workspace_path)} && git rev-parse HEAD",
            working_dir="~",
        )
        deployed_hash = verify_result.stdout.strip()
        if verify_result.exit_code != 0 or not deployed_hash:
            raise RuntimeError(f"Failed to verify deployed workspace: {verify_result.stderr}")
        if deployed_hash != commit_hash:
            logger.warning(
                "Async SSH deploy hash mismatch: local %s remote %s",
                commit_hash[:7],
                deployed_hash[:7],
            )
        await self._materialize_extra_python_projects(
            conn=conn,
            workspace_root=workspace_path,
            extra_python_projects=extra_python_projects,
            allow_dirty=allow_dirty,
        )

        if bootstrap_cmd:
            bootstrap_steps = (
                [bootstrap_cmd] if isinstance(bootstrap_cmd, str) else list(bootstrap_cmd)
            )
            total_steps = len(bootstrap_steps)
            for index, cmd in enumerate(bootstrap_steps):
                if on_bootstrap_step is not None:
                    on_bootstrap_step(cmd, index, total_steps)
                result = await self.exec(cmd, working_dir=workspace_path)
                if result.exit_code != 0:
                    raise RuntimeError(
                        f"Bootstrap step {index + 1}/{total_steps} failed: {cmd}\n{result.stderr}"
                    )

        root = workspace_path
        self._last_workspace = root
        return root

    async def materialize_workspace(
        self,
        workspace_path: str,
        bootstrap_cmd: str | list[str] | None = None,
        on_bootstrap_step: Callable[[str, int, int], None] | None = None,
        allow_dirty: bool = False,
        extra_python_projects: tuple[PythonProjectMaterialization, ...] = (),
    ) -> WorkspaceHandle:
        root = await self.push(
            workspace_path=workspace_path,
            bootstrap_cmd=bootstrap_cmd,
            on_bootstrap_step=on_bootstrap_step,
            allow_dirty=allow_dirty,
            extra_python_projects=extra_python_projects,
        )
        return WorkspaceHandle(root=root, backend=self.backend, requested_root=workspace_path)

    async def run(self, spec: ProcessSpec, timeout: float | None = None) -> ExecResult:
        working_dir = None if spec.cwd is not None else self._last_workspace or "~"
        return await self.exec(spec.build_command(), working_dir=working_dir, timeout=timeout)

    async def start_process(
        self,
        spec: ProcessSpec,
        *,
        name: str | None = None,
        workspace: WorkspaceHandle | None = None,
        timeout: float | None = None,
        log_file: str | None = None,
    ) -> ObservedProcessHandle:
        import shlex

        process_name = name or f"process-{int(os.times().elapsed)}"
        workspace_root = workspace.root if workspace is not None else self._last_workspace
        effective_spec = spec
        if effective_spec.cwd is None and workspace_root is not None:
            effective_spec = ProcessSpec(
                command=spec.command,
                args=spec.args,
                cwd=workspace_root,
                env=spec.env,
                cuda_device_ids=spec.cuda_device_ids,
            )
        elif effective_spec.cwd and effective_spec.cwd.startswith("~"):
            effective_spec = ProcessSpec(
                command=spec.command,
                args=spec.args,
                cwd=await self.expand_path(effective_spec.cwd),
                env=spec.env,
                cuda_device_ids=spec.cuda_device_ids,
            )

        if log_file is None:
            log_file = f"~/.bifrost/logs/{process_name}.attached"
        if log_file.startswith("~"):
            log_file = await self.expand_path(log_file)
        log_dir = str(Path(log_file).parent)
        await self.exec(f"mkdir -p {shlex.quote(log_dir)}")
        stdout_log_file = f"{log_file}.stdout.log"
        stderr_log_file = f"{log_file}.stderr.log"
        await self.exec(f": > {shlex.quote(stdout_log_file)} && : > {shlex.quote(stderr_log_file)}")
        lifecycle_events = create_jsonl_event_stream(
            backend=self.backend,
            handle_kind="process",
            handle_name=process_name,
        )
        append_lifecycle_event(
            lifecycle_events,
            event="process_launch_requested",
            backend=self.backend,
            handle_name=process_name,
            command=effective_spec.command,
            cwd=effective_spec.cwd,
        )

        conn = await self._get_connection()
        full_cmd = effective_spec.build_command()
        observed_cmd = full_cmd
        process = await _trio_wrap(conn.create_process)(observed_cmd)
        append_lifecycle_event(
            lifecycle_events,
            event="process_launch_succeeded",
            backend=self.backend,
            handle_name=process_name,
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        # TODO: This per-chunk remote append is semantically honest but may be
        # too chatty for very noisy processes. If it becomes a bottleneck,
        # replace it with a buffered sink abstraction rather than reintroducing
        # a PTY recorder like `script`.

        async def _stream_output() -> AsyncIterator[ProcessOutputLine]:
            send_channel, receive_channel = trio.open_memory_channel[ProcessOutputLine](256)

            async def _forward_stream(
                reader: Any, stream_name: Literal["stdout", "stderr"]
            ) -> None:
                async with send_channel.clone() as stream_send:
                    while True:
                        try:
                            line = await _trio_wrap(reader.readline)()
                        except EOFError:
                            break
                        if not line:
                            break
                        if stream_name == "stdout":
                            stdout_chunks.append(line)
                            await self.write_text(stdout_log_file, line, append=True)
                        else:
                            stderr_chunks.append(line)
                            await self.write_text(stderr_log_file, line, append=True)
                        await stream_send.send(
                            ProcessOutputLine(stream=stream_name, text=line.rstrip("\r\n"))
                        )

            async with trio.open_nursery() as nursery:
                nursery.start_soon(_forward_stream, process.stdout, "stdout")
                nursery.start_soon(_forward_stream, process.stderr, "stderr")
                send_channel.close()
                async with receive_channel:
                    async for item in receive_channel:
                        yield item

        async def _wait() -> ExecResult:
            exit_status = await _trio_wrap(process.wait)()
            return ExecResult(
                stdout="".join(stdout_chunks),
                stderr="".join(stderr_chunks),
                exit_code=exit_status or 0,
            )

        async def _terminate() -> None:
            process.close()

        return ObservedProcessHandle(
            name=process_name,
            backend=self.backend,
            spec=effective_spec,
            workspace=effective_spec.cwd,
            output_sink=OutputSink(
                kind="file",
                location=stdout_log_file,
                description=f"stdout mirrored to {stdout_log_file}; stderr mirrored to {stderr_log_file}",
            ),
            lifecycle_events=lifecycle_events,
            state=ProcessState.RUNNING,
            _stream_output=_stream_output,
            _wait=_wait,
            _terminate=_terminate,
            metadata={
                "transport": "ssh",
                "mode": "attached",
                "stdout_log_file": stdout_log_file,
                "stderr_log_file": stderr_log_file,
            },
        )

    async def serve_service(
        self,
        service: ServiceSpec,
        *,
        name: str,
        workspace: WorkspaceHandle | None = None,
        log_file: str | None = None,
    ) -> ServiceHandle:
        workspace_root = workspace.root if workspace is not None else self._last_workspace
        assert service is not None, "ServiceSpec required"
        assert service.port is not None, "ServiceSpec.port is required for SSH service launch"
        assert name, "service name required"

        if workspace_root is None:
            workspace_root = "~"
        if workspace_root.startswith("~"):
            workspace_root = await self.expand_path(workspace_root)

        if log_file is None:
            log_file = f"~/.bifrost/logs/{name}.log"
        if log_file.startswith("~"):
            log_file = await self.expand_path(log_file)

        effective_spec = service.process
        if effective_spec.cwd is None and workspace_root is not None:
            effective_spec = ProcessSpec(
                command=effective_spec.command,
                args=effective_spec.args,
                cwd=workspace_root,
                env=effective_spec.env,
                cuda_device_ids=effective_spec.cuda_device_ids,
            )
        elif effective_spec.cwd and effective_spec.cwd.startswith("~"):
            effective_spec = ProcessSpec(
                command=effective_spec.command,
                args=effective_spec.args,
                cwd=await self.expand_path(effective_spec.cwd),
                env=effective_spec.env,
                cuda_device_ids=effective_spec.cuda_device_ids,
            )

        stdout_log_file = f"{log_file}.stdout.log"
        stderr_log_file = f"{log_file}.stderr.log"
        pid_file = f"{log_file}.pid"
        service_id = f"bifrost-service-{name}"
        lifecycle_events = create_jsonl_event_stream(
            backend=self.backend,
            handle_kind="service",
            handle_name=name,
        )
        append_lifecycle_event(
            lifecycle_events,
            event="service_launch_requested",
            backend=self.backend,
            handle_name=name,
            port=service.port,
            cwd=effective_spec.cwd,
            readiness_probe=service.readiness_probe.kind,
            readiness_target=service.readiness_probe.target,
        )
        await self.exec(
            " && ".join((
                f"mkdir -p {shlex.quote(str(Path(log_file).parent))}",
                f": > {shlex.quote(stdout_log_file)}",
                f": > {shlex.quote(stderr_log_file)}",
                f"rm -f {shlex.quote(pid_file)}",
            )),
            working_dir="~",
        )

        full_cmd = effective_spec.build_command()
        launch_cmd = build_detached_service_launch_command(
            full_cmd=full_cmd,
            stdout_log_file=stdout_log_file,
            stderr_log_file=stderr_log_file,
            pid_file=pid_file,
        )
        result = await self.exec(launch_cmd, working_dir="~")
        if result.exit_code != 0:
            raise SSHConnectionError(f"Failed to start service {name}: {result.stderr}")
        append_lifecycle_event(
            lifecycle_events,
            event="service_launch_succeeded",
            backend=self.backend,
            handle_name=name,
            service_id=service_id,
            port=service.port,
        )

        health_target = service.readiness_probe.target

        async def _is_running() -> bool:
            result = await self.exec(
                f"test -f {shlex.quote(pid_file)} && kill -0 $(cat {shlex.quote(pid_file)}) 2>/dev/null",
                working_dir="~",
            )
            return result.exit_code == 0

        async def _is_healthy() -> bool:
            if not await _is_running():
                return False
            if service.readiness_probe.kind in {"none", "process_alive"}:
                return True
            if service.readiness_probe.kind == "http":
                target = health_target or "/health"
                url = (
                    target
                    if target.startswith("http://") or target.startswith("https://")
                    else f"http://localhost:{service.port}{target}"
                )
                result = await self.exec(
                    f"curl -s -o /dev/null -w '%{{http_code}}' {url} 2>/dev/null || echo 000",
                    working_dir="~",
                )
                return result.stdout.strip() == "200"
            assert service.readiness_probe.kind == "custom", (
                f"unsupported readiness probe kind: {service.readiness_probe.kind}"
            )
            assert health_target, "custom readiness probe requires target command"
            result = await self.exec(health_target, working_dir="~")
            return result.exit_code == 0

        async def _logs(tail: int) -> str:
            result = await self.exec(
                " && ".join((
                    f"echo '== stdout ==' && tail -n {tail} {shlex.quote(stdout_log_file)} 2>/dev/null || true",
                    f"echo '== stderr ==' && tail -n {tail} {shlex.quote(stderr_log_file)} 2>/dev/null || true",
                )),
                working_dir="~",
            )
            return result.stdout

        async def _stop() -> None:
            await self.exec(
                " && ".join((
                    f"test -f {shlex.quote(pid_file)} && kill -TERM $(cat {shlex.quote(pid_file)}) 2>/dev/null || true",
                    f"rm -f {shlex.quote(pid_file)}",
                )),
                working_dir="~",
            )

        readiness_target = (
            health_target
            if health_target
            and (health_target.startswith("http://") or health_target.startswith("https://"))
            else (
                f"http://localhost:{service.port}{health_target}"
                if health_target and service.readiness_probe.kind == "http"
                else service_id
            )
        )
        return ServerInfo(
            name=name,
            service_id=service_id,
            log_file=stdout_log_file,
            port=service.port,
            health_endpoint=health_target,
            workspace=effective_spec.cwd,
            backend=self.backend,
            output_sink=OutputSink(
                kind="file",
                location=stdout_log_file,
                description=f"stdout mirrored to {stdout_log_file}; stderr mirrored to {stderr_log_file}",
            ),
            lifecycle_events=lifecycle_events,
            readiness_probe=ReadinessProbe(
                kind=service.readiness_probe.kind,
                target=readiness_target,
                timeout_s=service.readiness_probe.timeout_s,
            ),
            initial_state=ServiceState.LAUNCHING,
            _is_running=_is_running,
            _is_healthy=_is_healthy,
            _stop=_stop,
            _logs=_logs,
            metadata={
                "transport": "ssh",
                "mode": "service",
                "pid_file": pid_file,
                "stdout_log_file": stdout_log_file,
                "stderr_log_file": stderr_log_file,
            },
        )

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
                await _close_sftp_client(sftp)

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
                await _close_sftp_client(sftp)

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
            await self._ensure_asyncio_loop()
            self._ssh_conn.close()
            await _trio_wrap(self._ssh_conn.wait_closed)()
            self._ssh_conn = None
        if self._owned_asyncio_loop_cm is not None:
            await self._owned_asyncio_loop_cm.__aexit__(None, None, None)
            self._owned_asyncio_loop_cm = None

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
