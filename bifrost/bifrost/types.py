"""Bifrost SDK data types and structures."""

import inspect
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional


@dataclass
class SSHConnection:
    """SSH connection information."""

    user: str
    host: str
    port: int

    @classmethod
    def from_string(cls, ssh_string: str) -> "SSHConnection":
        """Parse SSH string in multiple formats.

        Supports:
        - 'user@host:port' format
        - 'ssh -p port user@host' format (standard SSH command)
        """
        # Handle standard SSH command format: "ssh -p port user@host"
        if ssh_string.startswith("ssh "):
            return cls._parse_ssh_command(ssh_string)

        # Handle user@host:port format
        if "@" not in ssh_string or ":" not in ssh_string:
            raise ValueError(
                f"Invalid SSH format: {ssh_string}. Expected: user@host:port or ssh -p port user@host"
            )

        user_host, port_str = ssh_string.rsplit(":", 1)
        user, host = user_host.split("@", 1)

        try:
            port = int(port_str)
        except ValueError as e:
            raise ValueError(f"Invalid port: {port_str}") from e

        return cls(user=user, host=host, port=port)

    @classmethod
    def _parse_ssh_command(cls, ssh_cmd: str) -> "SSHConnection":
        """Parse SSH command format: 'ssh -p port user@host'"""
        import shlex

        try:
            parts = shlex.split(ssh_cmd)
        except ValueError as e:
            raise ValueError(f"Failed to parse SSH command: {e}") from e

        if not parts or parts[0] != "ssh":
            raise ValueError(f"Invalid SSH command: {ssh_cmd}")

        port = 22  # default SSH port
        user_host = None

        # Parse SSH command arguments
        i = 1
        while i < len(parts):
            if parts[i] == "-p" and i + 1 < len(parts):
                try:
                    port = int(parts[i + 1])
                    i += 2
                except ValueError as e:
                    raise ValueError(f"Invalid port in SSH command: {parts[i + 1]}") from e
            elif "@" in parts[i]:
                user_host = parts[i]
                break
            else:
                i += 1

        if not user_host or "@" not in user_host:
            raise ValueError(f"No user@host found in SSH command: {ssh_cmd}")

        user, host = user_host.split("@", 1)

        return cls(user=user, host=host, port=port)

    def __str__(self) -> str:
        return f"{self.user}@{self.host}:{self.port}"


@dataclass
class RemoteConfig:
    """Configuration for connecting to remote GPU instance."""

    host: str
    port: int
    user: str
    key_path: str

    def __post_init__(self) -> None:
        # Tiger Style assertions
        assert isinstance(self.host, str) and len(self.host) > 0, "host must be non-empty string"
        assert isinstance(self.port, int) and 0 < self.port < 65536, (
            f"port must be between 1-65535, got {self.port}"
        )
        assert isinstance(self.user, str) and len(self.user) > 0, "user must be non-empty string"
        assert isinstance(self.key_path, str) and len(self.key_path) > 0, (
            "key_path must be non-empty string"
        )


@dataclass
class ExecResult:
    """Result from executing a command via SSH."""

    stdout: str
    stderr: str
    exit_code: int

    def __post_init__(self) -> None:
        # Tiger Style assertions
        assert isinstance(self.stdout, str), "stdout must be string"
        assert isinstance(self.stderr, str), "stderr must be string"
        assert isinstance(self.exit_code, int), "exit_code must be int"

    @property
    def success(self) -> bool:
        """Returns True if command exited with code 0."""
        return self.exit_code == 0


@dataclass
class CopyResult:
    """Result of a file copy operation."""

    success: bool
    files_copied: int
    total_bytes: int
    duration_seconds: float
    error_message: str | None = None

    @property
    def throughput_mbps(self) -> float:
        """Calculate transfer throughput in MB/s."""
        if self.duration_seconds > 0:
            return (self.total_bytes / (1024 * 1024)) / self.duration_seconds
        return 0.0


@dataclass
class RemotePath:
    """Represents a remote file path."""

    path: str

    def __post_init__(self) -> None:
        # Ensure path is absolute for consistency
        if not self.path.startswith("/") and not self.path.startswith("~"):
            self.path = f"./{self.path}"


class BifrostError(Exception):
    """Base exception for Bifrost SDK errors."""

    pass


class SSHConnectionError(BifrostError):
    """SSH connection related errors."""

    pass


# Backwards compatibility alias (deprecated - use SSHConnectionError)
ConnectionError = SSHConnectionError  # noqa: A001


class JobError(BifrostError):
    """Job execution related errors."""

    pass


class TransferError(BifrostError):
    """File transfer related errors."""

    pass


# ============================================================================
# New Frozen Dataclasses (Type Improvements)
# ============================================================================


@dataclass(frozen=True)
class EnvironmentVariables:
    """Environment variables for remote command execution.

    Validates variable names follow shell naming rules.
    Immutable to prevent accidental modification during execution.
    """

    variables: dict[str, str]

    def __post_init__(self) -> None:
        # Tiger Style: assert all inputs
        assert isinstance(self.variables, dict), "variables must be dict"

        for key, value in self.variables.items():
            # Shell variable name rules: [A-Za-z_][A-Za-z0-9_]*
            assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key), (
                f"Invalid environment variable name: {key}"
            )
            assert isinstance(value, str), (
                f"Environment variable {key} must be string, got {type(value)}"
            )

        # Assert output invariant
        assert len(self.variables) >= 0, "variables dict created"

    def to_dict(self) -> dict[str, str]:
        """Convert to dict for backward compatibility."""
        return self.variables.copy()

    @classmethod
    def from_dict(cls, variables: dict[str, str] | None) -> Optional["EnvironmentVariables"]:
        """Create from optional dict (for gradual migration)."""
        if variables is None:
            return None
        return cls(variables=variables)


# ============================================================================
# v2 API Types (functions-over-classes pattern)
# ============================================================================


@dataclass(frozen=True)
class ProcessSpec:
    """Complete specification of a process to run.

    Immutable - represents what to run, not a running process.
    Can be serialized, logged, compared.

    Env vars are passed here, not cached on Session.

    Example:
        spec = ProcessSpec(
            command="python",
            args=("train.py", "--lr", "0.001"),
            cwd="/workspace",
            env={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
    """

    command: str
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env: dict[str, str] | None = None
    cuda_device_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        assert self.command, "command cannot be empty"
        assert isinstance(self.args, tuple), "args must be a tuple"

    def build_command(self) -> str:
        """Build full command string with proper escaping.

        Returns command string suitable for shell execution.
        """
        import shlex

        # Build base command with args
        cmd_parts = [self.command] + list(self.args)
        full_cmd = shlex.join(cmd_parts)

        # Add env vars prefix if needed
        if self.env:
            env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
            full_cmd = f"{env_prefix} {full_cmd}"

        # Add CUDA_VISIBLE_DEVICES if cuda_device_ids specified
        if self.cuda_device_ids is not None:
            devices_str = ",".join(str(d) for d in self.cuda_device_ids)
            full_cmd = f"CUDA_VISIBLE_DEVICES={devices_str} {full_cmd}"

        # Add cd if cwd specified
        if self.cwd:
            full_cmd = f"cd {shlex.quote(self.cwd)} && {full_cmd}"

        return full_cmd


class ProcessState(str, Enum):
    """Parent-observed lifecycle state for a launched process."""

    CREATED = "created"
    LAUNCHING = "launching"
    RUNNING = "running"
    EXITED = "exited"
    LAUNCH_FAILED = "launch_failed"
    TERMINATION_REQUESTED = "termination_requested"


class ServiceState(str, Enum):
    """Parent-observed lifecycle state for a launched service."""

    CREATED = "created"
    LAUNCHING = "launching"
    STARTING = "starting"
    READY = "ready"
    READINESS_FAILED = "readiness_failed"
    STOPPING = "stopping"
    EXITED = "exited"


@dataclass(frozen=True)
class OutputSink:
    """Reference to canonical raw process output owned by the parent launcher."""

    kind: Literal["file", "provider_stream", "unknown"] = "unknown"
    location: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        assert self.kind in {"file", "provider_stream", "unknown"}, "invalid output sink kind"


@dataclass(frozen=True)
class EventStreamRef:
    """Reference to canonical lifecycle events for a launched handle."""

    kind: Literal["jsonl_file", "provider_stream", "unknown"] = "unknown"
    location: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        assert self.kind in {"jsonl_file", "provider_stream", "unknown"}, (
            "invalid event stream kind"
        )


@dataclass(frozen=True)
class ReadinessProbe:
    """How a parent launcher determines that a service is ready."""

    kind: Literal["http", "process_alive", "custom", "none"] = "none"
    target: str | None = None
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        assert self.kind in {"http", "process_alive", "custom", "none"}, (
            "invalid readiness probe kind"
        )
        if self.timeout_s is not None:
            assert self.timeout_s > 0, "timeout_s must be positive"


@dataclass(frozen=True)
class WorkspaceMaterializationSpec:
    """Materialization request for a project snapshot on an execution session.

    `requested_root` is a hint, not a guarantee. Some backends can honor an
    exact workspace root; others may materialize into a backend-owned fixed
    root and report the realized path in the returned WorkspaceHandle.
    """

    requested_root: str | None = None
    bootstrap_commands: tuple[str, ...] = ()
    source_mode: Literal["git_bundle_committed"] = "git_bundle_committed"
    allow_dirty: bool = False

    def __post_init__(self) -> None:
        assert self.source_mode == "git_bundle_committed", "unsupported source_mode"


@dataclass(frozen=True)
class ServiceSpec:
    """Long-lived process denotation with readiness semantics."""

    process: ProcessSpec
    port: int | None = None
    readiness_probe: ReadinessProbe = ReadinessProbe()
    shutdown_signal: str | None = None

    def __post_init__(self) -> None:
        assert self.process is not None, "service process cannot be None"
        if self.port is not None:
            assert self.port > 0, "service port must be positive"


@dataclass(frozen=True)
class WorkspaceHandle:
    """Materialized workspace on a live execution session."""

    root: str
    backend: str = "ssh"
    source_ref: str | None = None
    materialization: str | None = None
    requested_root: str | None = None

    def __post_init__(self) -> None:
        assert self.root, "workspace root cannot be empty"
        assert self.backend, "workspace backend cannot be empty"


@dataclass(frozen=True)
class ChildEvent:
    """Optional semantic milestone emitted by the launched child process."""

    name: str
    detail: dict[str, str] | None = None

    def __post_init__(self) -> None:
        assert self.name, "child event name cannot be empty"


@dataclass(frozen=True)
class ProcessOutputLine:
    """One line of live process output with an explicit source stream."""

    stream: Literal["stdout", "stderr"]
    text: str

    def __post_init__(self) -> None:
        assert self.stream in ("stdout", "stderr"), "stream must be stdout or stderr"
        assert isinstance(self.text, str), "text must be a string"


@dataclass
class ObservedProcessHandle:
    """Live attached process handle with parent-owned output observation."""

    name: str
    backend: str
    spec: ProcessSpec
    workspace: str | None = None
    output_sink: OutputSink = OutputSink(kind="provider_stream")
    lifecycle_events: EventStreamRef = EventStreamRef(kind="provider_stream")
    state: ProcessState = ProcessState.LAUNCHING
    _stream_output: (
        Callable[[], AsyncIterator[ProcessOutputLine] | Iterator[ProcessOutputLine]] | None
    ) = None
    _wait: Callable[[], Awaitable[ExecResult] | ExecResult] | None = None
    _terminate: Callable[[], Awaitable[None] | None] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        assert self.name, "process handle name cannot be empty"
        assert self.backend, "process handle backend cannot be empty"

    async def stream_output(self) -> AsyncIterator[ProcessOutputLine]:
        assert self._stream_output is not None, "process output stream is unavailable"
        if self.state is ProcessState.LAUNCHING:
            self.state = ProcessState.RUNNING
        stream = self._stream_output()
        if hasattr(stream, "__aiter__"):
            async for line in stream:
                yield line
            return
        for line in stream:
            yield line

    async def wait(self) -> ExecResult:
        assert self._wait is not None, "process wait is unavailable"
        result = self._wait()
        if inspect.isawaitable(result):
            result = await result
        self.state = ProcessState.EXITED
        return result

    async def terminate(self) -> None:
        assert self._terminate is not None, "process termination is unavailable"
        self.state = ProcessState.TERMINATION_REQUESTED
        result = self._terminate()
        if inspect.isawaitable(result):
            await result


@dataclass(frozen=True)
class JobInfo:
    """Immutable detached-process handle.

    This is the new v2 JobInfo that follows the functions-over-classes pattern.
    It contains stable handle identity and output/event sink references, not
    live status. Status comes from job_status() and related functions.

    Returned by BifrostClient.submit().
    Used with job_status(), job_wait(), job_logs(), job_kill() functions.
    """

    name: str
    tmux_session: str
    log_file: str | None = None
    workspace: str | None = None
    backend: str = "ssh"
    output_sink: OutputSink = OutputSink()
    lifecycle_events: EventStreamRef = EventStreamRef()
    initial_state: ProcessState = ProcessState.CREATED

    def __post_init__(self) -> None:
        assert self.name, "name cannot be empty"
        assert self.tmux_session, "tmux_session cannot be empty"
        assert self.backend, "backend cannot be empty"


@dataclass(frozen=True)
class ServerInfo:
    """Immutable long-lived service handle.

    This is the new v2 ServerInfo that follows the functions-over-classes pattern.
    It contains stable handle identity plus readiness metadata, not live status.
    Health and lifecycle state come from server_is_healthy() and related
    functions.

    Returned by BifrostClient.serve().
    Used with server_is_healthy(), server_wait_until_healthy(), server_stop() functions.
    """

    name: str
    tmux_session: str
    log_file: str | None = None
    port: int | None = None
    health_endpoint: str | None = None
    workspace: str | None = None
    backend: str = "ssh"
    output_sink: OutputSink = OutputSink()
    lifecycle_events: EventStreamRef = EventStreamRef()
    readiness_probe: ReadinessProbe = ReadinessProbe()
    initial_state: ServiceState = ServiceState.CREATED

    def __post_init__(self) -> None:
        assert self.name, "name cannot be empty"
        assert self.tmux_session, "tmux_session cannot be empty"
        assert self.backend, "backend cannot be empty"

    @property
    def url(self) -> str | None:
        """Get server URL if port is known."""
        if self.port:
            return f"http://localhost:{self.port}"
        return None


# Public handle aliases. Keep the old names for compatibility while making the
# intended execution semantics explicit.
ProcessHandle = JobInfo | ObservedProcessHandle
ServiceHandle = ServerInfo
