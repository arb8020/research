"""Bifrost SDK data types and structures."""

import asyncio
import inspect
import json
import os
import re
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import sniffio


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
class LifecycleEvent:
    """Canonical parent-owned lifecycle event for a launched handle."""

    ts: str
    event: str
    backend: str
    handle_name: str
    data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        assert self.ts, "lifecycle event ts cannot be empty"
        assert self.event, "lifecycle event name cannot be empty"
        assert self.backend, "lifecycle event backend cannot be empty"
        assert self.handle_name, "lifecycle event handle_name cannot be empty"


def create_jsonl_event_stream(
    *, backend: str, handle_kind: str, handle_name: str
) -> EventStreamRef:
    """Create a canonical local JSONL event stream reference for a handle."""

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", handle_name).strip("-") or "handle"
    events_dir = Path(tempfile.gettempdir()) / "bifrost-events"
    events_dir.mkdir(parents=True, exist_ok=True)
    path = events_dir / f"{backend}-{handle_kind}-{safe_name}-{os.getpid()}-{time.time_ns()}.jsonl"
    path.touch()
    return EventStreamRef(
        kind="jsonl_file",
        location=str(path),
        description=f"Canonical parent-owned lifecycle events for {backend} {handle_kind}",
    )


def append_lifecycle_event(
    stream: EventStreamRef,
    *,
    event: str,
    backend: str,
    handle_name: str,
    **data: Any,
) -> None:
    """Append one lifecycle event to the canonical event sink if available."""

    if stream.kind != "jsonl_file" or not stream.location:
        return
    payload = LifecycleEvent(
        ts=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        + f".{int(time.time_ns() % 1_000_000_000):09d}Z",
        event=event,
        backend=backend,
        handle_name=handle_name,
        data=data or None,
    )
    with Path(stream.location).open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": payload.ts,
                    "event": payload.event,
                    "backend": payload.backend,
                    "handle_name": payload.handle_name,
                    "data": payload.data,
                },
                sort_keys=True,
            )
            + "\n"
        )


def read_lifecycle_events(stream: EventStreamRef) -> list[LifecycleEvent]:
    """Read all canonical lifecycle events currently available for a handle."""

    if stream.kind != "jsonl_file" or not stream.location:
        return []
    events: list[LifecycleEvent] = []
    path = Path(stream.location)
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        events.append(
            LifecycleEvent(
                ts=str(payload["ts"]),
                event=str(payload["event"]),
                backend=str(payload["backend"]),
                handle_name=str(payload["handle_name"]),
                data=payload.get("data"),
            )
        )
    return events


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
class PythonProjectMaterialization:
    """Extra Python project staged alongside the primary workspace."""

    local_root: str
    name: str | None = None
    primary_workspace_local_root: str | None = None
    source_mode: Literal["git_archive_committed"] = "git_archive_committed"
    install_mode: Literal["uv_run_with_editable"] = "uv_run_with_editable"

    def __post_init__(self) -> None:
        assert self.local_root, "local_root cannot be empty"
        assert self.source_mode == "git_archive_committed", "unsupported source_mode"
        assert self.install_mode == "uv_run_with_editable", "unsupported install_mode"
        if self.name is not None:
            assert re.match(r"^[A-Za-z0-9_.-]+$", self.name), "invalid extra project name"
        if self.primary_workspace_local_root is not None:
            assert Path(self.primary_workspace_local_root).is_absolute(), (
                "primary_workspace_local_root must be absolute"
            )

    @property
    def resolved_name(self) -> str:
        if self.name is not None:
            return self.name
        return Path(self.local_root).expanduser().resolve().name

    def remote_source_root(self, workspace_root: str) -> str:
        assert workspace_root, "workspace_root cannot be empty"
        return f"{workspace_root}/.bifrost-extra/src/{self.resolved_name}"


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
    extra_python_projects: tuple[PythonProjectMaterialization, ...] = ()

    def __post_init__(self) -> None:
        assert self.source_mode == "git_bundle_committed", "unsupported source_mode"
        seen_names: set[str] = set()
        for project in self.extra_python_projects:
            assert isinstance(project, PythonProjectMaterialization), (
                "extra_python_projects must contain PythonProjectMaterialization entries"
            )
            project_name = project.resolved_name
            assert project_name not in seen_names, f"duplicate extra project name: {project_name}"
            seen_names.add(project_name)


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
            append_lifecycle_event(
                self.lifecycle_events,
                event="process_output_stream_started",
                backend=self.backend,
                handle_name=self.name,
            )
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
        append_lifecycle_event(
            self.lifecycle_events,
            event="process_exit_observed",
            backend=self.backend,
            handle_name=self.name,
            exit_code=result.exit_code,
        )
        return result

    async def terminate(self) -> None:
        assert self._terminate is not None, "process termination is unavailable"
        self.state = ProcessState.TERMINATION_REQUESTED
        append_lifecycle_event(
            self.lifecycle_events,
            event="process_termination_requested",
            backend=self.backend,
            handle_name=self.name,
        )
        result = self._terminate()
        if inspect.isawaitable(result):
            await result

    def read_lifecycle_events(self) -> list[LifecycleEvent]:
        return read_lifecycle_events(self.lifecycle_events)


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
    """Generic long-lived service handle with explicit readiness semantics."""

    name: str
    service_id: str | None = None
    tmux_session: str | None = None
    log_file: str | None = None
    port: int | None = None
    health_endpoint: str | None = None
    workspace: str | None = None
    backend: str = "ssh"
    output_sink: OutputSink = OutputSink()
    lifecycle_events: EventStreamRef = EventStreamRef()
    readiness_probe: ReadinessProbe = ReadinessProbe()
    initial_state: ServiceState = ServiceState.CREATED
    _is_running: Callable[[], Awaitable[bool] | bool] | None = None
    _is_healthy: Callable[[], Awaitable[bool] | bool] | None = None
    _stop: Callable[[], Awaitable[None] | None] | None = None
    _logs: Callable[[int], Awaitable[str] | str] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        assert self.name, "name cannot be empty"
        assert self.backend, "backend cannot be empty"
        assert self.service_id or self.tmux_session or self.port is not None, (
            "service handle must expose a stable id, tmux session, or port"
        )

    @property
    def url(self) -> str | None:
        """Get server URL if port is known."""
        if self.port:
            return f"http://localhost:{self.port}"
        return None

    @property
    def handle_id(self) -> str | None:
        """Stable handle identifier for logs and control-plane diagnostics."""
        return self.service_id or self.tmux_session or self.url

    async def is_running(self) -> bool:
        if self._is_running is None:
            return False
        result = self._is_running()
        if inspect.isawaitable(result):
            result = await result
        append_lifecycle_event(
            self.lifecycle_events,
            event="service_running_check",
            backend=self.backend,
            handle_name=self.name,
            running=bool(result),
        )
        return bool(result)

    async def is_healthy(self) -> bool:
        if self._is_healthy is not None:
            result = self._is_healthy()
            if inspect.isawaitable(result):
                result = await result
            append_lifecycle_event(
                self.lifecycle_events,
                event="service_health_check",
                backend=self.backend,
                handle_name=self.name,
                healthy=bool(result),
                readiness_probe=self.readiness_probe.kind,
                readiness_target=self.readiness_probe.target,
            )
            return bool(result)
        if self.readiness_probe.kind == "none":
            return await self.is_running()
        return False

    async def wait_until_healthy(
        self,
        timeout: float = 300,
        poll_interval: float = 5.0,
    ) -> bool:
        assert timeout > 0, "timeout must be positive"
        assert poll_interval > 0, "poll_interval must be positive"
        started = time.monotonic()
        append_lifecycle_event(
            self.lifecycle_events,
            event="service_wait_until_healthy_started",
            backend=self.backend,
            handle_name=self.name,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        while time.monotonic() - started < timeout:
            if await self.is_healthy():
                append_lifecycle_event(
                    self.lifecycle_events,
                    event="service_ready",
                    backend=self.backend,
                    handle_name=self.name,
                    elapsed_sec=round(time.monotonic() - started, 3),
                )
                return True
            if not await self.is_running():
                append_lifecycle_event(
                    self.lifecycle_events,
                    event="service_exited_before_ready",
                    backend=self.backend,
                    handle_name=self.name,
                    elapsed_sec=round(time.monotonic() - started, 3),
                )
                return False
            await _sleep_current_async_library(poll_interval)
        append_lifecycle_event(
            self.lifecycle_events,
            event="service_readiness_timeout",
            backend=self.backend,
            handle_name=self.name,
            timeout=timeout,
        )
        return False

    async def logs(self, tail: int = 100) -> str:
        assert tail > 0, "tail must be positive"
        if self._logs is None:
            return ""
        result = self._logs(tail)
        if inspect.isawaitable(result):
            result = await result
        return str(result)

    async def stop(self) -> None:
        if self._stop is None:
            return
        append_lifecycle_event(
            self.lifecycle_events,
            event="service_stop_requested",
            backend=self.backend,
            handle_name=self.name,
        )
        result = self._stop()
        if inspect.isawaitable(result):
            await result
        append_lifecycle_event(
            self.lifecycle_events,
            event="service_stop_completed",
            backend=self.backend,
            handle_name=self.name,
        )

    def read_lifecycle_events(self) -> list[LifecycleEvent]:
        return read_lifecycle_events(self.lifecycle_events)


# Public handle aliases. Keep the old names for compatibility while making the
# intended execution semantics explicit.
ProcessHandle = JobInfo | ObservedProcessHandle
ServiceHandle = ServerInfo


async def _sleep_current_async_library(delay: float) -> None:
    """Sleep on the current async backend without hardcoding asyncio semantics."""

    # TODO(async-bridge): This backend switch is a stopgap. The cleaner model is
    # one explicit trio/asyncio bridge at the boundary rather than scattered
    # runtime dispatch inside handle methods.
    library = sniffio.current_async_library()
    if library == "asyncio":
        await asyncio.sleep(delay)
        return
    if library == "trio":
        import trio

        await trio.sleep(delay)
        return
    raise RuntimeError(f"Unsupported async library for service wait: {library}")
