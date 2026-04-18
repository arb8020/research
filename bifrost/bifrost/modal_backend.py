"""Transitional Modal execution backend for bifrost.

This module is the provider boundary for Modal-backed execution. Provider-owned
mechanics such as sandbox command execution and workspace materialization live
here, while Rollouts still owns workload-specific helper code.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import trio

from .types import (
    ExecResult,
    ObservedProcessHandle,
    OutputSink,
    ProcessOutputLine,
    ProcessSpec,
    ProcessState,
    ServerInfo,
    ServiceHandle,
    ServiceSpec,
    ServiceState,
    WorkspaceHandle,
    WorkspaceMaterializationSpec,
    append_lifecycle_event,
    create_jsonl_event_stream,
)

logger = logging.getLogger(__name__)

MODAL_PARENT_LEASE_PATH = "/tmp/bifrost/modal_parent_lease"
MODAL_PARENT_LEASE_TTL_S = 90.0
MODAL_PARENT_LEASE_REFRESH_INTERVAL_S = 15.0


def _is_modal_not_found_error(exc: BaseException) -> bool:
    """Return true when Modal reports a sandbox/object is already gone.

    Teardown/artifact polling should treat this as a terminal condition, not a
    new failure that masks the child process outcome.
    """

    return type(exc).__name__ == "NotFoundError"


def _json_decode_maybe_incomplete(payload: str, exc: json.JSONDecodeError) -> bool:
    """Return true when a JSON decode failure likely reflects a split chunk.

    Modal's stream callbacks can hand us very large stderr payloads in pieces.
    Child-side event JSON should stay on the event channel, not degrade into a
    parse failure plus raw stderr tail just because the SDK split the line.
    """

    stripped = payload.rstrip()
    if not stripped:
        return True
    if exc.msg.startswith("Unterminated string"):
        return True
    if exc.pos >= max(len(payload) - 1, 0):
        return True
    return stripped[-1] not in {"}", "]"}


def _consume_modal_diag_stream_chunk(
    *,
    existing_buffer: str,
    chunk: str,
    sentinel: str,
) -> tuple[list[str], list[tuple[str, dict[str, Any]]], str, str | None]:
    """Split mixed stderr into plain text and child events.

    This keeps partially delivered sentinel JSON buffered until the next chunk
    arrives, instead of spuriously emitting parse failures and journaling the
    tail as generic stderr payload.
    """

    decoder = json.JSONDecoder()
    buffer = f"{existing_buffer}{chunk}"
    plain_fragments: list[str] = []
    parsed_events: list[tuple[str, dict[str, Any]]] = []

    while buffer:
        sentinel_idx = buffer.find(sentinel)
        if sentinel_idx == -1:
            plain_fragments.append(buffer)
            return plain_fragments, parsed_events, "", None

        prefix = buffer[:sentinel_idx]
        if prefix:
            plain_fragments.append(prefix)

        payload = buffer[sentinel_idx + len(sentinel) :].lstrip()
        try:
            event_data, end_idx = decoder.raw_decode(payload)
        except json.JSONDecodeError as exc:
            if _json_decode_maybe_incomplete(payload, exc):
                return plain_fragments, parsed_events, f"{sentinel}{payload}", None
            return plain_fragments, parsed_events, "", f"{type(exc).__name__}: {exc}"

        event_name = event_data.pop("event", None)
        if isinstance(event_name, str) and event_name:
            parsed_events.append((event_name, event_data))
        buffer = payload[end_idx:].lstrip()

    return plain_fragments, parsed_events, "", None


def _normalize_run_logger_event_payload(
    *,
    run_name: str | None = None,
    provider: str | None = None,
    backend: str | None = None,
    handle_name: str | None = None,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Strip reserved parent keys while preserving differing child identity."""

    payload = dict(data)
    child_run_name = payload.pop("run_name", None)
    child_provider = payload.pop("provider", None)
    child_backend = payload.pop("backend", None)
    child_handle_name = payload.pop("handle_name", None)
    if child_run_name is not None and child_run_name != run_name:
        payload["child_run_name"] = child_run_name
    elif child_run_name is not None and run_name is None:
        payload["run_name"] = child_run_name
    if child_provider is not None and child_provider != provider:
        payload["child_provider"] = child_provider
    elif child_provider is not None and provider is None:
        payload["provider"] = child_provider
    if child_backend is not None and child_backend != backend:
        payload["child_backend"] = child_backend
    elif child_backend is not None and backend is None:
        payload["backend"] = child_backend
    if child_handle_name is not None and child_handle_name != handle_name:
        payload["child_handle_name"] = child_handle_name
    elif child_handle_name is not None and handle_name is None:
        payload["handle_name"] = child_handle_name
    return payload


def _exception_is_operator_interrupt(exc: BaseException) -> bool:
    """Return true when shutdown was initiated by the local operator/runtime."""

    if isinstance(exc, (KeyboardInterrupt, trio.Cancelled)):
        return True
    base_exception_group = globals().get("BaseExceptionGroup")
    if base_exception_group is not None and isinstance(exc, base_exception_group):
        return any(_exception_is_operator_interrupt(child) for child in exc.exceptions)
    return False


def _modal_supervisor_exit_code(event_name: str, event_data: dict[str, Any]) -> int | None:
    """Normalize supervisor child-exit events into the parent-visible exit code.

    The Modal workload supervisor emits `remote_supervisor_child_exit` when the
    actual training child finishes. That is the real completion boundary for the
    workload. Detached inference services may still keep inherited stdout/stderr
    pipes open afterward, so waiting for stream EOF is dishonest here.
    """

    if event_name != "remote_supervisor_child_exit":
        return None
    raw = event_data.get("child_returncode")
    if not isinstance(raw, int):
        return 1
    if raw < 0:
        return 128 + (-raw)
    return raw


def _read_modal_supervisor_status_sync(sandbox: Any, status_file: str) -> dict[str, Any] | None:
    """Read the supervisor status file from the sandbox, if present."""

    try:
        proc = sandbox.exec(
            "bash",
            "-lc",
            f"if [ -f {shlex.quote(status_file)} ]; then cat {shlex.quote(status_file)}; fi",
            timeout=30,
        )
    except BaseException as exc:
        if _is_modal_not_found_error(exc):
            return None
        raise
    stdout = "".join(proc.stdout)
    _ = "".join(proc.stderr)
    exit_code = proc.wait()
    if exit_code != 0 or not stdout.strip():
        return None
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _modal_primary_command(keep_alive: bool) -> tuple[str, ...]:
    if keep_alive:
        return (
            "python3",
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, lambda *_: exit(0)); "
            "signal.signal(signal.SIGINT, lambda *_: exit(0)); time.sleep(315360000)",
        )

    watchdog_script = (
        "import pathlib\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "\n"
        "lease_path = pathlib.Path(sys.argv[1])\n"
        "ttl_s = float(sys.argv[2])\n"
        "signal.signal(signal.SIGTERM, lambda *_: exit(0))\n"
        "signal.signal(signal.SIGINT, lambda *_: exit(0))\n"
        "startup_deadline = time.time() + ttl_s\n"
        "sleep_s = max(1.0, min(ttl_s / 6.0, 5.0))\n"
        "while True:\n"
        "    now = time.time()\n"
        "    if lease_path.exists():\n"
        "        if now - lease_path.stat().st_mtime > ttl_s:\n"
        "            break\n"
        "    elif now >= startup_deadline:\n"
        "        break\n"
        "    time.sleep(sleep_s)\n"
    )
    return (
        "python3",
        "-c",
        watchdog_script,
        MODAL_PARENT_LEASE_PATH,
        str(MODAL_PARENT_LEASE_TTL_S),
    )


def _refresh_modal_parent_lease_sync(sandbox: Any) -> None:
    lease_dir = shlex.quote(str(Path(MODAL_PARENT_LEASE_PATH).parent))
    lease_path = shlex.quote(MODAL_PARENT_LEASE_PATH)
    proc = sandbox.exec(
        "bash",
        "-lc",
        f"mkdir -p {lease_dir} && touch {lease_path}",
        timeout=30,
    )
    _ = "".join(proc.stdout)
    stderr = "".join(proc.stderr)
    exit_code = proc.wait()
    if exit_code != 0:
        raise RuntimeError(
            f"Failed to refresh Modal parent lease at {MODAL_PARENT_LEASE_PATH}: "
            f"exit_code={exit_code} stderr={stderr.strip()}"
        )


async def _refresh_modal_parent_lease(sandbox: Any) -> None:
    await trio.to_thread.run_sync(_refresh_modal_parent_lease_sync, sandbox)


async def _maintain_modal_parent_lease(
    sandbox: Any,
    emit: Callable[[str], None] | None = None,
    refresh_interval_s: float = MODAL_PARENT_LEASE_REFRESH_INTERVAL_S,
) -> None:
    def _emit(event: str, **data: Any) -> None:
        if emit is not None:
            emit(event, **data)

    _emit(
        "modal_parent_lease_heartbeat_started",
        lease_path=MODAL_PARENT_LEASE_PATH,
        ttl_s=MODAL_PARENT_LEASE_TTL_S,
        refresh_interval_s=refresh_interval_s,
    )
    while True:
        await _refresh_modal_parent_lease(sandbox)
        _emit("modal_parent_lease_refreshed", lease_path=MODAL_PARENT_LEASE_PATH)
        await trio.sleep(refresh_interval_s)


def _read_modal_text_artifact_sync(sandbox: Any, remote_path: str) -> str | None:
    """Read a text artifact from the sandbox, if present."""

    try:
        proc = sandbox.exec(
            "bash",
            "-lc",
            (
                f"if [ -f {shlex.quote(remote_path)} ]; then "
                f"cat {shlex.quote(remote_path)}; "
                "else exit 2; fi"
            ),
            timeout=30,
        )
    except BaseException as exc:
        if _is_modal_not_found_error(exc):
            return None
        raise
    stdout = "".join(proc.stdout)
    _ = "".join(proc.stderr)
    exit_code = proc.wait()
    if exit_code != 0:
        return None
    return stdout


async def _copy_modal_text_artifact(
    sandbox: Any,
    *,
    remote_path: str,
    local_path: Path,
    emit: Callable[[str], None] | None = None,
    event_prefix: str = "modal_artifact_copy",
) -> bool:
    """Copy a remote text artifact into the local run directory if it exists."""

    def _emit(event: str, **data: Any) -> None:
        if emit is not None:
            emit(event, **data)

    _emit(f"{event_prefix}_start", remote_path=remote_path, local_path=str(local_path))
    contents = await trio.to_thread.run_sync(
        lambda: _read_modal_text_artifact_sync(sandbox, remote_path)
    )
    if contents is None:
        _emit(f"{event_prefix}_missing", remote_path=remote_path, local_path=str(local_path))
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(contents)
    _emit(
        f"{event_prefix}_finished",
        remote_path=remote_path,
        local_path=str(local_path),
        bytes=len(contents.encode()),
    )
    return True


def _emit_projected_training_artifact_event(
    emit: Callable[..., None],
    data: dict[str, Any],
    *,
    projected_by: str,
) -> None:
    event_name = data.get("event")
    if event_name == "step_complete":
        emit(
            "step_complete",
            projected_from_artifact=True,
            projected_by=projected_by,
            projection_source="training.jsonl",
            step=data.get("step"),
            mean_reward=data.get("mean_reward"),
            pg_loss=data.get("pg_loss"),
            entropy=data.get("entropy"),
            num_samples=data.get("num_samples"),
            num_groups=data.get("num_groups"),
            step_total_ms=data.get("step_total_ms"),
            rollout_step_count=data.get("rollout_step_count"),
            gpu_allocated_gb=data.get("gpu_allocated_gb"),
            gpu_reserved_gb=data.get("gpu_reserved_gb"),
            ram_gb=data.get("ram_gb"),
        )
        return
    if event_name != "train_step_complete":
        return
    emit(
        "train_step_complete",
        projected_from_artifact=True,
        projected_by=projected_by,
        projection_source="training.jsonl",
        step=data.get("step"),
        mean_reward=data.get("mean_reward"),
        pg_loss=data.get("pg_loss"),
        entropy=data.get("entropy"),
        loss=data.get("loss"),
        grad_norm=data.get("grad_norm"),
        process_batch_ms=data.get("process_batch_ms"),
        checkpoint_ms=data.get("checkpoint_ms"),
        weight_sync_ms=data.get("weight_sync_ms"),
        step_total_ms=data.get("step_total_ms"),
        rollout_step_count=data.get("rollout_step_count"),
    )


def _emit_projected_metrics_artifact_event(
    emit: Callable[..., None],
    data: dict[str, Any],
    *,
    projected_by: str,
) -> None:
    emit(
        "metrics_update",
        projected_from_artifact=True,
        projected_by=projected_by,
        projection_source="metrics.jsonl",
        step=data.get("step"),
        mean_reward=data.get("mean_reward"),
        loss=data.get("loss"),
        grad_norm=data.get("grad_norm"),
        pg_loss=data.get("pg_loss"),
        entropy=data.get("entropy"),
        rollout_step_count=data.get("rollout_step_count"),
        rollout_samples_generated=data.get("rollout_samples_generated"),
        timestamp=data.get("timestamp"),
    )


def _project_modal_artifact_snapshot(
    *,
    file_name: str,
    contents: str,
    state: dict[str, Any],
    emit: Callable[..., None],
    projected_by: str,
) -> None:
    if not state.get("ready_announced"):
        state["ready_announced"] = True
        emit(
            "remote_artifact_file_ready",
            file=file_name,
            projected_by=projected_by,
        )

    previous_offset = int(state.get("offset", 0))
    if len(contents) < previous_offset:
        state["offset"] = 0
        state["buffer"] = ""
        previous_offset = 0
        emit(
            "remote_artifact_reset_detected",
            file=file_name,
            projected_by=projected_by,
        )

    chunk = contents[previous_offset:]
    state["offset"] = len(contents)
    if not chunk:
        return

    buffer = f"{state.get('buffer', '')}{chunk}"
    lines = buffer.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        state["buffer"] = lines.pop()
    else:
        state["buffer"] = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception as exc:
            emit(
                "remote_artifact_parse_failed",
                file=file_name,
                projected_by=projected_by,
                error=f"{type(exc).__name__}: {exc}",
                line_preview=line[:400],
            )
            continue
        if not isinstance(parsed, dict):
            continue
        if file_name == "training.jsonl":
            _emit_projected_training_artifact_event(
                emit,
                parsed,
                projected_by=projected_by,
            )
        elif file_name == "metrics.jsonl":
            _emit_projected_metrics_artifact_event(
                emit,
                parsed,
                projected_by=projected_by,
            )


async def _project_modal_run_artifacts(
    sandbox: Any,
    run_name: str,
    journal_event: Callable[..., None],
    stop_event: trio.Event,
    remote_run_dir: str,
    local_run_dir: Path | None,
    interval_s: float = 2.0,
) -> None:
    """Project remote JSONL artifact milestones into the local run journal."""

    watched_files = ("training.jsonl", "metrics.jsonl")
    state: dict[str, dict[str, Any]] = {
        name: {"offset": 0, "buffer": "", "ready_announced": False} for name in watched_files
    }
    projected_by = "modal_parent"
    journal_event(
        "remote_artifact_projection_started",
        run_name=run_name,
        projected_by=projected_by,
        files=list(watched_files),
        interval_s=interval_s,
        remote_run_dir=remote_run_dir,
    )

    async def _poll_once() -> None:
        for file_name in watched_files:
            remote_path = f"{remote_run_dir}/{file_name}"
            contents = await trio.to_thread.run_sync(
                lambda rp=remote_path: _read_modal_text_artifact_sync(sandbox, rp)
            )
            if contents is None:
                continue
            if local_run_dir is not None:
                (local_run_dir / file_name).write_text(contents)
            _project_modal_artifact_snapshot(
                file_name=file_name,
                contents=contents,
                state=state[file_name],
                emit=journal_event,
                projected_by=projected_by,
            )

    while not stop_event.is_set():
        await _poll_once()
        with trio.move_on_after(interval_s):
            await stop_event.wait()
    await _poll_once()


async def _wait_observed_process_nonblocking(process: ObservedProcessHandle) -> ExecResult:
    """Wait for a provider process without blocking sibling Trio tasks."""

    assert process._wait is not None, "process wait is unavailable"
    wait_fn = process._wait
    result = await trio.to_thread.run_sync(wait_fn)
    process.state = ProcessState.EXITED
    append_lifecycle_event(
        process.lifecycle_events,
        event="process_exit_observed",
        backend=process.backend,
        handle_name=process.name,
        exit_code=result.exit_code,
    )
    return result


def _append_run_journal_event(
    log_path: Path,
    *,
    run_name: str,
    provider: str,
    event: str,
    data: dict[str, Any],
) -> None:
    payload = {
        "ts": datetime.now().isoformat(),
        "event": event,
        "provider": provider,
        "run_name": run_name,
        **data,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=False))
        f.write("\n")
        f.flush()


@dataclass(frozen=True)
class ModalExecutionRequest:
    """Provider-owned execution request for the current Modal backend."""

    config_path: str
    runtime: Any
    materialization: Any
    source_sync_policy: Any
    timeout_hours: int = 4
    sandbox_id: str | None = None
    keep_alive: bool = False
    cleanup_scope: str = "run"
    run_name: str | None = None
    extra_source_roots: tuple[str, ...] = ()
    model_name: str | None = None
    pruning_recipe: str | None = None
    run_logger: Any = None
    tags: dict[str, str] = field(default_factory=dict)
    # split_env: inference venv installed separately from the trainer venv.
    # When set, add_inference_venv_to_image layers INFERENCE_VENV_DIR onto the
    # trainer image, and ROLLOUTS_INFERENCE_PYTHON is set in the trainer env so
    # weight_sync._python_module_launch uses the right interpreter.
    # Only valid when inference_deps.base_image == runtime.deps.base_image.
    # For a different base image, use two separate sandboxes (not yet implemented).
    inference_deps: Any = None
    encrypted_ports: tuple[int, ...] = ()
    unencrypted_ports: tuple[int, ...] = ()
    h2_ports: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModalSandboxHandle:
    """Live Modal sandbox handle owned by the bifrost boundary."""

    sandbox: Any
    sandbox_id: str
    keep_alive: bool = False
    backend: str = "modal"


@dataclass(frozen=True)
class ModalExecutionSession:
    """Live Modal execution session backed by one sandbox."""

    sandbox_handle: ModalSandboxHandle
    local_root: Path
    extra_source_roots: tuple[Path, ...] = ()
    backend: str = "modal"
    current_workspace_handle: WorkspaceHandle | None = None

    async def materialize(
        self,
        spec: WorkspaceMaterializationSpec,
        *,
        emit: Callable[[str], None] | None = None,
    ) -> WorkspaceHandle:
        workspace = await materialize_modal_workspace(
            self.sandbox_handle,
            self.local_root,
            extra_source_roots=self.extra_source_roots,
            emit=emit,
        )
        return WorkspaceHandle(
            root=workspace.root,
            backend=workspace.backend,
            source_ref=workspace.source_ref,
            materialization=workspace.materialization,
            requested_root=spec.requested_root,
        )

    async def materialize_workspace(
        self,
        workspace_path: str,
        bootstrap_cmd: str | list[str] | None = None,
        on_bootstrap_step: Callable[[str, int, int], None] | None = None,
        allow_dirty: bool = False,
    ) -> WorkspaceHandle:
        del bootstrap_cmd, on_bootstrap_step, allow_dirty
        return await self.materialize(WorkspaceMaterializationSpec(requested_root=workspace_path))

    async def exec(
        self,
        command: str,
        env: dict[str, str] | None = None,
        working_dir: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        import shlex

        prefix = ""
        if working_dir is not None:
            prefix += f"cd {shlex.quote(working_dir)} && "
        if env:
            prefix += " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "
        stdout, stderr, exit_code = await exec_modal_command(
            self.sandbox_handle,
            f"{prefix}{command}",
            timeout=int(timeout or 300),
        )
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

    async def run(self, spec: ProcessSpec, timeout: float | None = None) -> ExecResult:
        return await self.exec(spec.build_command(), timeout=timeout)

    async def stream_exec(
        self,
        command: str,
        env: dict[str, str] | None = None,
        working_dir: str | None = None,
    ) -> AsyncIterator[ProcessOutputLine]:
        import shlex

        prefix = ""
        if working_dir is not None:
            prefix += f"cd {shlex.quote(working_dir)} && "
        if env:
            prefix += " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "

        send_channel, receive_channel = trio.open_memory_channel[ProcessOutputLine](256)
        trio_token = trio.lowlevel.current_trio_token()
        result_holder: dict[str, Any] = {}

        def _worker() -> None:
            try:

                def _on_stdout_line(line: str) -> None:
                    trio.from_thread.run(
                        send_channel.send,
                        ProcessOutputLine(stream="stdout", text=line.rstrip("\r\n")),
                        trio_token=trio_token,
                    )

                def _on_stderr_line(line: str) -> None:
                    trio.from_thread.run(
                        send_channel.send,
                        ProcessOutputLine(stream="stderr", text=line.rstrip("\r\n")),
                        trio_token=trio_token,
                    )

                result_holder["result"] = exec_modal_command_sync(
                    self.sandbox_handle.sandbox,
                    f"{prefix}{command}",
                    timeout=300,
                    stream_output=False,
                    on_stdout_line=_on_stdout_line,
                    on_stderr_line=_on_stderr_line,
                )
            except Exception as exc:
                result_holder["error"] = exc
            finally:
                trio.from_thread.run_sync(send_channel.close, trio_token=trio_token)

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        try:
            async with receive_channel:
                async for item in receive_channel:
                    yield item
        finally:
            await trio.to_thread.run_sync(worker.join)

        if "error" in result_holder:
            raise result_holder["error"]

    async def terminate(self) -> None:
        await terminate_modal_sandbox(self.sandbox_handle)

    async def start_process(
        self,
        spec: ProcessSpec,
        *,
        name: str | None = None,
        timeout: int = 14400,
        emit: Callable[[str], None] | None = None,
        startup_sentinel: str | None = None,
        stdout_event_sentinel: str | None = None,
        stderr_event_sentinel: str | None = None,
        start_timeout_s: float | None = None,
    ) -> ObservedProcessHandle:
        """Launch and observe a live attached process inside the sandbox."""

        process_name = name or f"modal-process-{int(time.time())}"
        results: dict[str, Any] = {}
        done = threading.Event()
        shutdown_requested = threading.Event()
        startup_seen = threading.Event()
        process_started_ts = time.monotonic()
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        stdout_tail: deque[str] = deque(maxlen=20)
        stderr_tail: deque[str] = deque(maxlen=40)
        output_send, output_receive = trio.open_memory_channel[ProcessOutputLine](256)
        trio_token = trio.lowlevel.current_trio_token()
        supervisor_completion: dict[str, Any] = {}
        supervisor_completion_seen = threading.Event()
        lifecycle_events = create_jsonl_event_stream(
            backend=self.backend,
            handle_kind="process",
            handle_name=process_name,
        )
        completion_status_file = None
        if spec.env is not None:
            completion_status_file = spec.env.get("ARGUS_SUPERVISOR_STATUS_FILE")
        last_completion_status_poll = 0.0
        process_state: dict[str, Any] = {
            "stdout_line_count": 0,
            "stderr_line_count": 0,
            "last_stdout_line": None,
            "last_stderr_line": None,
            "last_stdout_elapsed_sec": None,
            "last_stderr_elapsed_sec": None,
            "stderr_event_buffer": "",
        }

        def _elapsed() -> float:
            return time.monotonic() - process_started_ts

        def _emit(event: str, **data: Any) -> None:
            append_lifecycle_event(
                lifecycle_events,
                event=event,
                backend=self.backend,
                handle_name=process_name,
                **data,
            )
            if emit is not None:
                emit(event, **data)

        def _emit_stream_line(event: str, line_count: int, text: str) -> None:
            max_chars = 4000
            _emit(
                event,
                line_no=line_count,
                elapsed_sec=round(_elapsed(), 3),
                text=text[:max_chars],
                truncated=len(text) > max_chars,
            )

        def _maybe_record_supervisor_completion(
            event_name: str, event_data: dict[str, Any]
        ) -> None:
            exit_code = _modal_supervisor_exit_code(event_name, event_data)
            if exit_code is None:
                return
            supervisor_completion["event_name"] = event_name
            supervisor_completion["event_data"] = dict(event_data)
            supervisor_completion["exit_code"] = exit_code
            supervisor_completion_seen.set()

        def _poll_supervisor_completion() -> None:
            nonlocal last_completion_status_poll
            if completion_status_file is None or supervisor_completion_seen.is_set():
                return
            now = time.monotonic()
            if now - last_completion_status_poll < 5.0:
                return
            last_completion_status_poll = now
            payload = _read_modal_supervisor_status_sync(
                self.sandbox_handle.sandbox,
                completion_status_file,
            )
            if not payload:
                return
            event_name = payload.get("event")
            if not isinstance(event_name, str):
                return
            _maybe_record_supervisor_completion(event_name, payload)

        def _send_output_line(line: ProcessOutputLine) -> None:
            try:
                trio.from_thread.run(
                    output_send.send,
                    line,
                    trio_token=trio_token,
                )
            except BaseException as exc:
                if shutdown_requested.is_set() or _exception_is_operator_interrupt(exc):
                    return
                logger.warning(
                    "Modal attached output forwarding failed: %s: %s", type(exc).__name__, exc
                )

        def _close_output_stream() -> None:
            try:
                trio.from_thread.run_sync(output_send.close, trio_token=trio_token)
            except BaseException as exc:
                if shutdown_requested.is_set() or _exception_is_operator_interrupt(exc):
                    return
                logger.warning(
                    "Modal attached output close failed: %s: %s", type(exc).__name__, exc
                )

        def _on_started() -> None:
            _emit("remote_entrypoint_invoked")
            _emit("remote_stdout_stream_open")
            _emit("remote_stderr_stream_open")

        def _on_stdout_line(line: str) -> None:
            if "\n" in line:
                for subline in line.splitlines():
                    _on_stdout_line(subline)
                return
            stripped = line.rstrip()
            if stdout_event_sentinel and stripped.startswith(stdout_event_sentinel):
                payload = stripped[len(stdout_event_sentinel) :]
                try:
                    import json

                    event_data = json.loads(payload)
                    event_name = event_data.pop("event", None)
                    if event_name:
                        _emit(
                            event_name,
                            **_normalize_run_logger_event_payload(
                                backend=self.backend,
                                handle_name=process_name,
                                data=event_data,
                            ),
                        )
                except Exception as exc:
                    _emit("remote_event_stream_parse_failed", error=f"{type(exc).__name__}: {exc}")
                return
            process_state["stdout_line_count"] += 1
            process_state["last_stdout_line"] = stripped
            process_state["last_stdout_elapsed_sec"] = round(_elapsed(), 3)
            stdout_lines.append(line if line.endswith("\n") else f"{line}\n")
            stdout_tail.append(stripped)
            _send_output_line(ProcessOutputLine(stream="stdout", text=stripped))
            _emit_stream_line("remote_stdout_line", process_state["stdout_line_count"], stripped)
            if startup_sentinel and startup_sentinel in stripped and not startup_seen.is_set():
                startup_seen.set()
                _emit("workload_entrypoint_started")

        def _on_stderr_line(line: str) -> None:
            stripped = line.rstrip()

            def _record_stderr_text(text: str) -> None:
                for fragment in text.splitlines():
                    fragment = fragment.rstrip()
                    if not fragment:
                        continue
                    process_state["stderr_line_count"] += 1
                    process_state["last_stderr_line"] = fragment
                    process_state["last_stderr_elapsed_sec"] = round(_elapsed(), 3)
                    stderr_lines.append(f"{fragment}\n")
                    stderr_tail.append(fragment)
                    _send_output_line(ProcessOutputLine(stream="stderr", text=fragment))
                    _emit_stream_line(
                        "remote_stderr_line",
                        process_state["stderr_line_count"],
                        fragment,
                    )

            if stderr_event_sentinel:
                # TODO(child-event-contract): `rollouts` should declare the
                # child semantic event algebra explicitly and `bifrost` should
                # carry it over a dedicated channel. Until then, keep bulky
                # service-local payload logs out of the parent lifecycle path
                # and only normalize small typed milestones here.
                plain_fragments, parsed_events, pending_buffer, parse_error = (
                    _consume_modal_diag_stream_chunk(
                        existing_buffer=str(process_state["stderr_event_buffer"]),
                        chunk=line,
                        sentinel=stderr_event_sentinel,
                    )
                )
                process_state["stderr_event_buffer"] = pending_buffer
                for fragment in plain_fragments:
                    _record_stderr_text(fragment)
                for event_name, event_data in parsed_events:
                    normalized_event_data = _normalize_run_logger_event_payload(
                        provider=self.backend,
                        backend=self.backend,
                        handle_name=process_name,
                        data=event_data,
                    )
                    _emit(event_name, **normalized_event_data)
                    _maybe_record_supervisor_completion(event_name, normalized_event_data)
                if parse_error is not None:
                    _emit("remote_diag_stream_parse_failed", error=parse_error)
                if pending_buffer or parsed_events:
                    return

            _record_stderr_text(stripped)

        def _on_heartbeat(elapsed_sec: float, silence_sec: float) -> None:
            _emit(
                "remote_process_heartbeat",
                elapsed_sec=round(elapsed_sec, 3),
                silence_sec=round(silence_sec, 3),
                stdout_line_count=process_state["stdout_line_count"],
                stderr_line_count=process_state["stderr_line_count"],
                last_stdout_elapsed_sec=process_state["last_stdout_elapsed_sec"],
                last_stderr_elapsed_sec=process_state["last_stderr_elapsed_sec"],
            )

        def _run_attached() -> None:
            try:
                stdout, stderr, exit_code = exec_modal_command_sync(
                    self.sandbox_handle.sandbox,
                    spec.build_command(),
                    timeout=timeout,
                    on_started=_on_started,
                    on_stdout_line=_on_stdout_line,
                    on_stderr_line=_on_stderr_line,
                    on_heartbeat=_on_heartbeat,
                    stop_requested=shutdown_requested,
                )
                results["stdout"] = stdout
                results["stderr"] = stderr
                results["exit_code"] = exit_code
            except BaseException as exc:
                if shutdown_requested.is_set():
                    results.setdefault("stdout", "")
                    results.setdefault("stderr", "")
                    results.setdefault("exit_code", 130)
                    return
                results["exception"] = exc
            finally:
                shutdown_requested.set()
                _close_output_stream()
                done.set()

        _emit("remote_entrypoint_invoke_start", command=spec.command)
        worker = threading.Thread(target=_run_attached, daemon=True)
        worker.start()

        if start_timeout_s is not None and startup_sentinel:
            deadline = trio.current_time() + start_timeout_s
            while not done.is_set() and not startup_seen.is_set():
                if trio.current_time() >= deadline:
                    _emit("workload_entrypoint_start_timeout", timeout_sec=start_timeout_s)
                    shutdown_requested.set()
                    await trio.to_thread.run_sync(self.sandbox_handle.sandbox.terminate)
                    worker.join(timeout=5.0)
                    results.setdefault(
                        "stderr",
                        f"Workload entrypoint did not emit startup sentinel within {start_timeout_s}s",
                    )
                    results.setdefault("stdout", "")
                    results.setdefault("exit_code", 124)
                    break
                await trio.sleep(1.0)

        def _wait() -> ExecResult:
            while True:
                _poll_supervisor_completion()
                if done.wait(timeout=0.1):
                    break
                if supervisor_completion_seen.is_set():
                    break
            pending_stderr_event = str(process_state.get("stderr_event_buffer", "")).strip()
            if pending_stderr_event:
                _emit(
                    "remote_diag_stream_parse_failed",
                    error="JSONDecodeError: modal stderr event stream ended with incomplete payload",
                    pending_chars=len(pending_stderr_event),
                )
                process_state["stderr_event_buffer"] = ""
            shutdown_requested.set()
            worker.join(timeout=1.0)
            if "exception" in results:
                raise results["exception"]
            if done.is_set():
                stdout = str(results.get("stdout", ""))
                stderr = str(results.get("stderr", ""))
                exit_code = int(results.get("exit_code", 1))
            else:
                stdout = "".join(stdout_lines)
                stderr = "".join(stderr_lines)
                exit_code = int(supervisor_completion.get("exit_code", 1))
            _emit(
                "remote_exit_observed",
                exit_code=exit_code,
                elapsed_sec=round(_elapsed(), 3),
                stdout_line_count=process_state["stdout_line_count"],
                stderr_line_count=process_state["stderr_line_count"],
                last_stdout_line=process_state["last_stdout_line"],
                last_stderr_line=process_state["last_stderr_line"],
                last_stdout_elapsed_sec=process_state["last_stdout_elapsed_sec"],
                last_stderr_elapsed_sec=process_state["last_stderr_elapsed_sec"],
                stdout_tail=list(stdout_tail),
                stderr_tail=list(stderr_tail),
                completion_event=supervisor_completion.get("event_name"),
            )
            return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

        def _terminate() -> None:
            shutdown_requested.set()
            self.sandbox_handle.sandbox.terminate()
            done.wait(timeout=5.0)
            worker.join(timeout=1.0)

        async def _stream_output() -> AsyncIterator[ProcessOutputLine]:
            async with output_receive:
                async for item in output_receive:
                    yield item

        return ObservedProcessHandle(
            name=process_name,
            backend=self.backend,
            spec=spec,
            workspace=spec.cwd,
            output_sink=OutputSink(
                kind="provider_stream",
                description="Modal attached process stdout/stderr stream",
            ),
            lifecycle_events=lifecycle_events,
            state=ProcessState.RUNNING,
            _stream_output=_stream_output,
            _wait=_wait,
            _terminate=_terminate,
            metadata={
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "process_state": process_state,
                "startup_seen": startup_seen,
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
        import shlex

        effective_spec = service.process
        workspace_root = workspace.root if workspace is not None else None
        if effective_spec.cwd is None and workspace_root is not None:
            effective_spec = ProcessSpec(
                command=effective_spec.command,
                args=effective_spec.args,
                cwd=workspace_root,
                env=effective_spec.env,
                cuda_device_ids=effective_spec.cuda_device_ids,
            )

        service_name = name
        if log_file is None:
            log_file = f"/tmp/bifrost-services/{service_name}"
        stdout_log = f"{log_file}.stdout.log"
        stderr_log = f"{log_file}.stderr.log"
        pid_file = f"{log_file}.pid"
        lifecycle_events = create_jsonl_event_stream(
            backend=self.backend,
            handle_kind="service",
            handle_name=service_name,
        )
        append_lifecycle_event(
            lifecycle_events,
            event="service_launch_requested",
            backend=self.backend,
            handle_name=service_name,
            port=service.port,
            cwd=effective_spec.cwd,
            readiness_probe=service.readiness_probe.kind,
            readiness_target=service.readiness_probe.target,
        )
        await self.exec(
            " && ".join((
                f"mkdir -p {shlex.quote(str(Path(log_file).parent))}",
                f": > {shlex.quote(stdout_log)}",
                f": > {shlex.quote(stderr_log)}",
                f"rm -f {shlex.quote(pid_file)}",
            ))
        )

        wrapped_command = (
            f"echo $$ > {shlex.quote(pid_file)}; "
            f"exec bash -lc {shlex.quote(effective_spec.build_command())} "
            f">> {shlex.quote(stdout_log)} 2>> {shlex.quote(stderr_log)}"
        )
        proc = await trio.to_thread.run_sync(
            lambda: self.sandbox_handle.sandbox.exec("bash", "-lc", wrapped_command, timeout=86400)
        )
        append_lifecycle_event(
            lifecycle_events,
            event="service_launch_succeeded",
            backend=self.backend,
            handle_name=service_name,
            service_id=f"{self.sandbox_handle.sandbox_id}:{service_name}",
            port=service.port,
        )

        async def _is_running() -> bool:
            return await trio.to_thread.run_sync(lambda: proc.poll()) is None

        async def _is_healthy() -> bool:
            if not await _is_running():
                return False
            probe = service.readiness_probe
            if probe.kind in {"none", "process_alive"}:
                return True
            if probe.kind == "http":
                assert service.port is not None, "HTTP readiness requires service.port"
                target = probe.target or "/health"
                url = (
                    target
                    if target.startswith("http://") or target.startswith("https://")
                    else (f"http://localhost:{service.port}{target}")
                )
                result = await self.exec(
                    f"curl -s -o /dev/null -w '%{{http_code}}' {shlex.quote(url)} 2>/dev/null || echo 000"
                )
                return result.stdout.strip() == "200"
            assert probe.kind == "custom", f"unsupported readiness probe kind: {probe.kind}"
            assert probe.target, "custom readiness probe requires target command"
            result = await self.exec(probe.target)
            return result.exit_code == 0

        async def _stop() -> None:
            await self.exec(
                f"test -f {shlex.quote(pid_file)} && kill -TERM $(cat {shlex.quote(pid_file)}) "
                "2>/dev/null || true"
            )

        async def _logs(tail: int) -> str:
            return (
                await self.exec(
                    " && ".join((
                        f"echo '== stdout ==' && tail -n {tail} {shlex.quote(stdout_log)} 2>/dev/null || true",
                        f"echo '== stderr ==' && tail -n {tail} {shlex.quote(stderr_log)} 2>/dev/null || true",
                    ))
                )
            ).stdout

        health_endpoint = None
        if service.readiness_probe.kind == "http" and service.readiness_probe.target:
            health_endpoint = service.readiness_probe.target
        return ServerInfo(
            name=service_name,
            service_id=f"{self.sandbox_handle.sandbox_id}:{service_name}",
            log_file=stdout_log,
            port=service.port,
            health_endpoint=health_endpoint,
            workspace=effective_spec.cwd,
            backend=self.backend,
            output_sink=OutputSink(
                kind="file",
                location=stdout_log,
                description=f"stdout mirrored to {stdout_log}; stderr mirrored to {stderr_log}",
            ),
            lifecycle_events=lifecycle_events,
            readiness_probe=service.readiness_probe,
            initial_state=ServiceState.LAUNCHING,
            _is_running=_is_running,
            _is_healthy=_is_healthy,
            _stop=_stop,
            _logs=_logs,
            metadata={
                "transport": "modal",
                "mode": "service",
                "sandbox_id": self.sandbox_handle.sandbox_id,
                "stdout_log_file": stdout_log,
                "stderr_log_file": stderr_log,
                "pid_file": pid_file,
            },
        )


async def create_modal_sandbox(request: ModalExecutionRequest) -> ModalSandboxHandle:
    """Create or attach to a Modal sandbox."""

    import modal
    import trio_asyncio
    from broker.providers.modal_image import (
        add_inference_venv_to_image,
        build_modal_image,
        eager_build_modal_image,
    )
    from rollouts.modal_workload import MODAL_APP_NAME

    if request.sandbox_id:
        logger.info("Reusing sandbox: %s", request.sandbox_id)

        def _attach() -> Any:
            return modal.Sandbox.from_id(request.sandbox_id)

        sandbox = await trio.to_thread.run_sync(_attach)
        assert sandbox is not None, f"Failed to reattach to sandbox: {request.sandbox_id}"
        assert sandbox.object_id, "Sandbox missing object_id"
        logger.info("Reattached to sandbox: %s", sandbox.object_id)
        return ModalSandboxHandle(
            sandbox=sandbox,
            sandbox_id=sandbox.object_id,
            keep_alive=request.keep_alive,
        )

    logger.info("Looking up app: %s", MODAL_APP_NAME)
    app = await trio_asyncio.aio_as_trio(
        modal.App.lookup.aio(MODAL_APP_NAME, create_if_missing=True)
    )

    owner_tags = {
        key: value for key in ("control_plane", "launcher_id") if (value := request.tags.get(key))
    }
    group_tags = {
        key: value
        for key in ("control_plane", "config_basename", "provider")
        if (value := request.tags.get(key))
    }

    def _list_owned_sandboxes() -> list[Any]:
        if request.cleanup_scope == "none":
            return []
        if request.cleanup_scope == "app":
            return list(modal.Sandbox.list(app_id=app.app_id))
        if request.cleanup_scope == "tag":
            return list(modal.Sandbox.list(app_id=app.app_id, tags=group_tags))
        assert request.cleanup_scope == "run"
        return list(modal.Sandbox.list(app_id=app.app_id, tags=owner_tags))

    cleanup_event: tuple[str, dict[str, Any]]
    if request.keep_alive:
        logger.info("Skipping pre-create sandbox cleanup because keep_alive=True")
        cleanup_event = ("modal_sandbox_cleanup_skipped", {"reason": "keep_alive_enabled"})
    elif request.cleanup_scope == "none":
        logger.info("Skipping pre-create sandbox cleanup because cleanup_scope=none")
        cleanup_event = ("modal_sandbox_cleanup_skipped", {"reason": "cleanup_scope_none"})
    elif request.cleanup_scope == "run" and not owner_tags:
        logger.info("Skipping pre-create sandbox cleanup because run owner tags are missing")
        cleanup_event = ("modal_sandbox_cleanup_skipped", {"reason": "missing_run_tags"})
    elif request.cleanup_scope == "tag" and not group_tags:
        logger.info("Skipping pre-create sandbox cleanup because tag scope tags are missing")
        cleanup_event = ("modal_sandbox_cleanup_skipped", {"reason": "missing_tag_tags"})
    else:
        if request.cleanup_scope == "run":
            cleanup_event = ("modal_sandbox_cleanup_scope", {"scope": "run", "tags": owner_tags})
        elif request.cleanup_scope == "tag":
            cleanup_event = ("modal_sandbox_cleanup_scope", {"scope": "tag", "tags": group_tags})
        else:
            cleanup_event = ("modal_sandbox_cleanup_scope", {"scope": "app"})
        existing = await trio.to_thread.run_sync(_list_owned_sandboxes)
        if existing:
            logger.info(
                "Cleaning up %s existing sandbox(es) for cleanup_scope=%s...",
                len(existing),
                request.cleanup_scope,
            )
            for sandbox in existing:
                try:
                    await trio_asyncio.aio_as_trio(sandbox.terminate.aio())
                    logger.info("  Terminated %s", sandbox.object_id)
                except Exception as exc:
                    logger.warning("  Failed to terminate %s: %s", sandbox.object_id, exc)
        else:
            logger.info(
                "No existing sandboxes found for cleanup_scope=%s",
                request.cleanup_scope,
            )

    gpu_spec = (
        f"{request.runtime.gpu_type}:{request.runtime.gpu_count}"
        if request.runtime.gpu_count > 1
        else request.runtime.gpu_type
    )
    ts = int(datetime.now(timezone.utc).timestamp())
    sandbox_name = f"rollouts-{request.runtime.gpu_type.lower()}-{ts}"
    timeout_seconds = request.timeout_hours * 3600
    primary_cmd = _modal_primary_command(request.keep_alive)
    create_timeout_s = 300
    create_heartbeat_s = 15
    create_attempts = 2

    def emit(event: str, **data: Any) -> None:
        if request.run_logger is not None:
            from rollouts.event_log import emit_run_event
            emit_run_event(
                request.run_logger,
                event,
                provider="modal",
                run_name=request.run_name,
                sandbox_name=sandbox_name,
                gpu_type=request.runtime.gpu_type,
                gpu_count=request.runtime.gpu_count,
                **data,
            )

    cleanup_event_name, cleanup_event_data = cleanup_event
    emit(cleanup_event_name, **cleanup_event_data)

    logger.info("Constructing Modal image...")
    assert request.runtime.deps is not None, "Modal deps must be present before sandbox creation"
    emit(
        "modal_image_construct_start",
        app_id=app.app_id,
        split_env=request.inference_deps is not None,
    )
    image = build_modal_image(modal, request.runtime.deps, request.runtime.gpu_type)
    if request.inference_deps is not None:
        trainer_base = request.runtime.deps.base_image
        inference_base = request.inference_deps.base_image
        assert trainer_base == inference_base, (
            f"split_env with different base images is not supported on Modal. "
            f"Modal sandboxes cannot share a GPU across separate containers, so "
            f"two-sandbox split_env would require two separate GPU allocations which "
            f"defeats the purpose. Install inference packages as a venv on top of "
            f"the trainer image instead (same base_image, separate pip_packages). "
            f"trainer={trainer_base!r} inference={inference_base!r}"
        )
        logger.info("Adding inference venv (split_env)...")
        image = add_inference_venv_to_image(
            modal, image, request.inference_deps, request.runtime.gpu_type
        )
    emit(
        "modal_image_construct_finished",
        app_id=app.app_id,
        source_ref=getattr(
            request.runtime.deps.resolved_image(request.runtime.gpu_type),
            "source_ref",
            None,
        ),
        split_env=request.inference_deps is not None,
    )
    logger.info("Eagerly building Modal image...")
    image = await eager_build_modal_image(image, app, emit)
    logger.info("Modal image ready: %s", getattr(image, "object_id", None))

    async def _create_once(attempt: int) -> Any:
        logger.info(
            "Creating sandbox: %s (gpu=%s) attempt=%s/%s",
            sandbox_name,
            gpu_spec,
            attempt,
            create_attempts,
        )
        emit("modal_sandbox_create_attempt_start", attempt=attempt, timeout_sec=create_timeout_s)

        result: dict[str, Any] = {}

        async def _create_task() -> None:
            result["sandbox"] = await trio_asyncio.aio_as_trio(
                modal.Sandbox.create.aio(
                    *primary_cmd,
                    app=app,
                    image=image,
                    gpu=gpu_spec,
                    timeout=timeout_seconds,
                    name=sandbox_name,
                    encrypted_ports=request.encrypted_ports,
                    unencrypted_ports=request.unencrypted_ports,
                    h2_ports=request.h2_ports,
                    verbose=True,
                )
            )

        start = trio.current_time()
        with trio.move_on_after(create_timeout_s) as scope:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(_create_task)
                while "sandbox" not in result:
                    elapsed = trio.current_time() - start
                    emit(
                        "modal_sandbox_create_heartbeat",
                        attempt=attempt,
                        elapsed_sec=round(elapsed, 3),
                    )
                    await trio.sleep(create_heartbeat_s)
                nursery.cancel_scope.cancel()

        if "sandbox" in result:
            elapsed = trio.current_time() - start
            emit(
                "modal_sandbox_create_attempt_succeeded",
                attempt=attempt,
                elapsed_sec=round(elapsed, 3),
            )
            return result["sandbox"]

        assert scope.cancelled_caught
        emit(
            "modal_sandbox_create_attempt_timeout",
            attempt=attempt,
            timeout_sec=create_timeout_s,
        )
        raise TimeoutError(
            f"Modal sandbox creation timed out after {create_timeout_s}s "
            f"(attempt {attempt}/{create_attempts})"
        )

    sandbox = None
    last_error: Exception | None = None
    for attempt in range(1, create_attempts + 1):
        try:
            sandbox = await _create_once(attempt)
            break
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Modal sandbox create attempt %s/%s failed: %s",
                attempt,
                create_attempts,
                exc,
            )
            if attempt == create_attempts:
                break
            emit(
                "modal_sandbox_create_retry_scheduled",
                attempt=attempt,
                next_attempt=attempt + 1,
                error=f"{type(exc).__name__}: {exc}",
            )

    if sandbox is None:
        emit(
            "modal_sandbox_create_failed",
            attempts=create_attempts,
            error=f"{type(last_error).__name__}: {last_error}" if last_error else "unknown",
        )
        raise RuntimeError(
            f"Modal sandbox creation failed after {create_attempts} attempts: {last_error}"
        ) from last_error

    assert sandbox.object_id, "Sandbox missing object_id"
    logger.info("Sandbox created: %s", sandbox.object_id)
    stabilize_window_s = 5.0
    stabilize_interval_s = 1.0
    stabilize_start = trio.current_time()
    stabilize_attempt = 0
    while True:
        stabilize_attempt += 1
        try:
            initial_returncode = await trio_asyncio.aio_as_trio(sandbox.poll.aio())
        except Exception as exc:
            emit(
                "modal_sandbox_poll_failed",
                phase="post_create_stabilization",
                attempt=stabilize_attempt,
                elapsed_sec=round(trio.current_time() - stabilize_start, 3),
                error=f"{type(exc).__name__}: {exc}",
            )
            initial_returncode = None

        sandbox_result = getattr(sandbox, "_result", None)
        elapsed = trio.current_time() - stabilize_start
        emit(
            "modal_sandbox_stabilization_probe",
            attempt=stabilize_attempt,
            elapsed_sec=round(elapsed, 3),
            returncode=initial_returncode,
            status=getattr(sandbox_result, "status", None),
            exception=getattr(sandbox_result, "exception", None),
        )
        if initial_returncode is not None:
            emit(
                "modal_sandbox_primary_exited_early",
                elapsed_sec=round(elapsed, 3),
                returncode=initial_returncode,
                status=getattr(sandbox_result, "status", None),
                exception=getattr(sandbox_result, "exception", None),
            )
            raise RuntimeError(
                "Modal sandbox primary process exited before first exec: "
                f"returncode={initial_returncode}"
            )
        if elapsed >= stabilize_window_s:
            break
        await trio.sleep(stabilize_interval_s)

    emit(
        "modal_sandbox_stabilized",
        elapsed_sec=round(trio.current_time() - stabilize_start, 3),
        probe_count=stabilize_attempt,
    )
    if request.keep_alive:
        emit("modal_sandbox_keepalive_configured", command=list(primary_cmd))
    else:
        emit(
            "modal_sandbox_parent_lease_watchdog_configured",
            command=list(primary_cmd),
            lease_path=MODAL_PARENT_LEASE_PATH,
            ttl_s=MODAL_PARENT_LEASE_TTL_S,
        )

    if request.tags:
        try:
            sandbox.set_tags(request.tags)
            emit("modal_sandbox_tags_set", tags=request.tags)
        except Exception as exc:
            emit("modal_sandbox_tags_failed", error=f"{type(exc).__name__}: {exc}")

    return ModalSandboxHandle(
        sandbox=sandbox,
        sandbox_id=sandbox.object_id,
        keep_alive=request.keep_alive,
    )


async def materialize_modal_workspace(
    sandbox: ModalSandboxHandle,
    local_root: Path,
    *,
    extra_source_roots: tuple[Path, ...] = (),
    emit: Callable[[str], None] | None = None,
) -> WorkspaceHandle:
    """Materialize the current repo into a Modal sandbox workspace."""

    workspace = await _sync_code_to_sandbox(
        sandbox.sandbox,
        local_root,
        extra_source_roots=extra_source_roots,
        emit=emit,
    )
    return WorkspaceHandle(root=workspace, backend="modal")


async def terminate_modal_sandbox(sandbox: ModalSandboxHandle) -> None:
    """Terminate a live Modal sandbox."""

    if sandbox.keep_alive:
        return

    def _terminate() -> None:
        try:
            sandbox.sandbox.terminate()
        except BaseException as exc:
            if _is_modal_not_found_error(exc):
                return
            raise

    await trio.to_thread.run_sync(_terminate)


def exec_modal_command_sync(
    sandbox: Any,
    command: str,
    timeout: int = 300,
    *,
    stream_output: bool = True,
    on_started: Callable[[], None] | None = None,
    on_stdout_line: Callable[[str], None] | None = None,
    on_stderr_line: Callable[[str], None] | None = None,
    on_heartbeat: Callable[[float, float], None] | None = None,
    heartbeat_interval_s: float = 15.0,
    stop_requested: threading.Event | None = None,
) -> tuple[str, str, int]:
    """Execute a command inside a Modal sandbox with streaming output."""

    proc = sandbox.exec("bash", "-c", command, timeout=timeout)
    if on_started is not None:
        on_started()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    thread_errors: list[BaseException] = []
    stop_heartbeat = threading.Event()
    activity_lock = threading.Lock()
    last_activity_ts = time.monotonic()

    def mark_activity() -> None:
        nonlocal last_activity_ts
        with activity_lock:
            last_activity_ts = time.monotonic()

    def read_stdout() -> None:
        try:
            for line in proc.stdout:
                stdout_lines.append(line)
                mark_activity()
                if stream_output:
                    logger.info("[sandbox] %s", line.rstrip())
                if on_stdout_line is not None:
                    on_stdout_line(line)
        except BaseException as exc:
            if stop_requested is not None and stop_requested.is_set():
                return
            thread_errors.append(exc)

    def read_stderr() -> None:
        try:
            for line in proc.stderr:
                stderr_lines.append(line)
                mark_activity()
                if stream_output:
                    logger.warning("[sandbox stderr] %s", line.rstrip())
                if on_stderr_line is not None:
                    on_stderr_line(line)
        except BaseException as exc:
            if stop_requested is not None and stop_requested.is_set():
                return
            thread_errors.append(exc)

    def emit_heartbeats() -> None:
        if on_heartbeat is None:
            return
        started_ts = time.monotonic()
        while not stop_heartbeat.wait(heartbeat_interval_s):
            with activity_lock:
                silence_sec = time.monotonic() - last_activity_ts
            elapsed_sec = time.monotonic() - started_ts
            on_heartbeat(elapsed_sec, silence_sec)

    stdout_thread = threading.Thread(target=read_stdout)
    stderr_thread = threading.Thread(target=read_stderr)
    heartbeat_thread = threading.Thread(target=emit_heartbeats, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    heartbeat_thread.start()
    stdout_thread.join()
    stderr_thread.join()

    proc.wait()
    stop_heartbeat.set()
    heartbeat_thread.join(timeout=1.0)
    if thread_errors and not (stop_requested is not None and stop_requested.is_set()):
        raise thread_errors[0]

    return "".join(stdout_lines), "".join(stderr_lines), proc.returncode


async def exec_modal_command(
    sandbox: ModalSandboxHandle,
    command: str,
    *,
    timeout: int = 300,
    stream_output: bool = True,
    on_started: Callable[[], None] | None = None,
    on_stdout_line: Callable[[str], None] | None = None,
    on_stderr_line: Callable[[str], None] | None = None,
    on_heartbeat: Callable[[float, float], None] | None = None,
    heartbeat_interval_s: float = 15.0,
    stop_requested: threading.Event | None = None,
) -> tuple[str, str, int]:
    """Execute a command inside a Modal sandbox."""

    return await trio.to_thread.run_sync(
        lambda: exec_modal_command_sync(
            sandbox.sandbox,
            command,
            timeout=timeout,
            stream_output=stream_output,
            on_started=on_started,
            on_stdout_line=on_stdout_line,
            on_stderr_line=on_stderr_line,
            on_heartbeat=on_heartbeat,
            heartbeat_interval_s=heartbeat_interval_s,
            stop_requested=stop_requested,
        )
    )


async def _sync_code_to_sandbox(
    sandbox: Any,
    local_root: Path,
    *,
    extra_source_roots: tuple[Path, ...] = (),
    emit: Callable[[str], None] | None = None,
) -> str:
    """Sync the local repo into a Modal sandbox workspace via a git bundle."""

    clone_dir = "/workspace/research"
    workspace = "/workspace/research/rollouts"

    def _emit_progress(stage: str, **data: Any) -> None:
        if emit is not None:
            emit("modal_repo_sync_progress", stage=stage, **data)

    def _sanitize_mount_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
        return cleaned or "project"

    def _remote_extra_mounts() -> list[tuple[Path, str]]:
        mounts: list[tuple[Path, str]] = []
        used_names: set[str] = set()
        for index, source_root in enumerate(extra_source_roots, start=1):
            source_name = _sanitize_mount_name(source_root.name)
            mount_name = source_name
            if mount_name in used_names:
                mount_name = f"{source_name}-{index}"
            used_names.add(mount_name)
            mounts.append((source_root, f"/workspace/external/{mount_name}"))
        return mounts

    def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        path = Path(info.name)
        excluded = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", "results"}
        if any(part in excluded for part in path.parts):
            return None
        return info

    def _sync() -> None:
        with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as f:
            bundle_path = f.name

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(local_root),
                capture_output=True,
                text=True,
                check=True,
            )
            commit = result.stdout.strip()
            logger.info("Bundling commit %s...", commit[:8])
            _emit_progress("bundle_create_start", commit=commit)

            subprocess.run(
                ["git", "bundle", "create", bundle_path, "HEAD"],
                cwd=str(local_root),
                check=True,
                capture_output=True,
            )

            bundle_size = os.path.getsize(bundle_path)
            logger.info("Bundle size: %.1f MB", bundle_size / 1024 / 1024)
            _emit_progress(
                "bundle_create_finished",
                commit=commit,
                bundle_size_mb=round(bundle_size / 1024 / 1024, 3),
            )

            with open(bundle_path, "rb") as f:
                bundle_data = f.read()

            _emit_progress("workspace_prepare_start", path="/workspace")
            exec_modal_command_sync(sandbox, "mkdir -p /workspace", timeout=30)
            _emit_progress("workspace_prepare_finished", path="/workspace")

            logger.info("Uploading bundle via sandbox.open()...")
            _emit_progress(
                "bundle_upload_start",
                remote_path="/tmp/repo.bundle",
                bundle_size_mb=round(len(bundle_data) / 1024 / 1024, 3),
            )
            remote_file = sandbox.open("/tmp/repo.bundle", "wb")
            remote_file.write(bundle_data)
            remote_file.close()
            logger.info("Uploaded %.1f MB", len(bundle_data) / 1024 / 1024)
            _emit_progress(
                "bundle_upload_finished",
                remote_path="/tmp/repo.bundle",
                bundle_size_mb=round(len(bundle_data) / 1024 / 1024, 3),
            )

            logger.info("Extracting bundle...")
            _emit_progress("clone_start", clone_dir=clone_dir, workspace=workspace)

            def _on_clone_stderr_line(line: str) -> None:
                stripped = line.rstrip()
                if not stripped:
                    return
                if "Cloning into" in stripped:
                    _emit_progress("clone_progress", message=stripped)
                    return
                if "switching to" in stripped or "Updating files:" in stripped:
                    _emit_progress("checkout_progress", message=stripped)

            exec_modal_command_sync(
                sandbox,
                "cd /workspace && git clone /tmp/repo.bundle research && "
                "cd research && git checkout HEAD",
                timeout=120,
                on_stderr_line=_on_clone_stderr_line,
            )
            _emit_progress("clone_finished", clone_dir=clone_dir, workspace=workspace)

            logger.info("Code synced to %s", workspace)
            _emit_progress("sync_finished", workspace=workspace, commit=commit)

            extra_mounts = _remote_extra_mounts()
            if extra_mounts:
                exec_modal_command_sync(sandbox, "mkdir -p /workspace/external", timeout=30)
            for source_root, remote_root in extra_mounts:
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as extra_file:
                    archive_path = extra_file.name

                remote_archive = f"/tmp/{Path(remote_root).name}.tar.gz"
                try:
                    _emit_progress(
                        "external_project_archive_start",
                        local_root=str(source_root),
                        remote_root=remote_root,
                    )
                    with tarfile.open(archive_path, "w:gz") as archive:
                        archive.add(
                            str(source_root),
                            arcname=Path(remote_root).name,
                            filter=_tar_filter,
                        )
                    archive_size = os.path.getsize(archive_path)
                    _emit_progress(
                        "external_project_archive_finished",
                        local_root=str(source_root),
                        remote_root=remote_root,
                        archive_size_mb=round(archive_size / 1024 / 1024, 3),
                    )

                    with open(archive_path, "rb") as archive_handle:
                        remote_file = sandbox.open(remote_archive, "wb")
                        remote_file.write(archive_handle.read())
                        remote_file.close()

                    _emit_progress(
                        "external_project_upload_finished",
                        local_root=str(source_root),
                        remote_root=remote_root,
                    )
                    exec_modal_command_sync(
                        sandbox,
                        f"mkdir -p {shlex.quote(str(Path(remote_root).parent))} && "
                        f"tar -xzf {shlex.quote(remote_archive)} -C "
                        f"{shlex.quote(str(Path(remote_root).parent))}",
                        timeout=120,
                    )
                    _emit_progress(
                        "external_project_extract_finished",
                        local_root=str(source_root),
                        remote_root=remote_root,
                    )
                finally:
                    try:
                        os.unlink(archive_path)
                    except FileNotFoundError:
                        pass
        finally:
            os.unlink(bundle_path)

    await trio.to_thread.run_sync(_sync)
    return workspace


async def _verify_modal_gpu(
    sandbox_handle: ModalSandboxHandle,
    *,
    sandbox_id: str,
    emit: Callable[[str], None] | None = None,
) -> None:
    """Fail loudly if the sandbox cannot see the requested GPU."""

    import trio

    def _emit(event: str, **data: Any) -> None:
        if emit is not None:
            emit(event, **data)

    _emit("modal_gpu_verify_start", sandbox_id=sandbox_id)
    gpu_verify_attempts = 3
    gpu_verify_retry_delay_s = 3.0
    command_timeout_s = 30

    for attempt in range(1, gpu_verify_attempts + 1):
        _emit(
            "modal_gpu_verify_attempt_start",
            sandbox_id=sandbox_id,
            attempt=attempt,
            timeout_sec=command_timeout_s,
        )
        start = trio.current_time()

        def _run_nvidia_smi() -> tuple[str, str, int]:
            return exec_modal_command_sync(
                sandbox_handle.sandbox,
                "nvidia-smi",
                timeout=command_timeout_s,
                stream_output=False,
            )

        try:
            with trio.fail_after(command_timeout_s + 10):
                stdout, stderr, exit_code = await trio.to_thread.run_sync(
                    _run_nvidia_smi,
                    abandon_on_cancel=True,
                )
        except trio.TooSlowError:
            elapsed = trio.current_time() - start
            _emit(
                "modal_gpu_verify_attempt_timeout",
                sandbox_id=sandbox_id,
                attempt=attempt,
                elapsed_sec=round(elapsed, 3),
                timeout_sec=command_timeout_s + 10,
            )
            if attempt < gpu_verify_attempts:
                _emit(
                    "modal_gpu_verify_retrying",
                    sandbox_id=sandbox_id,
                    attempt=attempt,
                    retry_delay_sec=gpu_verify_retry_delay_s,
                )
                await trio.sleep(gpu_verify_retry_delay_s)
                continue
            raise RuntimeError(
                f"nvidia-smi timed out after {gpu_verify_attempts} attempts in sandbox {sandbox_id}"
            ) from None
        elapsed = trio.current_time() - start
        _emit(
            "modal_gpu_verify_exec_finished",
            sandbox_id=sandbox_id,
            attempt=attempt,
            elapsed_sec=round(elapsed, 3),
            exit_code=exit_code,
        )
        if stdout:
            logger.info("[sandbox] %s", stdout)
        if stderr:
            logger.warning("[sandbox stderr] %s", stderr)
        if exit_code == 0:
            _emit(
                "modal_gpu_verified",
                sandbox_id=sandbox_id,
                elapsed_sec=round(elapsed, 3),
                attempt=attempt,
            )
            return
        _emit(
            "modal_gpu_verify_attempt_failed",
            sandbox_id=sandbox_id,
            attempt=attempt,
            exit_code=exit_code,
            elapsed_sec=round(elapsed, 3),
            stdout_tail=stdout[-1000:],
            stderr_tail=stderr[-1000:],
        )
        if attempt < gpu_verify_attempts:
            _emit(
                "modal_gpu_verify_retrying",
                sandbox_id=sandbox_id,
                attempt=attempt,
                retry_delay_sec=gpu_verify_retry_delay_s,
            )
            await trio.sleep(gpu_verify_retry_delay_s)

    raise RuntimeError(f"nvidia-smi failed after {gpu_verify_attempts} attempts")


async def _wait_for_modal_exec_ready(
    sandbox_handle: ModalSandboxHandle,
    *,
    sandbox_id: str,
    emit: Callable[[str], None] | None = None,
) -> None:
    """Wait until the Modal sandbox can reliably spawn exec commands.

    `sandbox.poll() is None` only proves the sandbox primary is still alive.
    The actual exec boundary is closer to "Modal has assigned a sandbox task ID
    and a cheap exec can start promptly".
    """

    import trio
    import trio_asyncio

    def _emit(event: str, **data: Any) -> None:
        if emit is not None:
            emit(event, **data)

    sandbox = sandbox_handle.sandbox
    task_id_attempts = 6
    task_id_timeout_s = 20
    retry_delay_s = 2.0
    exec_probe_attempts = 3
    exec_probe_timeout_s = 10

    _emit("modal_exec_ready_wait_start", sandbox_id=sandbox_id)

    task_id_getter = getattr(getattr(sandbox, "_get_task_id", None), "aio", None)
    if task_id_getter is not None:
        # TODO(modal-sdk-boundary): This uses Modal's private `_get_task_id`
        # because the SDK does not expose a public "wait until exec-ready"
        # primitive. Replace this with a public readiness boundary if Modal adds
        # one; until then, this is a more honest proxy than treating `poll() is
        # None` as proof that exec will start promptly.
        for attempt in range(1, task_id_attempts + 1):
            _emit(
                "modal_exec_ready_task_id_attempt_start",
                sandbox_id=sandbox_id,
                attempt=attempt,
                timeout_sec=task_id_timeout_s,
            )
            start = trio.current_time()
            try:
                with trio.fail_after(task_id_timeout_s):
                    task_id = await trio_asyncio.aio_as_trio(task_id_getter())
            except trio.TooSlowError:
                elapsed = trio.current_time() - start
                _emit(
                    "modal_exec_ready_task_id_attempt_timeout",
                    sandbox_id=sandbox_id,
                    attempt=attempt,
                    elapsed_sec=round(elapsed, 3),
                    timeout_sec=task_id_timeout_s,
                )
                if attempt < task_id_attempts:
                    _emit(
                        "modal_exec_ready_retrying",
                        sandbox_id=sandbox_id,
                        attempt=attempt,
                        retry_delay_sec=retry_delay_s,
                        phase="task_id",
                    )
                    await trio.sleep(retry_delay_s)
                    continue
                raise RuntimeError(
                    f"Modal sandbox {sandbox_id} never became exec-ready: task id unavailable"
                ) from None
            else:
                elapsed = trio.current_time() - start
                _emit(
                    "modal_exec_ready_task_id_ready",
                    sandbox_id=sandbox_id,
                    attempt=attempt,
                    elapsed_sec=round(elapsed, 3),
                    task_id=task_id,
                )
                break

    def _run_exec_probe() -> tuple[str, str, int]:
        return exec_modal_command_sync(
            sandbox,
            "true",
            timeout=exec_probe_timeout_s,
            stream_output=False,
        )

    for attempt in range(1, exec_probe_attempts + 1):
        _emit(
            "modal_exec_ready_exec_probe_start",
            sandbox_id=sandbox_id,
            attempt=attempt,
            timeout_sec=exec_probe_timeout_s,
        )
        start = trio.current_time()

        try:
            with trio.fail_after(exec_probe_timeout_s + 5):
                stdout, stderr, exit_code = await trio.to_thread.run_sync(
                    _run_exec_probe,
                    abandon_on_cancel=True,
                )
        except trio.TooSlowError:
            elapsed = trio.current_time() - start
            _emit(
                "modal_exec_ready_exec_probe_timeout",
                sandbox_id=sandbox_id,
                attempt=attempt,
                elapsed_sec=round(elapsed, 3),
                timeout_sec=exec_probe_timeout_s + 5,
            )
            if attempt < exec_probe_attempts:
                _emit(
                    "modal_exec_ready_retrying",
                    sandbox_id=sandbox_id,
                    attempt=attempt,
                    retry_delay_sec=retry_delay_s,
                    phase="exec_probe",
                )
                await trio.sleep(retry_delay_s)
                continue
            raise RuntimeError(
                f"Modal sandbox {sandbox_id} never became exec-ready: cheap exec probe timed out"
            ) from None

        elapsed = trio.current_time() - start
        _emit(
            "modal_exec_ready_exec_probe_finished",
            sandbox_id=sandbox_id,
            attempt=attempt,
            elapsed_sec=round(elapsed, 3),
            exit_code=exit_code,
        )
        if exit_code == 0:
            _emit("modal_exec_ready", sandbox_id=sandbox_id, elapsed_sec=round(elapsed, 3))
            return

        _emit(
            "modal_exec_ready_exec_probe_failed",
            sandbox_id=sandbox_id,
            attempt=attempt,
            elapsed_sec=round(elapsed, 3),
            exit_code=exit_code,
            stdout_tail=stdout[-1000:],
            stderr_tail=stderr[-1000:],
        )
        if attempt < exec_probe_attempts:
            _emit(
                "modal_exec_ready_retrying",
                sandbox_id=sandbox_id,
                attempt=attempt,
                retry_delay_sec=retry_delay_s,
                phase="exec_probe",
            )
            await trio.sleep(retry_delay_s)
            continue
        raise RuntimeError(
            f"Modal sandbox {sandbox_id} failed exec-ready probe with exit code {exit_code}"
        )


def _build_argus_local_process_spec(
    *,
    workspace: str,
    config_path: str,
    extra_source_roots: tuple[str, ...] = (),
    run_name: str,
    deps: Any,
    gpu_type: str,
    inference_deps: Any = None,
) -> ProcessSpec:
    """Lower the Modal workload launch into one observed process spec."""

    # TODO(workload-staging): This still launches an inner `argus.run --local`
    # control plane via the supervisor trampoline. Replace it with one resolved
    # workload launch spec produced by Argus so Modal and RunPod execute the
    # same denotation and `rollouts.modal_runner` can disappear completely.
    # If we adopt MiniRay/Heinrich-style worker semantics here, the honest place
    # is *inside* the sandbox as the local child-process substrate. Modal/Bifrost
    # should still own provider/session/materialization semantics above it.
    from rollouts.modal_workload import (
        IMAGE_VENV_DIR,
        IMAGE_VENV_PYTHON,
        REPO_ROOT,
        sandbox_runtime_diag_python,
        sandbox_runtime_supervisor_python,
    )

    def _sanitize_mount_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
        return cleaned or "project"

    remote_extra_roots: list[tuple[Path, str]] = []
    used_names: set[str] = set()
    for index, source_root in enumerate(extra_source_roots, start=1):
        source_path = Path(source_root).resolve()
        mount_name = _sanitize_mount_name(source_path.name)
        if mount_name in used_names:
            mount_name = f"{mount_name}-{index}"
        used_names.add(mount_name)
        remote_extra_roots.append((source_path, f"/workspace/external/{mount_name}"))

    config_p = Path(config_path)
    if config_p.is_absolute():
        try:
            config_rel = config_p.relative_to(REPO_ROOT)
            remote_config_path = str(config_rel)
        except ValueError:
            remote_config_path = None
            for local_root, remote_root in remote_extra_roots:
                try:
                    config_rel = config_p.relative_to(local_root)
                except ValueError:
                    continue
                remote_config_path = f"{remote_root}/{config_rel.as_posix()}"
                break
            if remote_config_path is None:
                raise ValueError(
                    f"Modal process spec cannot map config path outside staged sources: {config_path}"
                )
    else:
        remote_config_path = str(config_p)

    image_python = IMAGE_VENV_PYTHON
    image_path_prefix = f"{IMAGE_VENV_DIR}/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    if deps is not None:
        image = deps.resolved_image(gpu_type)
        if image.python_runtime == "image_owned":
            image_python = image.python_executable
            image_path_prefix = (
                "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            )

    pythonpath_entries = [workspace, "/workspace/research", "/root/Megatron-LM", "/root"]
    pythonpath_entries.extend(remote_root for _, remote_root in remote_extra_roots)

    env = {
        "PYTHONUNBUFFERED": "1",
        "PATH": image_path_prefix,
        "PYTHONPATH": ":".join(dict.fromkeys(pythonpath_entries)),
        "ARGUS_EMIT_STARTUP_SENTINEL": "1",
        "ARGUS_RUN_EVENT_STREAM": "1",
        "ARGUS_SUPERVISOR_STATUS_FILE": f"{workspace}/results/rl/{run_name}/modal_supervisor_status.json",
        "ROLLOUTS_RUN_NAME": run_name,
        "ROLLOUTS_OUTPUT_DIR": f"results/rl/{run_name}",
    }
    if inference_deps is not None:
        from broker.providers.modal_image import INFERENCE_VENV_PYTHON

        env["ROLLOUTS_INFERENCE_PYTHON"] = INFERENCE_VENV_PYTHON

    return ProcessSpec(
        command=image_python,
        args=(
            "-u",
            "-c",
            sandbox_runtime_supervisor_python(),
            workspace,
            image_python,
            sandbox_runtime_diag_python(),
            remote_config_path,
        ),
        cwd=workspace,
        env=env,
    )


async def run_modal_request(request: ModalExecutionRequest) -> dict[str, Any]:
    """Run a Modal-backed workload through the bifrost-owned Modal session."""

    import modal
    import trio_asyncio
    from rollouts.event_log import ARGUS_RUN_EVENT_SENTINEL
    from rollouts.modal_workload import (
        ARGUS_DIAG_EVENT_SENTINEL,
        REPO_ROOT,
        WORKLOAD_ENTRYPOINT_SENTINEL,
        collect_modal_failure_diagnostics,
        download_and_snapshot_model,
        download_prune_and_snapshot_model,
        get_cached_snapshot,
        mount_cached_weights,
        prune_mounted_model_and_snapshot,
        save_snapshot_to_cache,
    )
    from rollouts.remote_runtime import enforce_source_sync_policy

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = request.run_name or f"modal_{timestamp}"
    local_run_dir: Path | None = None
    emit_event_sink = getattr(request.run_logger, "emit_event", None)
    local_run_log = getattr(emit_event_sink, "log_file", None)
    if local_run_log is not None:
        local_run_dir = Path(local_run_log).parent

    def emit(event: str, **data: Any) -> None:
        if request.run_logger is not None:
            from rollouts.event_log import emit_run_event
            emit_run_event(
                request.run_logger,
                event,
                provider="modal",
                run_name=run_name,
                **_normalize_run_logger_event_payload(
                    run_name=run_name,
                    provider="modal",
                    data=data,
                ),
            )

    def journal_artifact_event(event: str, **data: Any) -> None:
        if local_run_log is not None:
            _append_run_journal_event(
                Path(local_run_log),
                run_name=run_name,
                provider="modal",
                event=event,
                data=data,
            )
            return
        emit(event, **data)

    enforce_source_sync_policy(request.source_sync_policy, repo_root=REPO_ROOT)
    emit(
        "submit_start",
        config_path=request.config_path,
        gpu_type=request.runtime.gpu_type,
        gpu_count=request.runtime.gpu_count,
    )

    with modal.enable_output():
        async with trio_asyncio.open_loop():
            sandbox_handle = await create_modal_sandbox(request)
            session = ModalExecutionSession(
                sandbox_handle=sandbox_handle,
                local_root=REPO_ROOT,
                extra_source_roots=tuple(Path(root) for root in request.extra_source_roots),
            )
            sandbox_id = sandbox_handle.sandbox_id
            emit("modal_sandbox_created", sandbox_id=sandbox_id)

            try:
                await _wait_for_modal_exec_ready(sandbox_handle, sandbox_id=sandbox_id, emit=emit)
                if not request.keep_alive:
                    await _refresh_modal_parent_lease(sandbox_handle.sandbox)
                    emit(
                        "modal_parent_lease_initialized",
                        sandbox_id=sandbox_id,
                        lease_path=MODAL_PARENT_LEASE_PATH,
                        ttl_s=MODAL_PARENT_LEASE_TTL_S,
                    )
                await _verify_modal_gpu(sandbox_handle, sandbox_id=sandbox_id, emit=emit)

                if request.model_name:
                    cached_snapshot = await get_cached_snapshot(
                        request.model_name,
                        request.pruning_recipe,
                    )
                    if cached_snapshot:
                        await mount_cached_weights(sandbox_handle.sandbox, cached_snapshot)
                    else:
                        if request.pruning_recipe:
                            base_snapshot = await get_cached_snapshot(request.model_name, None)
                            if base_snapshot:
                                await mount_cached_weights(sandbox_handle.sandbox, base_snapshot)
                                snapshot = await prune_mounted_model_and_snapshot(
                                    sandbox_handle.sandbox,
                                    request.model_name,
                                    request.pruning_recipe,
                                )
                            else:
                                snapshot = await download_prune_and_snapshot_model(
                                    sandbox_handle.sandbox,
                                    request.model_name,
                                    request.pruning_recipe,
                                )
                        else:
                            snapshot = await download_and_snapshot_model(
                                sandbox_handle.sandbox,
                                request.model_name,
                            )
                        if snapshot:
                            await save_snapshot_to_cache(
                                request.model_name,
                                snapshot,
                                request.pruning_recipe,
                            )

                emit("modal_repo_sync_start", sandbox_id=sandbox_id)
                workspace = await session.materialize(
                    WorkspaceMaterializationSpec(
                        requested_root=getattr(request.materialization, "workspace_root", None)
                    ),
                    emit=emit,
                )
                emit("modal_repo_synced", sandbox_id=sandbox_id, workspace=workspace.root)

                emit("modal_training_start", sandbox_id=sandbox_id, workspace=workspace.root)
                remote_run_dir = f"{workspace.root}/results/rl/{run_name}"
                remote_training_jsonl = f"{remote_run_dir}/training.jsonl"
                artifact_poll_stop = trio.Event()
                process = await session.start_process(
                    _build_argus_local_process_spec(
                        workspace=workspace.root,
                        config_path=request.config_path,
                        extra_source_roots=request.extra_source_roots,
                        run_name=run_name,
                        deps=request.runtime.deps,
                        gpu_type=request.runtime.gpu_type,
                        inference_deps=request.inference_deps,
                    ),
                    name=run_name,
                    timeout=14400,
                    emit=emit,
                    startup_sentinel=WORKLOAD_ENTRYPOINT_SENTINEL,
                    stdout_event_sentinel=ARGUS_RUN_EVENT_SENTINEL,
                    stderr_event_sentinel=ARGUS_DIAG_EVENT_SENTINEL,
                    start_timeout_s=60.0,
                )
                try:
                    async with trio.open_nursery() as nursery:
                        if not request.keep_alive:
                            nursery.start_soon(
                                _maintain_modal_parent_lease,
                                sandbox_handle.sandbox,
                                emit,
                            )
                        nursery.start_soon(
                            _project_modal_run_artifacts,
                            sandbox_handle.sandbox,
                            run_name,
                            journal_artifact_event,
                            artifact_poll_stop,
                            remote_run_dir,
                            local_run_dir,
                        )
                        try:
                            result = await _wait_observed_process_nonblocking(process)
                        finally:
                            artifact_poll_stop.set()
                            nursery.cancel_scope.cancel()
                except BaseException as exc:
                    if not _exception_is_operator_interrupt(exc):
                        raise
                    emit(
                        "modal_training_interrupted",
                        sandbox_id=sandbox_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    with trio.CancelScope(shield=True):
                        await process.terminate()
                    emit(
                        "modal_training_finished",
                        sandbox_id=sandbox_id,
                        success=False,
                        exit_code=130,
                        cancelled=True,
                    )
                    return {
                        "success": False,
                        "exit_code": 130,
                        "stderr": "Interrupted by local operator",
                        "cancelled": True,
                    }

                if result.exit_code != 0:
                    failure_diagnostics = await collect_modal_failure_diagnostics(
                        sandbox_handle.sandbox,
                        exit_code=result.exit_code,
                        emit=emit,
                    )
                    if local_run_dir is not None:
                        await _copy_modal_text_artifact(
                            sandbox_handle.sandbox,
                            remote_path=remote_training_jsonl,
                            local_path=local_run_dir / "training.jsonl",
                            emit=emit,
                            event_prefix="modal_training_log_copy",
                        )
                    emit(
                        "modal_training_finished",
                        sandbox_id=sandbox_id,
                        success=False,
                        exit_code=result.exit_code,
                    )
                    return {
                        "success": False,
                        "exit_code": result.exit_code,
                        "stderr": result.stderr,
                        "failure_diagnostics": failure_diagnostics,
                    }

                if local_run_dir is not None:
                    await _copy_modal_text_artifact(
                        sandbox_handle.sandbox,
                        remote_path=remote_training_jsonl,
                        local_path=local_run_dir / "training.jsonl",
                        emit=emit,
                        event_prefix="modal_training_log_copy",
                    )
                emit(
                    "modal_training_finished",
                    sandbox_id=sandbox_id,
                    success=True,
                    exit_code=result.exit_code,
                )
                return {"success": True, "exit_code": 0}
            finally:
                if request.keep_alive:
                    emit("modal_sandbox_kept_alive", sandbox_id=sandbox_id)
                else:
                    emit("modal_sandbox_terminate_start", sandbox_id=sandbox_id)
                    await session.terminate()
                    emit("modal_sandbox_terminated", sandbox_id=sandbox_id)
