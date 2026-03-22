import asyncio
import inspect

import pytest
import trio

from bifrost import (
    EventStreamRef,
    ExecutionSession,
    LifecycleEvent,
    ObservedProcessHandle,
    OutputSink,
    ProcessHandle,
    ProcessOutputLine,
    ProcessSpec,
    ProcessState,
    ReadinessProbe,
    ServiceHandle,
    ServiceSpec,
    ServiceState,
    WorkspaceHandle,
    WorkspaceMaterializationSpec,
    connect,
    create_jsonl_event_stream,
    read_lifecycle_events,
)
from bifrost.modal_backend import (
    _exception_is_operator_interrupt,
    _modal_supervisor_exit_code,
    _normalize_run_logger_event_payload,
)
from bifrost.server import server_is_healthy
from bifrost.types import ExecResult, JobInfo, ServerInfo


def test_job_info_exposes_process_handle_shape() -> None:
    job = JobInfo(name="train", tmux_session="bifrost-job-train")

    assert isinstance(job, ProcessHandle)
    assert job.backend == "ssh"
    assert job.output_sink == OutputSink()
    assert job.lifecycle_events == EventStreamRef()
    assert job.initial_state is ProcessState.CREATED


def test_server_info_exposes_service_handle_shape() -> None:
    server = ServerInfo(
        name="sglang",
        service_id="service-123",
        tmux_session="bifrost-server-sglang",
        port=30000,
        health_endpoint="/health",
    )

    assert isinstance(server, ServiceHandle)
    assert server.backend == "ssh"
    assert server.readiness_probe == ReadinessProbe()
    assert server.initial_state is ServiceState.CREATED
    assert server.url == "http://localhost:30000"


def test_service_handle_exposes_readiness_and_stop_surface() -> None:
    healthy = True
    running = True
    stopped = False

    def _is_running() -> bool:
        return running

    def _is_healthy() -> bool:
        return healthy

    def _stop() -> None:
        nonlocal stopped
        stopped = True

    def _logs(tail: int) -> str:
        return f"tail={tail}"

    server = ServerInfo(
        name="qed-vllm",
        service_id="modal:sandbox:qed-vllm",
        backend="modal",
        readiness_probe=ReadinessProbe(kind="http", target="/health"),
        _is_running=_is_running,
        _is_healthy=_is_healthy,
        _stop=_stop,
        _logs=_logs,
    )

    async def _exercise() -> None:
        assert await server.is_running() is True
        assert await server.is_healthy() is True
        assert await server.wait_until_healthy(timeout=0.01, poll_interval=0.001) is True
        assert await server.logs(7) == "tail=7"
        await server.stop()
        assert stopped is True

    asyncio.run(_exercise())


def test_service_handle_wait_until_healthy_works_under_trio() -> None:
    attempts = 0

    def _is_healthy() -> bool:
        nonlocal attempts
        attempts += 1
        return attempts >= 2

    server = ServerInfo(
        name="slime-sglang",
        service_id="service-456",
        readiness_probe=ReadinessProbe(kind="http", target="/health"),
        _is_running=lambda: True,
        _is_healthy=_is_healthy,
    )

    async def _exercise() -> None:
        assert await server.wait_until_healthy(timeout=0.05, poll_interval=0.001) is True

    trio.run(_exercise)


def test_sync_server_helpers_fail_honestly_inside_asyncio_loop() -> None:
    server = ServerInfo(
        name="qed-vllm",
        service_id="service-789",
        readiness_probe=ReadinessProbe(kind="none"),
        _is_running=lambda: True,
    )

    async def _exercise() -> None:
        with pytest.raises(RuntimeError, match="cannot be called from a running asyncio loop"):
            server_is_healthy(object(), server)

    asyncio.run(_exercise())


def test_workspace_handle_requires_explicit_root() -> None:
    workspace = WorkspaceHandle(root="/tmp/workspace", source_ref="abc123")

    assert workspace.root == "/tmp/workspace"
    assert workspace.backend == "ssh"
    assert workspace.source_ref == "abc123"


def test_workspace_materialization_spec_keeps_requested_root_optional() -> None:
    spec = WorkspaceMaterializationSpec(requested_root=None, allow_dirty=True)

    assert spec.requested_root is None
    assert spec.source_mode == "git_bundle_committed"
    assert spec.allow_dirty is True


def test_service_spec_preserves_readiness_probe() -> None:
    probe = ReadinessProbe(kind="http", target="/health", timeout_s=30)
    service = ServiceSpec(
        process=ProcessSpec(command="python"),
        port=30000,
        readiness_probe=probe,
    )

    assert service.port == 30000
    assert service.readiness_probe == probe


def test_connect_returns_execution_session_protocol() -> None:
    assert inspect.iscoroutinefunction(connect)
    assert hasattr(ExecutionSession, "__dict__")


def test_process_output_line_preserves_explicit_stream() -> None:
    line = ProcessOutputLine(stream="stderr", text="boom")

    assert line.stream == "stderr"
    assert line.text == "boom"


def test_modal_event_payload_normalization_preserves_child_identity() -> None:
    payload = _normalize_run_logger_event_payload(
        run_name="outer-run",
        provider="modal",
        data={
            "run_name": "inner-run",
            "provider": "local",
            "phase": "inference_startup",
        },
    )

    assert payload == {
        "child_run_name": "inner-run",
        "child_provider": "local",
        "phase": "inference_startup",
    }


def test_modal_event_payload_normalization_protects_parent_lifecycle_keys() -> None:
    payload = _normalize_run_logger_event_payload(
        backend="modal",
        handle_name="attached-train",
        data={
            "backend": "megatron",
            "handle_name": "inner-workload",
            "phase": "training_preflight",
        },
    )

    assert payload == {
        "child_backend": "megatron",
        "child_handle_name": "inner-workload",
        "phase": "training_preflight",
    }


def test_modal_interrupt_detection_handles_exception_groups() -> None:
    exc = BaseExceptionGroup("shutdown", [KeyboardInterrupt()])

    assert _exception_is_operator_interrupt(exc) is True


def test_modal_supervisor_exit_code_uses_child_exit_boundary() -> None:
    assert _modal_supervisor_exit_code("other_event", {"child_returncode": 0}) is None
    assert _modal_supervisor_exit_code("remote_supervisor_child_exit", {"child_returncode": 0}) == 0
    assert _modal_supervisor_exit_code("remote_supervisor_child_exit", {"child_returncode": 7}) == 7
    assert (
        _modal_supervisor_exit_code("remote_supervisor_child_exit", {"child_returncode": -15})
        == 143
    )


def test_observed_process_handle_exposes_live_process_surface() -> None:
    waited = False
    terminated = False

    def _stream() -> list[str]:
        return ["one", "two"]

    def _wait() -> ExecResult:
        nonlocal waited
        waited = True
        return ExecResult(stdout="one\ntwo\n", stderr="", exit_code=0)

    def _terminate() -> None:
        nonlocal terminated
        terminated = True

    handle = ObservedProcessHandle(
        name="attached-train",
        backend="ssh",
        spec=ProcessSpec(command="python", args=("train.py",)),
        _stream_output=lambda: iter([
            ProcessOutputLine(stream="stdout", text="one"),
            ProcessOutputLine(stream="stderr", text="two"),
        ]),
        _wait=_wait,
        _terminate=_terminate,
    )

    async def _exercise() -> None:
        lines = [line async for line in handle.stream_output()]
        assert lines == [
            ProcessOutputLine(stream="stdout", text="one"),
            ProcessOutputLine(stream="stderr", text="two"),
        ]
        result = await handle.wait()
        assert result.exit_code == 0
        assert waited is True
        await handle.terminate()
        assert terminated is True

    asyncio.run(_exercise())


def test_lifecycle_event_stream_records_parent_owned_process_events() -> None:
    stream = create_jsonl_event_stream(
        backend="ssh",
        handle_kind="process",
        handle_name="attached-train",
    )
    handle = ObservedProcessHandle(
        name="attached-train",
        backend="ssh",
        spec=ProcessSpec(command="python", args=("train.py",)),
        lifecycle_events=stream,
        _stream_output=lambda: iter([ProcessOutputLine(stream="stdout", text="one")]),
        _wait=lambda: ExecResult(stdout="one\n", stderr="", exit_code=0),
        _terminate=lambda: None,
    )

    async def _exercise() -> None:
        _ = [line async for line in handle.stream_output()]
        _ = await handle.wait()
        await handle.terminate()

    asyncio.run(_exercise())
    events = read_lifecycle_events(stream)
    assert [event.event for event in events] == [
        "process_output_stream_started",
        "process_exit_observed",
        "process_termination_requested",
    ]
    assert all(event.handle_name == "attached-train" for event in events)
    assert all(event.backend == "ssh" for event in events)


def test_service_handle_records_canonical_lifecycle_events() -> None:
    stream = create_jsonl_event_stream(
        backend="modal",
        handle_kind="service",
        handle_name="qed-vllm",
    )
    server = ServerInfo(
        name="qed-vllm",
        service_id="service-123",
        backend="modal",
        lifecycle_events=stream,
        readiness_probe=ReadinessProbe(kind="http", target="/health"),
        _is_running=lambda: True,
        _is_healthy=lambda: True,
        _stop=lambda: None,
    )

    async def _exercise() -> None:
        assert await server.wait_until_healthy(timeout=0.01, poll_interval=0.001) is True
        await server.stop()

    asyncio.run(_exercise())
    events = read_lifecycle_events(stream)
    assert [event.event for event in events] == [
        "service_wait_until_healthy_started",
        "service_health_check",
        "service_ready",
        "service_stop_requested",
        "service_stop_completed",
    ]
    assert all(isinstance(event, LifecycleEvent) for event in events)
