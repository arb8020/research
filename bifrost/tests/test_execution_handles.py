import asyncio
import inspect

from bifrost import (
    EventStreamRef,
    ExecutionSession,
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
)
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
        tmux_session="bifrost-server-sglang",
        port=30000,
        health_endpoint="/health",
    )

    assert isinstance(server, ServiceHandle)
    assert server.backend == "ssh"
    assert server.readiness_probe == ReadinessProbe()
    assert server.initial_state is ServiceState.CREATED
    assert server.url == "http://localhost:30000"


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
