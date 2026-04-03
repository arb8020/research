from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import trio

from rollouts.eval.configs import EndpointConfig, InferenceServerConfig
from rollouts.eval.endpoint_realization import (
    _emit_log_lines,
    _legacy_worker_from_eval_surface,
    _wait_for_modal_sandbox_baseline,
    _remote_inference_python,
    _remote_service_spec,
    _tail_remote_trace,
    realize_worker_backed_endpoint,
)
from rollouts.eval.run import run_with_sglang_provision
from rollouts.image_spec import ImageSpec
from rollouts.run_logger import RunLogger
from rollouts.training.configs import (
    DepsConfig,
    HardwareConfig,
    InferenceConfig,
    InferenceRoleBinding,
    InferenceWorkerConfig,
    WorkerTopologyConfig,
)


class _FakeEngine:
    def __init__(self) -> None:
        self.api_base = "http://localhost:30000/v1"
        self.launch_called = False
        self.tailer_called = False
        self.ready_waits: list[float] = []
        self.shutdown_called = False

    def launch(self) -> str:
        self.launch_called = True
        return "fake-session"

    def start_log_tailer(self) -> object:
        self.tailer_called = True
        return object()

    async def wait_until_ready(self, max_wait: float = 120.0) -> None:
        self.ready_waits.append(max_wait)

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_legacy_worker_from_eval_surface_uses_endpoint_and_server_settings() -> None:
    worker = _legacy_worker_from_eval_surface(
        endpoint_config=EndpointConfig(provider="sglang", model="Qwen/Qwen2.5-0.5B-Instruct"),
        server_config=InferenceServerConfig(
            port=31000,
            mem_fraction=0.55,
            tensor_parallel_size=1,
            startup_timeout=123,
        ),
    )

    assert worker.worker_id == "eval-endpoint"
    assert worker.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert worker.inference.spec == "slime-sglang"
    assert worker.inference.port == 31000
    assert worker.inference.mem_fraction == 0.55
    assert worker.inference.startup_timeout == 123


def test_remote_inference_python_uses_remote_runtime_contract() -> None:
    default_managed = HardwareConfig(provider="modal", gpu_count=1, deps=DepsConfig())
    assert _remote_inference_python(default_managed) == "/opt/venvs/rollouts/bin/python"

    managed = HardwareConfig(
        provider="modal",
        gpu_count=1,
        deps=DepsConfig(
            image=ImageSpec.from_registry(
                "nvidia/cuda:12.4.0-devel-ubuntu22.04",
                python_runtime="managed_venv",
            )
        ),
    )
    assert _remote_inference_python(managed) == "/opt/venvs/rollouts/bin/python"

    image_owned = HardwareConfig(
        provider="modal",
        gpu_count=1,
        deps=DepsConfig(
            image=ImageSpec.from_registry(
                "slimerl/slime:v0.2.3",
                python_runtime="image_owned",
                python_executable="python3",
            )
        ),
    )
    assert _remote_inference_python(image_owned) == "python3"


def test_remote_service_spec_builds_with_remote_python(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRemoteEngine:
        health_url = "http://localhost:30000/health"

        def build_launch_cmd(self) -> str:
            import os

            return os.environ["ROLLOUTS_INFERENCE_PYTHON"]

    monkeypatch.delenv("ROLLOUTS_INFERENCE_PYTHON", raising=False)
    monkeypatch.setattr(
        "rollouts.eval.endpoint_realization._build_engine",
        lambda **_: _FakeRemoteEngine(),
    )

    launch_cmd, readiness_target = _remote_service_spec(
        worker=InferenceWorkerConfig(
            worker_id="actor",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            inference=InferenceConfig(),
        ),
        output_dir=Path("/tmp/unused"),
        remote_python="/opt/venvs/rollouts/bin/python",
    )

    assert launch_cmd == "/opt/venvs/rollouts/bin/python"
    assert readiness_target == "/health"
    assert "ROLLOUTS_INFERENCE_PYTHON" not in os.environ


def test_emit_log_lines_emits_startup_phase_once() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    run_logger = RunLogger(emit_event=lambda event, **data: events.append((event, data)))

    _emit_log_lines(
        run_logger=run_logger,
        log_blob=(
            "== stdout ==\n"
            "Started server process [123]\n"
            "Started server process [123]\n"
            "== stderr ==\n"
            "Application startup complete.\n"
        ),
        seen_lines={"stdout": set(), "stderr": set()},
        emitted_startup_phases=set(),
        startup_context={"service_name": "eval-endpoint"},
    )

    phase_events = [event for event, _data in events if event == "inference_startup_phase"]
    assert phase_events == ["inference_startup_phase", "inference_startup_phase"]
    assert [event for event, _data in events if event == "eval_inference_service_log"] == [
        "eval_inference_service_log",
        "eval_inference_service_log",
    ]


@pytest.mark.trio
async def test_tail_remote_trace_is_best_effort() -> None:
    class _FailingSession:
        async def exec(self, _command: str) -> object:
            raise RuntimeError("sandbox unavailable")

    trace_tail = await _tail_remote_trace(
        session=_FailingSession(),
        trace_path=Path("/tmp/trace.jsonl"),
    )

    assert trace_tail == "<failed to read remote trace: RuntimeError: sandbox unavailable>"


@pytest.mark.trio
async def test_wait_for_modal_sandbox_baseline_emits_cleanup_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    run_logger = RunLogger(emit_event=lambda event, **data: events.append((event, data)))
    sandbox_sets = iter(({"sb-old", "sb-new"}, {"sb-old"}))

    async def fake_list_modal_sandbox_ids() -> set[str]:
        return next(sandbox_sets)

    monkeypatch.setattr(
        "rollouts.eval.endpoint_realization._list_modal_sandbox_ids",
        fake_list_modal_sandbox_ids,
    )

    await _wait_for_modal_sandbox_baseline(
        baseline_ids={"sb-old"},
        run_logger=run_logger,
        run_name="run-test",
        timeout_s=0.01,
    )

    assert events[-1][0] == "modal_sandbox_cleanup_converged"


@pytest.mark.trio
async def test_wait_for_modal_sandbox_baseline_emits_incomplete_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    run_logger = RunLogger(emit_event=lambda event, **data: events.append((event, data)))

    async def fake_list_modal_sandbox_ids() -> set[str]:
        return {"sb-old", "sb-stuck"}

    async def fake_sleep(_seconds: float) -> None:
        return None

    clock = {"now": 0.0}

    def fake_current_time() -> float:
        value = clock["now"]
        clock["now"] += 1.0
        return value

    monkeypatch.setattr(
        "rollouts.eval.endpoint_realization._list_modal_sandbox_ids",
        fake_list_modal_sandbox_ids,
    )
    monkeypatch.setattr("rollouts.eval.endpoint_realization.trio.sleep", fake_sleep)
    monkeypatch.setattr("rollouts.eval.endpoint_realization.trio.current_time", fake_current_time)

    await _wait_for_modal_sandbox_baseline(
        baseline_ids={"sb-old"},
        run_logger=run_logger,
        run_name="run-test",
        timeout_s=0.5,
    )

    assert events[-1][0] == "modal_sandbox_cleanup_incomplete"
    assert events[-1][1]["residual_sandbox_ids"] == ["sb-stuck"]


@pytest.mark.trio
async def test_realize_worker_backed_endpoint_launches_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_engine = _FakeEngine()
    monkeypatch.setattr(
        "rollouts.eval.endpoint_realization._build_engine",
        lambda **_: fake_engine,
    )

    async with realize_worker_backed_endpoint(
        endpoint_config=EndpointConfig(provider="sglang", model="Qwen/Qwen2.5-0.5B-Instruct"),
        output_dir=tmp_path,
        hardware_config=HardwareConfig(provider="local", gpu_count=1),
        server_config=InferenceServerConfig(startup_timeout=45),
    ) as realized:
        assert realized.endpoint_config.base_url == "http://localhost:30000/v1"
        assert fake_engine.launch_called is True
        assert fake_engine.tailer_called is True
        assert fake_engine.ready_waits == [45]

    assert fake_engine.shutdown_called is True


@pytest.mark.trio
async def test_run_with_sglang_provision_prefers_topology_actor_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    topology = WorkerTopologyConfig(
        hardware=HardwareConfig(provider="local", gpu_count=1),
        inference_workers=(
            InferenceWorkerConfig(
                worker_id="actor",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                inference=InferenceConfig(port=32000, startup_timeout=33.0),
            ),
        ),
        role_bindings=(InferenceRoleBinding(role="actor", worker_id="actor"),),
    )
    config_module = type("ConfigModule", (), {"worker_topology": topology})
    output_config = type("OutputConfig", (), {"output_dir": tmp_path})
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def fake_realize(**kwargs: object) -> AsyncIterator[object]:
        captured.update(kwargs)
        yield type(
            "Realized",
            (),
            {
                "endpoint_config": EndpointConfig(
                    provider="sglang",
                    model="Qwen/Qwen2.5-0.5B-Instruct",
                    base_url="http://localhost:32000/v1",
                )
            },
        )()

    async def fake_run_with_api(
        config_module: object,
        endpoint_config: EndpointConfig,
        run_config: object,
        output_config: object,
        cancel_scope: object | None = None,
    ) -> dict[str, object]:
        del config_module, run_config, output_config, cancel_scope
        return {"base_url": endpoint_config.base_url}

    monkeypatch.setattr("rollouts.eval.run.realize_worker_backed_endpoint", fake_realize)
    monkeypatch.setattr("rollouts.eval.run.run_with_api", fake_run_with_api)

    result = await run_with_sglang_provision(
        config_module=config_module,
        endpoint_config=EndpointConfig(provider="sglang", model="Qwen/Qwen2.5-0.5B-Instruct"),
        run_config=object(),
        output_config=output_config,
        hardware_config=topology.hardware,
        server_config=InferenceServerConfig(),
    )

    assert result == {"base_url": "http://localhost:32000/v1"}
    assert captured["worker"] == topology.get_worker_for_role("actor")


@pytest.mark.trio
async def test_modal_endpoint_exposes_port_at_sandbox_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class _FakeTunnel:
        url = "https://endpoint.modal.run"

    class _FakeSandbox:
        object_id = "sb-test"

        def tunnels(self, timeout: int = 5) -> dict[int, _FakeTunnel]:
            del timeout
            return {30000: _FakeTunnel()}

    class _FakeService:
        async def logs(self, tail: int = 120) -> str:
            del tail
            return "== stdout ==\nApplication startup complete.\n== stderr ==\n"

        async def is_healthy(self) -> bool:
            return True

        async def is_running(self) -> bool:
            return True

    class _FakeSession:
        def __init__(self, *, sandbox_handle: object, local_root: Path) -> None:
            captured["sandbox_handle"] = sandbox_handle
            captured["local_root"] = local_root

        async def materialize(self, _spec: object) -> object:
            return SimpleNamespace(root="/workspace/research/rollouts")

        async def serve_service(self, _service_spec: object, **_kwargs: object) -> _FakeService:
            return _FakeService()

        async def start_process(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("Modal eval endpoint should not start an in-sandbox forwarder")

        async def exec(self, _command: str) -> object:
            return SimpleNamespace(stdout="", stderr="", exit_code=0)

    async def fake_create_modal_sandbox(request: object) -> object:
        captured["request"] = request
        return SimpleNamespace(sandbox=_FakeSandbox(), sandbox_id="sb-test")

    async def fake_terminate_modal_sandbox(_handle: object) -> None:
        captured["terminated"] = True

    async def fake_refresh_modal_parent_lease(_sandbox: object) -> None:
        captured["lease_initialized"] = True

    async def fake_maintain_modal_parent_lease(_sandbox: object, _emit: object) -> None:
        await trio.sleep_forever()

    @asynccontextmanager
    async def fake_open_loop() -> AsyncIterator[None]:
        yield

    @contextmanager
    def fake_enable_output() -> AsyncIterator[None]:
        yield

    monkeypatch.setattr("bifrost.modal_backend.create_modal_sandbox", fake_create_modal_sandbox)
    monkeypatch.setattr(
        "bifrost.modal_backend.terminate_modal_sandbox", fake_terminate_modal_sandbox
    )
    monkeypatch.setattr("bifrost.modal_backend.ModalExecutionSession", _FakeSession)
    monkeypatch.setattr(
        "bifrost.modal_backend._refresh_modal_parent_lease", fake_refresh_modal_parent_lease
    )
    monkeypatch.setattr(
        "bifrost.modal_backend._maintain_modal_parent_lease",
        fake_maintain_modal_parent_lease,
    )
    monkeypatch.setattr("modal.enable_output", fake_enable_output)
    monkeypatch.setattr("trio_asyncio.open_loop", fake_open_loop)

    async with realize_worker_backed_endpoint(
        endpoint_config=EndpointConfig(provider="sglang", model="Qwen/Qwen2.5-0.5B-Instruct"),
        output_dir=tmp_path,
        hardware_config=HardwareConfig(provider="modal", gpu_count=1, deps=DepsConfig()),
        server_config=None,
        worker=InferenceWorkerConfig(
            worker_id="actor",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            inference=InferenceConfig(port=30000, startup_timeout=5.0),
        ),
        run_name="eval-modal-test",
    ) as realized:
        assert realized.endpoint_config.base_url == "https://endpoint.modal.run/v1"

    request = captured["request"]
    assert request.encrypted_ports == (30000,)
    assert captured["lease_initialized"] is True
    assert captured["terminated"] is True
