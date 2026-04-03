from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from rollouts.eval.configs import EndpointConfig, InferenceServerConfig
from rollouts.eval.endpoint_realization import (
    _emit_log_lines,
    _legacy_worker_from_eval_surface,
    _remote_inference_python,
    _remote_service_spec,
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
    assert readiness_target == ":30000/health"
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
