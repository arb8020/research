from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from argus import eval_supervisor
from rollouts.eval.configs import EndpointConfig, EvalOutputConfig, EvalRunConfig, EvalTaskSpec


def test_eval_supervisor_passes_realized_base_url_to_eval_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "eval_config.py"
    output_dir = tmp_path / "results" / "eval" / "run_123"
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def fake_realize_worker_backed_endpoint(**kwargs: object):
        captured["realize_kwargs"] = kwargs
        yield SimpleNamespace(
            endpoint_config=EndpointConfig(
                provider="sglang",
                model="Qwen/Qwen2.5-7B-Instruct",
                base_url="https://example.test/v1",
            )
        )

    def fake_load_config_module(path: Path) -> object:
        assert path == config_path
        return object()

    def fake_resolve_eval_task_spec(module: object) -> EvalTaskSpec:
        del module
        return EvalTaskSpec(
            tasks=[{"id": "sample-1"}],
            run_spec=SimpleNamespace(
                endpoint=EndpointConfig(
                    provider="sglang",
                    model="Qwen/Qwen2.5-7B-Instruct",
                ),
                attempt_executor=None,
            ),
            scorer=object(),
            run=EvalRunConfig(),
            output=EvalOutputConfig(),
            hardware=SimpleNamespace(provider="modal"),
            server=object(),
        )

    class _FakeProc:
        def wait(self) -> int:
            return 0

    def fake_popen(command: list[str], *, cwd: str, env: dict[str, str]) -> _FakeProc:
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        return _FakeProc()

    monkeypatch.setattr(eval_supervisor, "load_config_module", fake_load_config_module)
    monkeypatch.setattr(eval_supervisor, "resolve_eval_task_spec", fake_resolve_eval_task_spec)
    monkeypatch.setattr(
        eval_supervisor,
        "realize_worker_backed_endpoint",
        fake_realize_worker_backed_endpoint,
    )
    monkeypatch.setattr(eval_supervisor.subprocess, "Popen", fake_popen)

    result = eval_supervisor.main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert captured["command"][:3] == [os.fspath(eval_supervisor.sys.executable), "-m", "rollouts.eval.run"]
    assert captured["env"]["ROLLOUTS_OUTPUT_DIR"] == os.fspath(output_dir.resolve())
    assert captured["env"]["ROLLOUTS_ENDPOINT_BASE_URL"] == "https://example.test/v1"
    assert captured["realize_kwargs"]["run_logger"] is not None
