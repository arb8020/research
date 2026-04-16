from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from rollouts.eval import supervisor as eval_supervisor
from rollouts.eval.configs import EndpointConfig, EvalOutputConfig, EvalRunConfig, EvalTaskSpec


def test_eval_supervisor_passes_realized_base_url_to_eval_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "eval_config.py"
    output_dir = tmp_path / "results" / "eval" / "run_123"
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def fake_realize_worker_backed_endpoint(
        **kwargs: object,
    ) -> AsyncIterator[SimpleNamespace]:
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

    def fake_popen(
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        stdin: object,
        stdout: object,
        stderr: object,
    ) -> _FakeProc:
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["stdin"] = stdin
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        return _FakeProc()

    monkeypatch.setattr(eval_supervisor, "load_config_module", fake_load_config_module)
    monkeypatch.setattr(eval_supervisor, "resolve_eval_task_spec", fake_resolve_eval_task_spec)
    monkeypatch.setattr(
        eval_supervisor,
        "realize_worker_backed_endpoint",
        fake_realize_worker_backed_endpoint,
    )
    monkeypatch.setattr(eval_supervisor.subprocess, "Popen", fake_popen)

    result = eval_supervisor.main([
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ])

    command = cast(list[str], captured["command"])
    env = cast(dict[str, str], captured["env"])
    realize_kwargs = cast(dict[str, object], captured["realize_kwargs"])

    assert result == 0
    assert command[:3] == [
        os.fspath(eval_supervisor.sys.executable),
        "-m",
        "rollouts.eval.run",
    ]
    assert env["ROLLOUTS_OUTPUT_DIR"] == os.fspath(output_dir.resolve())
    assert env["ROLLOUTS_ENDPOINT_BASE_URL"] == "https://example.test/v1"
    assert captured["stdin"] is eval_supervisor.subprocess.DEVNULL
    assert getattr(captured["stdout"], "name", "").endswith("/stdout.log")
    assert getattr(captured["stderr"], "name", "").endswith("/stderr.log")
    assert realize_kwargs["run_logger"] is not None


def test_eval_supervisor_projects_nonzero_child_failure_into_control_journal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "eval_config.py"
    output_dir = tmp_path / "results" / "eval" / "run_123"

    def fake_load_config_module(path: Path) -> object:
        assert path == config_path
        return object()

    def fake_resolve_eval_task_spec(module: object) -> EvalTaskSpec:
        del module
        return EvalTaskSpec(
            tasks=[{"id": "sample-1"}],
            run_spec=SimpleNamespace(
                endpoint=None,
                attempt_executor=object(),
            ),
            scorer=object(),
            run=EvalRunConfig(),
            output=EvalOutputConfig(),
        )

    class _FakeProc:
        def wait(self) -> int:
            return 1

    def fake_popen(
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        stdin: object,
        stdout: object,
        stderr: object,
    ) -> _FakeProc:
        del command, cwd, env, stdin, stdout
        stderr.write("Fatal Python error: init_sys_streams\n")
        stderr.write("OSError: [Errno 9] Bad file descriptor\n")
        stderr.flush()
        return _FakeProc()

    monkeypatch.setattr(eval_supervisor, "load_config_module", fake_load_config_module)
    monkeypatch.setattr(eval_supervisor, "resolve_eval_task_spec", fake_resolve_eval_task_spec)
    monkeypatch.setattr(eval_supervisor.subprocess, "Popen", fake_popen)

    result = eval_supervisor.main([
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ])

    assert result == 1
    control_events = [
        json.loads(line) for line in (output_dir / "control.jsonl").read_text().splitlines()
    ]
    assert control_events[-2]["event"] == "eval_child_failed"
    assert control_events[-2]["exit_code"] == 1
    assert control_events[-2]["error_summary"] == "OSError: [Errno 9] Bad file descriptor"
    assert control_events[-1] == {
        "ts": control_events[-1]["ts"],
        "event": "run_end",
        "status": "failed",
        "exit_code": 1,
    }
