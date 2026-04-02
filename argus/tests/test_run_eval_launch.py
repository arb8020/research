from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from argus import run as argus_run


def test_classify_config_module_detects_evaluation() -> None:
    module = SimpleNamespace(
        tasks=[{"id": "sample-1"}],
        prepare_messages=lambda row: [],
        scorer=object(),
    )

    kind = argus_run._classify_config_module(module, Path("configs/eval.py"))

    assert kind == "evaluation"


def test_run_main_launches_eval_via_detached_subprocess(
    monkeypatch: object, tmp_path: Path
) -> None:
    config_path = tmp_path / "eval_config.py"
    config_path.write_text(
        "\n".join([
            "from rollouts.eval import AgentRunSpec, EndpointConfig",
            "tasks = [{'id': 'sample-1'}]",
            "run_spec = AgentRunSpec(endpoint=EndpointConfig(), prepare_messages=lambda row: [])",
            "scorer = object()",
        ])
    )

    captured: dict[str, object] = {}

    def fake_spawn_eval_subprocess(
        *,
        config_path: Path,
        output_dir: Path,
        max_samples: int | None,
        log: object,
        force_deploy_committed: bool = False,
    ) -> int:
        captured["config_path"] = config_path
        captured["output_dir"] = output_dir
        captured["max_samples"] = max_samples
        captured["force_deploy_committed"] = force_deploy_committed
        return 4242

    monkeypatch.setattr(argus_run, "_spawn_eval_subprocess", fake_spawn_eval_subprocess)
    monkeypatch.setattr(argus_run, "_launch_eval_monitor", lambda *, run_dir, tail: 99)
    monkeypatch.setattr(argus_run, "_write_launch_record", lambda payload: tmp_path / "launch.json")
    monkeypatch.setattr(argus_run, "_remove_launch_record", lambda path: None)
    monkeypatch.setattr(argus_run, "_active_launches", lambda: [])

    result = argus_run.main(["--config", str(config_path)])

    assert result == 0
    assert captured["config_path"] == config_path
    assert isinstance(captured["output_dir"], Path)
    assert str(captured["output_dir"]).endswith("/results/eval/" + captured["output_dir"].name)
    assert captured["max_samples"] is None
    assert captured["force_deploy_committed"] is False


def test_run_main_eval_tui_hands_off_to_monitor(monkeypatch: object, tmp_path: Path) -> None:
    config_path = tmp_path / "eval_config.py"
    config_path.write_text(
        "\n".join([
            "from rollouts.eval import AgentRunSpec, EndpointConfig",
            "tasks = [{'id': 'sample-1'}]",
            "run_spec = AgentRunSpec(endpoint=EndpointConfig(), prepare_messages=lambda row: [])",
            "scorer = object()",
        ])
    )

    captured: dict[str, object] = {}

    def fake_spawn_eval_subprocess(
        *,
        config_path: Path,
        output_dir: Path,
        max_samples: int | None,
        log: object,
        force_deploy_committed: bool = False,
    ) -> int:
        captured["output_dir"] = output_dir
        return 4242

    def fake_launch_eval_monitor(*, run_dir: Path, tail: bool) -> int:
        captured["monitor_run_dir"] = run_dir
        captured["monitor_tail"] = tail
        return 17

    monkeypatch.setattr(argus_run, "_spawn_eval_subprocess", fake_spawn_eval_subprocess)
    monkeypatch.setattr(argus_run, "_launch_eval_monitor", fake_launch_eval_monitor)
    monkeypatch.setattr(argus_run, "_write_launch_record", lambda payload: tmp_path / "launch.json")
    monkeypatch.setattr(argus_run, "_remove_launch_record", lambda path: None)
    monkeypatch.setattr(argus_run, "_active_launches", lambda: [])

    result = argus_run.main(["--config", str(config_path), "--tui"])

    assert result == 17
    assert captured["monitor_run_dir"] == captured["output_dir"]
    assert captured["monitor_tail"] is False


def test_spawn_eval_subprocess_sets_rollouts_output_dir(
    monkeypatch: object, tmp_path: Path
) -> None:
    config_path = tmp_path / "eval_config.py"
    config_path.write_text("# test config")
    output_dir = tmp_path / "results" / "eval" / "run_123"

    captured: dict[str, object] = {}

    class _FakeProc:
        pid = 4242

    def fake_popen(*args: object, **kwargs: object) -> _FakeProc:
        captured["command"] = kwargs.get("args", args[0] if args else None)
        captured["env"] = kwargs["env"]
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    def log(event: str, **data: object) -> None:
        captured["event"] = event
        captured["log"] = data

    pid = argus_run._spawn_eval_subprocess(
        config_path=config_path,
        output_dir=output_dir,
        max_samples=None,
        log=log,
    )

    assert pid == 4242
    assert captured["command"][:3] == [os.fspath(argus_run.sys.executable), "-m", "argus.eval_supervisor"]
    assert "--output-dir" in captured["command"]
    assert captured["env"]["ROLLOUTS_OUTPUT_DIR"] == os.fspath(output_dir)
