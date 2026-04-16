from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from argus import run as argus_run
from rollouts.launch_plan import LocalSubprocessLaunchPlan


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

    def fake_build_local_eval_launch_plan(
        *,
        config_path: Path,
        repo_root: Path,
        max_samples: int | None,
        force_deploy_committed: bool,
        python_executable: str | None = None,
    ) -> LocalSubprocessLaunchPlan:
        output_dir = repo_root / "results" / "eval" / "run_test"
        captured["config_path"] = config_path
        captured["output_dir"] = output_dir
        captured["max_samples"] = max_samples
        captured["force_deploy_committed"] = force_deploy_committed
        return LocalSubprocessLaunchPlan(
            run_name="run_test",
            run_dir=output_dir,
            journal_name="control.jsonl",
            command=(python_executable or "python", "-m", "rollouts.eval.supervisor"),
            kind="evaluation",
        )

    def fake_spawn_local_subprocess(*, plan: LocalSubprocessLaunchPlan, log: object) -> int:
        captured["spawn_plan"] = plan
        return 4242

    monkeypatch.setattr(
        argus_run, "build_local_eval_launch_plan", fake_build_local_eval_launch_plan
    )
    monkeypatch.setattr(argus_run, "_spawn_local_subprocess", fake_spawn_local_subprocess)
    monkeypatch.setattr(
        argus_run, "_launch_eval_monitor", lambda *, run_dir, tail, fmt="pretty": 99
    )
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

    def fake_build_local_eval_launch_plan(
        *,
        config_path: Path,
        repo_root: Path,
        max_samples: int | None,
        force_deploy_committed: bool,
        python_executable: str | None = None,
    ) -> LocalSubprocessLaunchPlan:
        del config_path, max_samples, force_deploy_committed, python_executable
        output_dir = repo_root / "results" / "eval" / "run_test"
        captured["output_dir"] = output_dir
        return LocalSubprocessLaunchPlan(
            run_name="run_test",
            run_dir=output_dir,
            journal_name="control.jsonl",
            command=("python", "-m", "rollouts.eval.supervisor"),
            kind="evaluation",
        )

    def fake_spawn_local_subprocess(*, plan: LocalSubprocessLaunchPlan, log: object) -> int:
        del plan, log
        return 4242

    def fake_launch_eval_monitor(*, run_dir: Path, tail: bool, fmt: str = "pretty") -> int:
        captured["monitor_run_dir"] = run_dir
        captured["monitor_tail"] = tail
        captured["monitor_fmt"] = fmt
        return 17

    monkeypatch.setattr(
        argus_run, "build_local_eval_launch_plan", fake_build_local_eval_launch_plan
    )
    monkeypatch.setattr(argus_run, "_spawn_local_subprocess", fake_spawn_local_subprocess)
    monkeypatch.setattr(argus_run, "_launch_eval_monitor", fake_launch_eval_monitor)
    monkeypatch.setattr(argus_run, "_write_launch_record", lambda payload: tmp_path / "launch.json")
    monkeypatch.setattr(argus_run, "_remove_launch_record", lambda path: None)
    monkeypatch.setattr(argus_run, "_active_launches", lambda: [])

    result = argus_run.main(["--config", str(config_path), "--tui"])

    assert result == 17
    assert captured["monitor_run_dir"] == captured["output_dir"]
    assert captured["monitor_tail"] is False
    assert captured["monitor_fmt"] == "pretty"


def test_spawn_local_subprocess_sets_rollouts_output_dir(
    monkeypatch: object, tmp_path: Path
) -> None:
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

    pid = argus_run._spawn_local_subprocess(
        plan=LocalSubprocessLaunchPlan(
            run_name="run_123",
            run_dir=output_dir,
            journal_name="control.jsonl",
            command=(os.fspath(argus_run.sys.executable), "-m", "rollouts.eval.supervisor"),
            kind="evaluation",
        ),
        log=log,  # type: ignore[arg-type]
    )

    assert pid == 4242
    command = cast(list[Any], captured["command"])
    env = cast(dict[str, str], captured["env"])
    assert command[:3] == [
        os.fspath(argus_run.sys.executable),
        "-m",
        "rollouts.eval.supervisor",
    ]
    assert env["ROLLOUTS_OUTPUT_DIR"] == os.fspath(output_dir)
