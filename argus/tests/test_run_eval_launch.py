from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from argus import run as argus_run
from rollouts.launch_plan import (
    LocalInProcessLaunchPlan,
    LocalSubprocessLaunchPlan,
    build_local_workload_plan,
    resolve_workload_kind,
)


def test_resolve_workload_kind_detects_evaluation() -> None:
    module = SimpleNamespace(
        tasks=[{"id": "sample-1"}],
        prepare_messages=lambda row: [],
        scorer=object(),
    )

    kind = resolve_workload_kind(module, Path("configs/eval.py"))

    assert kind == "evaluation"


def test_build_local_workload_plan_returns_eval_subprocess_plan(tmp_path: Path) -> None:
    config_path = tmp_path / "eval_config.py"
    config_path.write_text(
        "\n".join([
            "from rollouts.eval import AgentRunSpec, EndpointConfig",
            "tasks = [{'id': 'sample-1'}]",
            "run_spec = AgentRunSpec(endpoint=EndpointConfig(), prepare_messages=lambda row: [])",
            "scorer = object()",
        ])
    )
    module = argus_run.load_config_module(config_path)

    plan = build_local_workload_plan(
        config_module=module,
        config_path=config_path,
        repo_root=tmp_path,
        max_samples=5,
        force_deploy_committed=True,
        python_executable="python3",
        stream_run_events=False,
        provider="ssh",
        gpu_type="B200",
        gpu_count=1,
    )

    assert isinstance(plan, LocalSubprocessLaunchPlan)
    assert plan.journal_name == "control.jsonl"
    assert plan.kind == "evaluation"


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

    def fake_build_local_workload_plan(**kwargs: object) -> LocalSubprocessLaunchPlan:
        config_path = cast(Path, kwargs["config_path"])
        repo_root = cast(Path, kwargs["repo_root"])
        max_samples = cast(int | None, kwargs["max_samples"])
        force_deploy_committed = cast(bool, kwargs["force_deploy_committed"])
        python_executable = cast(str | None, kwargs["python_executable"])
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

    monkeypatch.setattr(argus_run, "build_local_workload_plan", fake_build_local_workload_plan)
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

    def fake_build_local_workload_plan(**kwargs: object) -> LocalSubprocessLaunchPlan:
        del kwargs["config_path"], kwargs["max_samples"], kwargs["force_deploy_committed"]
        del kwargs["python_executable"], kwargs["config_module"], kwargs["stream_run_events"]
        del kwargs["provider"], kwargs["gpu_type"], kwargs["gpu_count"]
        repo_root = cast(Path, kwargs["repo_root"])
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

    monkeypatch.setattr(argus_run, "build_local_workload_plan", fake_build_local_workload_plan)
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


def test_run_local_entrypoint_executes_rollouts_owned_training_plan(monkeypatch: object) -> None:
    captured: dict[str, object] = {}

    def fake_run() -> dict[str, object]:
        captured["ran"] = True
        return {"metrics_history": [1, 2, 3]}

    monkeypatch.setenv("ARGUS_EMIT_STARTUP_SENTINEL", "1")
    result = argus_run._run_local_entrypoint(
        LocalInProcessLaunchPlan(
            kind="training",
            emit_startup_sentinel=True,
            run=fake_run,
            render_result=None,
        )
    )

    assert captured["ran"] is True
    assert result == {"metrics_history": [1, 2, 3]}
