from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from argus.event_log import stream_run_event_sinks


@dataclass(frozen=True)
class LocalSubprocessLaunchPlan:
    """Rollouts-owned local subprocess launch plan.

    Argus should own the generic run shell and subprocess supervision.
    Rollouts owns which local entrypoint to run, where artifacts live, and
    which journal is canonical for that workload.
    """

    run_name: str
    run_dir: Path
    journal_name: str
    command: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class LocalInProcessLaunchPlan:
    """Rollouts-owned in-process entrypoint plan."""

    kind: str
    emit_startup_sentinel: bool
    run: Callable[[], Any]
    render_result: Callable[[Any], str | None] | None = None


def build_local_eval_launch_plan(
    *,
    config_path: Path,
    consumer_project_root: Path,
    max_samples: int | None,
    force_deploy_committed: bool,
    python_executable: str | None = None,
) -> LocalSubprocessLaunchPlan:
    """Build the local eval subprocess plan.

    Rollouts owns this boundary:
    - local eval runs live under `<consumer_project_root>/results/eval/run_<timestamp>`
    - the supervisor entrypoint is `rollouts.eval.supervisor`
    - the control-plane journal for eval is `control.jsonl`
    """

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"
    run_dir = consumer_project_root / "results" / "eval" / run_name
    command = [
        python_executable or sys.executable,
        "-m",
        "rollouts.eval.supervisor",
        "--config",
        str(config_path),
        "--output-dir",
        str(run_dir),
    ]
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    if force_deploy_committed:
        command.append("--force-deploy-committed")
    return LocalSubprocessLaunchPlan(
        run_name=run_name,
        run_dir=run_dir,
        journal_name="control.jsonl",
        command=tuple(command),
        kind="evaluation",
    )


def build_local_serving_launch_plan(
    *,
    config_path: Path,
    consumer_project_root: Path,
    force_deploy_committed: bool,
    python_executable: str | None = None,
) -> LocalSubprocessLaunchPlan:
    """Build the local serving subprocess plan."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"
    run_dir = consumer_project_root / "results" / "serving" / run_name
    command = [
        python_executable or sys.executable,
        "-m",
        "rollouts.serving.supervisor",
        "--config",
        str(config_path),
        "--output-dir",
        str(run_dir),
    ]
    if force_deploy_committed:
        command.append("--force-deploy-committed")
    return LocalSubprocessLaunchPlan(
        run_name=run_name,
        run_dir=run_dir,
        journal_name="control.jsonl",
        command=tuple(command),
        kind="serving",
    )


def build_local_training_launch_plan(
    *,
    config_module: Any,
    max_samples: int | None,
    stream_run_events: bool,
) -> LocalInProcessLaunchPlan:
    """Build the direct training entrypoint plan."""

    kwargs: dict[str, Any] = {}
    if stream_run_events:
        kwargs["run_logger"] = stream_run_event_sinks()
    if max_samples is not None:
        kwargs["max_samples"] = max_samples

    def _run() -> Any:
        return config_module.train(config=config_module.config, **kwargs)

    return LocalInProcessLaunchPlan(
        kind="training",
        emit_startup_sentinel=True,
        run=_run,
        render_result=lambda result: (
            f"Training complete. {len(result.get('metrics_history', []))} steps"
        ),
    )


def build_local_benchmark_launch_plan(
    *,
    config_module: Any,
    gpu_type: str,
    gpu_count: int,
) -> LocalInProcessLaunchPlan:
    """Build the direct benchmark entrypoint plan."""

    def _run() -> Any:
        import trio

        from rollouts.inference.benchmark.runner import run_benchmark_local

        return trio.run(
            run_benchmark_local,
            config_module.config,
            gpu_type,
            gpu_count,
        )

    return LocalInProcessLaunchPlan(
        kind="benchmark",
        emit_startup_sentinel=True,
        run=_run,
        render_result=lambda result: json.dumps(result.to_dict(), indent=2),
    )


def resolve_workload_kind(config_module: Any, config_path: Path) -> str:
    """Resolve the Rollouts workload kind for one config module."""
    from rollouts.config_contracts import (
        validate_eval_config_module,
        validate_serving_config_module,
        validate_train_config_module,
    )

    train_error: ValueError | None = None
    eval_error: ValueError | None = None
    serving_error: ValueError | None = None

    try:
        validate_train_config_module(config_module, config_path)
    except ValueError as exc:
        train_error = exc
    else:
        # Lazy import: rollouts.inference.benchmark.config → rollouts.inference
        # (package __init__) → rollouts.inference.core does `import torch` at
        # module load. Dispatching a serving or eval config shouldn't require
        # torch in the local venv, so this stays scoped to the training branch
        # where BenchmarkConfig is actually needed.
        from rollouts.inference.benchmark.config import BenchmarkConfig  # noqa: PLC0415

        if isinstance(config_module.config, BenchmarkConfig):
            return "benchmark"
        return "training"

    try:
        validate_serving_config_module(config_module, config_path)
        return "serving"
    except ValueError as exc:
        serving_error = exc

    try:
        validate_eval_config_module(config_module, config_path)
        return "evaluation"
    except ValueError as exc:
        eval_error = exc

    raise ValueError(
        f"Config {config_path} is neither a valid training config, eval config, nor serving config.\n"
        f"Training contract error: {train_error}\n"
        f"Eval contract error: {eval_error}\n"
        f"Serving contract error: {serving_error}"
    )


def build_local_workload_plan(
    *,
    config_module: Any,
    config_path: Path,
    consumer_project_root: Path,
    max_samples: int | None,
    force_deploy_committed: bool,
    python_executable: str | None,
    stream_run_events: bool,
    provider: str,
    gpu_type: str,
    gpu_count: int,
) -> LocalSubprocessLaunchPlan | LocalInProcessLaunchPlan | None:
    """Build one Rollouts-owned local workload plan when the launch shape is local."""

    workload_kind = resolve_workload_kind(config_module, config_path)
    if workload_kind == "evaluation":
        return build_local_eval_launch_plan(
            config_path=config_path,
            consumer_project_root=consumer_project_root,
            max_samples=max_samples,
            force_deploy_committed=force_deploy_committed,
            python_executable=python_executable,
        )
    if workload_kind == "serving":
        return build_local_serving_launch_plan(
            config_path=config_path,
            consumer_project_root=consumer_project_root,
            force_deploy_committed=force_deploy_committed,
            python_executable=python_executable,
        )
    if provider != "local":
        return None
    if workload_kind == "benchmark":
        return build_local_benchmark_launch_plan(
            config_module=config_module,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
        )
    return build_local_training_launch_plan(
        config_module=config_module,
        max_samples=max_samples,
        stream_run_events=stream_run_events,
    )
