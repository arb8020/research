from __future__ import annotations

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


def build_local_eval_launch_plan(
    *,
    config_path: Path,
    repo_root: Path,
    max_samples: int | None,
    force_deploy_committed: bool,
    python_executable: str | None = None,
) -> LocalSubprocessLaunchPlan:
    """Build the local eval subprocess plan.

    Rollouts owns this boundary:
    - local eval runs live under `results/eval/run_<timestamp>`
    - the supervisor entrypoint is `rollouts.eval.supervisor`
    - the control-plane journal for eval is `control.jsonl`
    """

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"
    run_dir = repo_root / "results" / "eval" / run_name
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
    )
