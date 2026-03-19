"""Template rollouts eval config for reasoning-theater experiments.

This file is intentionally incomplete. Wire in a real backend before running.

Run:
    python -m rollouts.eval.run --config dev/reasoning-theater/eval_rollouts.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

ROLLOUTS_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "rollouts"
if str(ROLLOUTS_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROLLOUTS_PROJECT_ROOT))

from attempt_executor import build_attempt_executor
from scorers import ReasoningTheaterScorer
from rollouts.training.types import AttemptRow


def _load_tasks_from_env() -> list[dict[str, Any]]:
    tasks_path = os.getenv("REASONING_THEATER_TASKS")
    if not tasks_path:
        return []
    path = Path(tasks_path)
    tasks: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


class _UnconfiguredBackend:
    async def run_primary_attempt(
        self,
        sample_data: dict[str, Any],
        sample_id: str,
        environment: Any | None,
        run_config: Any,
    ) -> AttemptRow:
        del sample_data, sample_id, environment, run_config
        raise NotImplementedError(
            "Wire in a real backend that returns AttemptRow with trajectory/completions"
        )


tasks = _load_tasks_from_env()
endpoint = None
attempt_executor = build_attempt_executor(
    backend=_UnconfiguredBackend(),
    prefix_fractions=[0.1, 0.25, 0.5, 0.75, 1.0],
)
sample_scorer = ReasoningTheaterScorer()
