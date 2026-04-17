"""Full: all 3 scenarios, 15 turns each."""

from __future__ import annotations

import json
from pathlib import Path

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig

TASKS_PATH = Path(__file__).parent.parent / "tasks.jsonl"

endpoint = EndpointConfig(
    model="claude-haiku-4-5-20251001",
    provider="anthropic",
)

run = RunConfig(
    max_turns=15,
    max_concurrent=1,
    limit=3,
    verbose=True,
    show_progress=True,
)

output = OutputConfig(experiment_name="sillytavern_text_full")


def _load_tasks() -> list[dict]:
    return [json.loads(line) for line in TASKS_PATH.read_text().splitlines() if line.strip()]


tasks_override = _load_tasks()
