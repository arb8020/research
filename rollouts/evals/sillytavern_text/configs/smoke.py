"""Smoke: 1 scenario, 8 turns, Haiku both sides."""

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
    max_turns=8,
    max_concurrent=1,
    limit=1,
    verbose=True,
    show_progress=True,
)

output = OutputConfig(experiment_name="sillytavern_text_smoke")


def _load_tasks() -> list[dict]:
    return [json.loads(line) for line in TASKS_PATH.read_text().splitlines() if line.strip()]


# run_eval.py picks up `tasks_override` and passes it through to the runner.
# Limit to the first scenario (RWBY/Ruby — the card with an embedded book
# that we strip on the fly via prepare.py's pairing contract).
tasks_override = _load_tasks()[:1]
