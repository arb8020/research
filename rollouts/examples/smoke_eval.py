"""Smoke test for EvalSpec + run_eval_from_spec pipeline.

Runs 2 samples: asks Claude to add numbers, scores by checking the answer.
Tests the full config-tiers → EvalSpec → runner → evaluate() pipeline.

Usage:
    cd rollouts && uv run python examples/smoke_eval.py
"""

from __future__ import annotations

from typing import Any

from rollouts.config.tiers import EndpointConfig, OutputConfig, RunConfig
from rollouts.core import Message, Metric, Score
from rollouts.eval_runner import EvalSpec, run_eval_from_spec
from rollouts.training.scoring import FunctionScorer
from rollouts.training.types import RowAttempt

# ── Score function ──


def score_addition(sample: RowAttempt, _context: object) -> Score:
    """Check if the model's response contains the correct sum."""
    sample_data = sample.trajectory.metadata.get("sample_data", {}) if sample.trajectory else {}
    expected = sample_data.get("expected", "")
    last_msg = ""
    if sample.trajectory and sample.trajectory.messages:
        last_msg = sample.trajectory.messages[-1].content or ""
    correct = str(expected) in last_msg
    return Score(
        metrics=(Metric(name="correct", value=1.0 if correct else 0.0, weight=1.0),),
    )


# ── Prepare messages ──


def prepare_messages(sample_data: dict[str, Any]) -> list[Message]:
    """Turn a dataset row into messages."""
    return [
        Message(role="user", content=sample_data["prompt"]),
    ]


# ── Dataset ──

tasks = [
    {"id": "add_1", "prompt": "What is 7 + 13? Reply with just the number.", "expected": 20},
    {"id": "add_2", "prompt": "What is 99 + 1? Reply with just the number.", "expected": 100},
]


# ── Eval spec ──

spec = EvalSpec(
    name="smoke_addition",
    prepare_messages=prepare_messages,
    scorer=FunctionScorer(score_addition),
)

if __name__ == "__main__":
    # Requires ANTHROPIC_API_KEY in environment
    result = run_eval_from_spec(
        spec,
        tasks=tasks,
        endpoint=EndpointConfig(model="claude-sonnet-4-20250514", temperature=0.0),
        run=RunConfig(max_concurrent=2, max_turns=1),
        output=OutputConfig(experiment_name="smoke_addition"),
    )
    print(f"\nResult: {result}")
