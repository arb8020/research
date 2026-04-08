"""Shared helpers for cheap inference-engine smoke witnesses.

These witnesses are intentionally small and boring:
- own the inference-server lifecycle where possible
- run a fixed reverse-text workload
- dump the normal eval artifacts plus runtime telemetry

Keep the workload and scoring stable so backend comparisons stay honest even
when the launch/runtime path differs across engines.
"""

from __future__ import annotations

from collections.abc import Sequence

from rollouts.core import Message, Metric, Score
from rollouts.eval import AgentRunSpec, EvalOutputConfig, EvalRunConfig, EvalTaskSpec
from rollouts.training.scoring import FunctionScorer
from rollouts.training.types import RowAttempt

SMOKE_TASKS: list[dict[str, str]] = [
    {"text": "hello"},
    {"text": "world"},
    {"text": "attention is all you need"},
    {"text": "continuous batching"},
]


def prepare_reverse_text_messages(sample: dict[str, str]) -> list[Message]:
    text = sample["text"]
    return [
        Message(
            role="user",
            content=(
                "Reverse the following text character-by-character. "
                "Put your answer in <answer> tags.\n\n"
                f"Text: {text}"
            ),
        )
    ]


def reverse_text_score_fn(sample: RowAttempt, _context: object) -> Score:
    import re

    expected = sample.input["text"][::-1]
    response = sample.response
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    parsed = match.group(1).strip() if match else response.strip()
    exact = parsed == expected
    return Score(
        metrics=(
            Metric("exact_match", 1.0 if exact else 0.0, weight=1.0),
            Metric("has_tags", 1.0 if match else 0.0, weight=0.0),
        )
    )


def make_smoke_eval_task(
    *,
    endpoint: object,
    hardware: object | None,
    experiment_name: str,
    tasks: Sequence[dict[str, str]] = SMOKE_TASKS,
    max_concurrent: int = 2,
    max_turns: int = 1,
) -> EvalTaskSpec:
    return EvalTaskSpec(
        tasks=list(tasks),
        run_spec=AgentRunSpec(
            endpoint=endpoint,
            prepare_messages=prepare_reverse_text_messages,
        ),
        scorer=FunctionScorer(reverse_text_score_fn),
        run=EvalRunConfig(
            max_concurrent=max_concurrent,
            max_samples=len(tasks),
            max_turns=max_turns,
            verbose=True,
            show_progress=True,
        ),
        output=EvalOutputConfig(experiment_name=experiment_name),
        hardware=hardware,
    )
