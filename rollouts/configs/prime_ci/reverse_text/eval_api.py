"""Prime-CI reverse-text eval config."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from examples.rl.reverse_text.base_config import SYSTEM_PROMPT, parse_reversed_text
from rollouts.config_status import import_tested
from rollouts.core import Message, Metric, Score
from rollouts.eval import EndpointConfig, EvalOutputConfig, EvalRunConfig
from rollouts.training.scoring import FunctionSampleScorer
from rollouts.training.types import AttemptRow

config_status = import_tested(
    "70bce1bf",
    "Imports cleanly and exercises the shared eval path with explicit sample scoring.",
)

endpoint = EndpointConfig(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    temperature=0.0,
    max_tokens=256,
)

run = EvalRunConfig(
    max_concurrent=4,
    max_samples=12,
    max_turns=1,
    verbose=True,
    show_progress=True,
)

output = EvalOutputConfig(
    experiment_name="prime_ci_reverse_text_eval_api",
)

tasks = [
    {"text": "hello world"},
    {"text": "prime intellect"},
    {"text": "reverse this string"},
    {"text": "attention is all you need"},
    {"text": "kernelbench is harder than reverse text"},
    {"text": "abc123xyz"},
]


def prepare_messages(sample: dict[str, str]) -> list[Message]:
    text = sample["text"]
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(
            role="user",
            content=(
                "Reverse the following text character-by-character. "
                "Put your answer in <reversed_text> tags.\n\n"
                f"Text to reverse: {text}"
            ),
        ),
    ]


def reverse_text_eval_score_fn(sample: AttemptRow) -> Score:
    expected = sample.input["text"][::-1]
    response = sample.response
    parsed = parse_reversed_text(response)

    if parsed is not None:
        normalized = parsed
    else:
        normalized = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip().strip(
            "\"'"
        )

    similarity = SequenceMatcher(None, normalized, expected).ratio()
    exact_match = normalized == expected

    return Score(
        metrics=(
            Metric("similarity", similarity, weight=1.0),
            Metric("exact_match", 1.0 if exact_match else 0.0, weight=0.0),
            Metric("has_tags", 1.0 if parsed is not None else 0.0, weight=0.0),
        )
    )


sample_scorer = FunctionSampleScorer(reverse_text_eval_score_fn)
