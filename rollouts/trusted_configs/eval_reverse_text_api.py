"""Trusted reverse-text eval config.

Small canonical eval that exercises the shared eval path with the explicit
sample-scorer stage.
"""

from __future__ import annotations

from examples.rl.reverse_text.base_config import SYSTEM_PROMPT, reverse_text_score_fn
from rollouts.core import Message
from rollouts.eval import EndpointConfig, EvalOutputConfig, EvalRunConfig
from rollouts.training.scoring import FunctionSampleScorer

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
    experiment_name="trusted_eval_reverse_text_api",
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
        Message(role="user", content=text),
    ]


sample_scorer = FunctionSampleScorer(reverse_text_score_fn)
