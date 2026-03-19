"""Prime-CI reverse-text eval via OpenHands headless CLI."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from examples.rl.reverse_text.base_config import SYSTEM_PROMPT, parse_reversed_text
from rollouts.config_status import import_tested
from rollouts.core import Metric, Score
from rollouts.eval import (
    AgentRunSpec,
    EvalOutputConfig,
    EvalRunConfig,
    MaxTurnsStop,
    make_external_attempt_executor,
)
from rollouts.training.scoring import FunctionScorer
from rollouts.training.types import AttemptResult

tasks = [
    {"text": "hello world"},
]

run = EvalRunConfig(
    max_concurrent=1,
    max_samples=1,
    stop_handler=MaxTurnsStop(1),
    verbose=True,
    show_progress=True,
)


def build_prompt(sample: dict[str, str]) -> str:
    text = sample["text"]
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Reverse the following text character-by-character. "
        "Put your answer in <reversed_text> tags.\n\n"
        f"Text to reverse: {text}"
    )


def reverse_text_eval_score_fn(sample: AttemptResult, _context: object) -> Score:
    expected = sample.input["text"][::-1]
    response = sample.response
    parsed = parse_reversed_text(response)

    if parsed is not None:
        normalized = parsed
    else:
        normalized = (
            re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip().strip("\"'")
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


scorer = FunctionScorer(reverse_text_eval_score_fn)

config_status = import_tested(
    "uncommitted",
    "Imports cleanly and exercises the direct-attempt eval path with OpenHands headless CLI.",
)

run_spec = AgentRunSpec(
    attempt_executor=make_external_attempt_executor(
        "openhands",
        prompt_builder=build_prompt,
        cwd=Path.cwd(),
        model="gpt-5.1-codex-mini",
        api_key_env_var="OPENAI_API_KEY",
        timeout_seconds=300.0,
    ),
)

output = EvalOutputConfig(experiment_name="prime_ci_reverse_text_eval_openhands")
