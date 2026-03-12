"""Shared helpers for training_architecture smoke configs.

These configs aim to be runnable without external datasets: we embed a small,
synthetic "reverse text" prompt set and reuse the existing verifier logic.
"""

from __future__ import annotations

from typing import Any

from examples.rl.reverse_text.base_config import SYSTEM_PROMPT, reverse_text_score_fn
from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.grpo import GRPOConfig, grpo_train
from rollouts.training.scoring import FunctionSampleScorer


def make_synthetic_reverse_text_prompts(max_samples: int = 128) -> list[dict[str, Any]]:
    """Create a small fixed prompt set (no HF dataset dependency)."""
    texts = [
        "hello world",
        "the quick brown fox",
        "openai",
        "pipelines are fun",
        "fsdp",
        "mixture of experts",
        "rollouts training",
        "distributed systems",
        "abcdefghijk",
        "12345",
        "reverse me please",
        "a man a plan a canal panama",
    ]

    prompts: list[dict[str, Any]] = []
    i = 0
    while len(prompts) < max_samples:
        text = texts[i % len(texts)]
        i += 1
        prompts.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "reversed": text[::-1],
        })

    return prompts


def train(config: GRPOConfig, max_samples: int = 128) -> dict[str, Any]:
    prompts = make_synthetic_reverse_text_prompts(max_samples=max_samples)
    return grpo_train(
        config=config,
        prompts=prompts,
        sample_scorer=FunctionSampleScorer(reverse_text_score_fn),
        environment_cls=BasicEnvironment,
    )
