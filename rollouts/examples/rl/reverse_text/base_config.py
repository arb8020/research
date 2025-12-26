"""Reverse Text RL training config.

Simple single-turn task: reverse a string.
Based on Prime-RL's reverse-text example.
"""

from __future__ import annotations

import random
import string
from difflib import SequenceMatcher
from typing import Any

from rollouts.dtypes import Metric, Score
from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.grpo import GRPOConfig, grpo_train

# ──────────────────────── Dataset Generation ────────────────────────────────


SYSTEM_PROMPT = """\
You are a text reversal assistant. When given text, reverse it character by character.

Example:
Input: hello world
Output: dlrow olleh

Just output the reversed text, nothing else.
"""


def _generate_random_text(length: int, rng: random.Random) -> str:
    """Generate random text to reverse."""
    words = ["hello", "world", "python", "code", "test", "data", "model", "train"]
    text = []
    while len(" ".join(text)) < length:
        if rng.random() < 0.7:
            text.append(rng.choice(words))
        else:
            text.append("".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8))))
    return " ".join(text)[:length]


def generate_reverse_text_prompts(
    num_samples: int = 1000,
    min_length: int = 10,
    max_length: int = 50,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate reverse text prompts."""
    rng = random.Random(seed)
    prompts = []

    for _ in range(num_samples):
        length = rng.randint(min_length, max_length)
        text = _generate_random_text(length, rng)
        reversed_text = text[::-1]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompts.append({
            "messages": messages,
            "reversed": reversed_text,
        })

    return prompts


# ──────────────────────── Score Function ────────────────────────────────────


def reverse_text_score_fn(sample: Any) -> Score:
    """Score function for reverse text.

    Returns similarity score between prediction and expected reversal.
    """
    expected = sample.metadata.get("reversed", "")
    response = sample.response if hasattr(sample, "response") else ""

    # Clean up response (remove quotes, extra whitespace)
    response = response.strip().strip("\"'")

    similarity = SequenceMatcher(None, response, expected).ratio()

    return Score(
        metrics=(
            Metric("similarity", similarity, weight=1.0),
            Metric("exact_match", 1.0 if response == expected else 0.0, weight=0.0),
        )
    )


# ──────────────────────── Training ──────────────────────────────────────────


def train(
    config: GRPOConfig | None = None,
    num_samples: int = 1000,
) -> dict[str, Any]:
    """Run reverse text RL training.

    Args:
        config: Training config. If None, uses defaults from Prime-RL.
        num_samples: Number of prompts to generate.

    Returns:
        Dict with metrics_history.
    """
    if config is None:
        config = GRPOConfig(
            experiment_name="reverse_text_grpo",
            lr=3e-6,  # Prime-RL uses 3e-6
            n_samples_per_prompt=16,  # Prime-RL uses 16
            temperature=0.7,
            num_steps=20,  # Prime-RL uses 20 steps
            max_seq_len=256,
            max_tokens=128,
        )

    prompts = generate_reverse_text_prompts(num_samples=num_samples)

    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=reverse_text_score_fn,
        environment_cls=BasicEnvironment,
    )
