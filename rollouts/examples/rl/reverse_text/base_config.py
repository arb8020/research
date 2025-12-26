"""Reverse Text RL training config.

Simple single-turn task: reverse a string.
Based on Prime-RL's reverse-text example.

Key differences from naive approach:
1. Uses Prime's actual RL dataset (PrimeIntellect/Reverse-Text-RL)
2. Uses correct system prompt with <reversed_text> tag instruction
3. Parses <reversed_text> tags from response before scoring
4. Uses LCS ratio for reward (same as Prime's verifiers)
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from rollouts.dtypes import Metric, Score
from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.grpo import GRPOConfig, grpo_train

# ──────────────────────── System Prompt ─────────────────────────────────────

# Must match what the SFT model was trained on!
SYSTEM_PROMPT = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."


# ──────────────────────── Dataset Loading ───────────────────────────────────


def load_reverse_text_prompts(
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL",
    split: str = "train",
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load prompts from Prime's Reverse-Text-RL dataset.

    This dataset contains real Wikipedia sentences (not random gibberish),
    which is what the SFT model was trained on.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split
        max_samples: Maximum number of samples to load

    Returns:
        List of prompt dicts with messages and reversed text
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts = []
    for row in dataset:
        text = row["prompt"]
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


# ──────────────────────── XML Parsing ───────────────────────────────────────


def parse_reversed_text(response: str) -> str | None:
    """Extract content from <reversed_text> tags.

    Also handles <final> tags (some model variants use this).

    Args:
        response: Full model response

    Returns:
        Extracted text or None if no tags found
    """
    # Try <reversed_text> first (what Prime uses)
    match = re.search(r"<reversed_text>\s*(.*?)\s*</reversed_text>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fall back to <final> (some model variants)
    match = re.search(r"<final>\s*(.*?)\s*</final>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


# ──────────────────────── Score Function ────────────────────────────────────


def reverse_text_score_fn(sample: Any) -> Score:
    """Score function for reverse text (matches Prime's verifiers).

    Uses LCS ratio between parsed response and expected reversal.
    Parses <reversed_text> or <final> tags from response.
    """
    expected = sample.metadata.get("reversed", "")
    response = sample.response if hasattr(sample, "response") else ""

    # Parse the response - extract from XML tags
    parsed = parse_reversed_text(response)

    if parsed is not None:
        # Got tagged response - use it
        response = parsed
    else:
        # No tags found - try to clean up raw response
        # Remove any thinking tags
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = response.strip().strip("\"'")

    # LCS ratio (same as Prime's verifiers)
    similarity = SequenceMatcher(None, response, expected).ratio()

    # Check exact match
    exact_match = response == expected

    return Score(
        metrics=(
            Metric("similarity", similarity, weight=1.0),
            Metric("exact_match", 1.0 if exact_match else 0.0, weight=0.0),
            Metric("has_tags", 1.0 if parsed is not None else 0.0, weight=0.0),
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
        num_samples: Number of prompts to load from dataset.

    Returns:
        Dict with metrics_history.
    """
    if config is None:
        config = GRPOConfig(
            experiment_name="reverse_text_grpo",
            model_name="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
            lr=3e-6,  # Prime-RL uses 3e-6
            n_samples_per_prompt=16,  # Prime-RL uses 16
            temperature=0.7,
            num_steps=20,  # Prime-RL uses 20 steps
            max_seq_len=256,
            max_tokens=128,
        )

    # Load Prime's actual RL dataset (not random gibberish!)
    prompts = load_reverse_text_prompts(max_samples=num_samples)

    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=reverse_text_score_fn,
        environment_cls=BasicEnvironment,
    )
