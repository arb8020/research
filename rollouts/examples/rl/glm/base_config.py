"""GLM RL training base config.

Simple single-turn task for testing GLM-4.7-Flash training.
Uses the same reverse text task as the Qwen examples.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from rollouts.dtypes import Metric, Score
from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.grpo import GRPOConfig, grpo_train

# ──────────────────────── System Prompt ─────────────────────────────────────

SYSTEM_PROMPT = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."


# ──────────────────────── Dataset Loading ───────────────────────────────────


def load_reverse_text_prompts(
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL",
    split: str = "train",
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load prompts from Prime's Reverse-Text-RL dataset."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts = []
    for row in dataset:  # type: ignore[not-subscriptable]
        text = row["prompt"]  # type: ignore[index]
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
    """Extract content from <reversed_text> tags."""
    match = re.search(r"<reversed_text>\s*(.*?)\s*</reversed_text>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"<final>\s*(.*?)\s*</final>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


# ──────────────────────── Score Function ────────────────────────────────────


def reverse_text_score_fn(sample: Any) -> Score:
    """Score function for reverse text."""
    expected = sample.metadata.get("reversed", "")
    response = sample.response if hasattr(sample, "response") else ""

    parsed = parse_reversed_text(response)

    if parsed is not None:
        response = parsed
    else:
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = response.strip().strip("\"'")

    similarity = SequenceMatcher(None, response, expected).ratio()
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
    """Run GLM RL training."""
    if config is None:
        from rollouts.training.grpo import (
            CheckpointConfig,
            GRPOOutputConfig,
            InferenceConfig,
            ModelConfig,
            RolloutConfig,
            TrainerConfig,
        )

        config = GRPOConfig(
            output=GRPOOutputConfig(experiment_name="glm_grpo"),
            model=ModelConfig(name="zai-org/GLM-4.7-Flash"),
            trainer=TrainerConfig(
                backend="torchtitan",
                torchtitan_model="glm",
                torchtitan_model_size="4.7-flash",
                lr=1e-6,
            ),
            rollout=RolloutConfig(
                batch_size=4,
                n_samples_per_prompt=8,
                temperature=0.7,
                max_seq_len=2048,
                max_tokens=128,
            ),
            inference=InferenceConfig(
                mem_fraction=0.6,
            ),
            checkpoint=CheckpointConfig(num_steps=20),
        )

    prompts = load_reverse_text_prompts(max_samples=num_samples)

    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=reverse_text_score_fn,
        environment_cls=BasicEnvironment,
    )
