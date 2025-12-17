"""GSM8K RL training config.

Single-turn math reasoning with GRPO.
Model outputs answer in \\boxed{} format.

Based on:
- Miles GSM8K recipe: n_samples_per_prompt=8, lr=1e-6, temp=0.8
- Prime-RL patterns: group-wise advantage normalization
"""

from __future__ import annotations

import re
from typing import Any

from rollouts.dtypes import Metric, Score
from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.grpo import GRPOConfig, grpo_train

# ──────────────────────── Dataset Loading ───────────────────────────────────


SYSTEM_PROMPT = """\
Solve the following math problem step by step.
Show your reasoning clearly, then put your final numerical answer in \\boxed{}.

Example format:
Step 1: ...
Step 2: ...
Therefore, the answer is \\boxed{42}
"""


def load_gsm8k_prompts(
    max_samples: int | None = None,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Load GSM8K samples and format as chat prompts."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)

    prompts = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        # Extract final answer from solution (format: "#### 42")
        solution = row["answer"]
        match = re.search(r"####\s*([\d,.-]+)", solution)
        answer = match.group(1).replace(",", "") if match else ""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]
        prompts.append({
            "messages": messages,
            "answer": answer,
        })

    return prompts


# ──────────────────────── Score Function ────────────────────────────────────


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if match:
        answer = match.group(1).strip()
        answer = answer.replace(",", "").replace("$", "").strip()
        return answer
    return None


def normalize_answer(answer: str) -> float | None:
    """Normalize answer string to float."""
    if not answer:
        return None
    try:
        if "/" in answer:
            parts = answer.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        if "%" in answer:
            return float(answer.replace("%", "")) / 100
        return float(answer)
    except ValueError:
        return None


def gsm8k_score_fn(sample: Any) -> Score:
    """Score function for GSM8K.

    Returns Score with reward=1.0 if correct, 0.0 otherwise.
    """
    ground_truth = sample.metadata.get("answer")
    if ground_truth is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    response = sample.response if hasattr(sample, "response") else ""
    predicted = extract_boxed_answer(response)

    if predicted is None:
        return Score(
            metrics=(
                Metric("correct", 0.0, weight=1.0),
                Metric("parse_failed", 1.0, weight=0.0),
            )
        )

    pred_val = normalize_answer(predicted)
    true_val = normalize_answer(ground_truth)

    if pred_val is None or true_val is None:
        return Score(
            metrics=(
                Metric("correct", 0.0, weight=1.0),
                Metric("parse_failed", 1.0, weight=0.0),
            )
        )

    is_correct = abs(pred_val - true_val) < 0.01
    return Score(metrics=(Metric("correct", 1.0 if is_correct else 0.0, weight=1.0),))


# ──────────────────────── Training ──────────────────────────────────────────


def train(
    config: GRPOConfig | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run GSM8K RL training.

    Args:
        config: Training config. If None, uses defaults tuned for GSM8K.
        max_samples: Limit dataset size (useful for testing).

    Returns:
        Dict with metrics_history.
    """
    if config is None:
        config = GRPOConfig(
            experiment_name="gsm8k_grpo",
            lr=1e-6,
            n_samples_per_prompt=8,
            temperature=0.8,
        )

    prompts = load_gsm8k_prompts(max_samples=max_samples)

    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=gsm8k_score_fn,
        environment_cls=BasicEnvironment,
    )
