"""Calculator RL training config.

Multi-turn task with tool use (calculator environment).
Model uses add/subtract/multiply/divide tools to compute answers.

Uses the unified agent infrastructure (agent_rollout_to_sample + CalculatorEnvironment).
"""

from __future__ import annotations

import re
from typing import Any

from rollouts.dtypes import Metric, Score
from rollouts.environments.calculator import CalculatorEnvironment
from rollouts.training.grpo import GRPOConfig, grpo_train

# ──────────────────────── Dataset ───────────────────────────────────────────


def load_calculator_prompts(max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load calculator task prompts.

    Returns list of dicts with:
        - messages: List of chat messages
        - ground_truth: float (expected answer)
    """
    tasks = [
        {"prompt": "What is 5 + 3?", "ground_truth": 8.0},
        {"prompt": "What is 12 - 7?", "ground_truth": 5.0},
        {"prompt": "What is 6 * 4?", "ground_truth": 24.0},
        {"prompt": "What is 20 / 5?", "ground_truth": 4.0},
        {"prompt": "What is 15 + 7 - 3?", "ground_truth": 19.0},
        {"prompt": "What is 8 * 3 + 2?", "ground_truth": 26.0},
        {"prompt": "What is 100 / 4 - 10?", "ground_truth": 15.0},
        {"prompt": "What is 7 + 8 * 2?", "ground_truth": 23.0},
        {"prompt": "What is 50 - 25 + 10?", "ground_truth": 35.0},
        {"prompt": "What is 9 * 9?", "ground_truth": 81.0},
        {"prompt": "What is 144 / 12?", "ground_truth": 12.0},
        {"prompt": "What is 33 + 67?", "ground_truth": 100.0},
    ]

    if max_samples is not None:
        tasks = tasks[:max_samples]

    # Convert to messages format
    prompts = []
    for task in tasks:
        prompts.append({
            "messages": [{"role": "user", "content": task["prompt"]}],
            "ground_truth": task["ground_truth"],
        })

    return prompts


# ──────────────────────── Score Function ────────────────────────────────────


def calculator_score_fn(sample: Any) -> Score:
    """Score function for calculator tasks.

    Compares final_result from complete_task tool to ground_truth.
    Returns Score with reward 1.0 if correct, 0.0 otherwise.
    """
    ground_truth = sample.metadata.get("ground_truth")
    if ground_truth is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    # Extract final_result from response
    final_result = None
    response = sample.response if hasattr(sample, "response") else ""

    # Look for "Final result: X" pattern
    match = re.search(r"Final result:\s*([\d.-]+)", response)
    if match:
        try:
            final_result = float(match.group(1))
        except ValueError:
            pass

    if final_result is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    # Check if correct (with tolerance)
    is_correct = abs(final_result - ground_truth) < 0.01
    reward = 1.0 if is_correct else 0.0

    return Score(
        metrics=(
            Metric("correct", reward, weight=1.0),
            Metric("final_result", final_result, weight=0.0),
            Metric("ground_truth", ground_truth, weight=0.0),
        )
    )


# ──────────────────────── Training ──────────────────────────────────────────


def train(
    config: GRPOConfig | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run calculator RL training.

    Args:
        config: Training config. If None, uses defaults.
        max_samples: Limit dataset size.

    Returns:
        Dict with metrics_history.
    """
    if config is None:
        config = GRPOConfig(
            experiment_name="calculator_grpo",
            lr=1e-5,
            n_samples_per_prompt=4,
            temperature=0.7,
            num_steps=10,
            max_turns=10,  # Multi-turn for tool use
            max_seq_len=2048,
        )

    prompts = load_calculator_prompts(max_samples=max_samples)

    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=calculator_score_fn,
        environment_cls=CalculatorEnvironment,
    )
