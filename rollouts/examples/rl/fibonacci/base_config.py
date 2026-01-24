"""Fibonacci RL training config.

Single-turn code generation task - model writes Python code, we grade correctness + speed.
Used for testing TTT (test-time training) infrastructure.
"""

from __future__ import annotations

from typing import Any

import trio

from rollouts.dtypes import Metric, Score
from rollouts.environments.ttt.code_challenge import (
    FIBONACCI_TEST_CASES,
    CodeChallengeEnvironment,
    FibonacciEnvironment,
    extract_code,
    run_code_with_tests,
)
from rollouts.training.grpo import GRPOConfig, grpo_train
from rollouts.training.types import Sample

# ──────────────────────── Environment Factory ─────────────────────────────────


def FibonacciEnvFactory() -> CodeChallengeEnvironment:
    """Factory for GRPO - creates configured FibonacciEnvironment.

    GRPO calls environment_cls() with no args, so we need a callable
    that returns a configured environment.
    """
    return FibonacciEnvironment(timeout=2.0)


# ──────────────────────── Dataset ─────────────────────────────────────────────


def load_fibonacci_prompts(n_prompts: int = 8) -> list[dict[str, Any]]:
    """Load Fibonacci task prompts.

    For TTT, we typically use the same prompt repeated (training on one problem).
    For standard RL, we might vary the problem slightly.

    Returns list of dicts with:
        - messages: List of chat messages
    """
    env = FibonacciEnvironment()
    prompts = []

    for _ in range(n_prompts):
        prompts.append({
            "messages": [{"role": "user", "content": env.prompt}],
        })

    return prompts


# ──────────────────────── Score Function ──────────────────────────────────────


def fibonacci_score_fn(sample: Sample) -> Score:
    """Score function for Fibonacci task.

    Extracts code from response, runs it, grades correctness + speed.
    """
    response = sample.response if hasattr(sample, "response") else ""
    code = extract_code(response)

    if code is None:
        return Score(
            metrics=(
                Metric("correctness", 0.0, weight=1.0, metadata={"error": "No code found"}),
                Metric("runtime_ms", 0.0, weight=0),
                Metric("speed_bonus", 0.0, weight=0.1),
            )
        )

    # Run grading (score_fn must be sync, so we use trio.run)
    result = trio.run(
        run_code_with_tests,
        code,
        "fib",
        FIBONACCI_TEST_CASES,
        2.0,  # timeout
    )

    # Speed bonus: faster = higher
    if result.timed_out:
        speed_bonus = 0.0
    else:
        speed_bonus = max(0.0, 1.0 - result.runtime_ms / 2000.0)

    return Score(
        metrics=(
            Metric(
                "correctness",
                result.correctness,
                weight=1.0,
                metadata={
                    "passed": result.passed,
                    "total": result.total,
                    "error": result.error,
                },
            ),
            Metric("runtime_ms", result.runtime_ms, weight=0),
            Metric("speed_bonus", speed_bonus, weight=0.1),
        )
    )


# ──────────────────────── Training ────────────────────────────────────────────


def train(
    config: GRPOConfig | None = None,
    n_prompts: int = 8,
) -> dict[str, Any]:
    """Run Fibonacci RL training.

    Args:
        config: Training config. If None, uses defaults.
        n_prompts: Number of prompts in dataset.

    Returns:
        Dict with metrics_history.
    """
    if config is None:
        config = GRPOConfig(
            experiment_name="fibonacci_grpo",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            lr=1e-6,
            n_samples_per_prompt=4,
            temperature=0.8,
            num_steps=10,
            max_turns=1,  # Single turn - just generate code
            max_seq_len=1024,
            max_tokens=512,
        )

    prompts = load_fibonacci_prompts(n_prompts=n_prompts)

    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=fibonacci_score_fn,
        environment_cls=FibonacciEnvFactory,  # Pass the class, not instance
    )
