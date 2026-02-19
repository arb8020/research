"""KernelBench scoring function.

Computes rewards for kernel optimization based on correctness and speedup.
Follows the Kevin paper formula: S = 0.3 * correct + speedup (if correct)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rollouts.dtypes import Score
    from rollouts.training.types import Sample


def kernelbench_score_fn(sample: Sample) -> Score:
    """Score function for multi-turn KernelBench evaluation/training.

    Implements the Kevin reward formula:
    - S = 0.3 * correct + speedup (if correct)

    For multi-turn trajectories, we compute reward based on:
    - Best speedup achieved across all turns
    - Whether any kernel was correct

    Args:
        sample: Sample with trajectory and metadata

    Returns:
        Score with reward and metrics
    """
    from rollouts.dtypes import Metric, Score

    metadata = sample.metadata if hasattr(sample, "metadata") else {}

    # Extract results from metadata (set by environment)
    best_speedup = metadata.get("best_speedup", 0.0)
    has_correct = metadata.get("has_correct_kernel", False)
    turns_used = metadata.get("turns_used", 0)
    turn_history = metadata.get("turn_history", [])

    # Kevin reward formula: 0.3 * correct + speedup
    if has_correct:
        reward = 0.3 + best_speedup
    else:
        reward = 0.0

    # Additional metrics for tracking
    compiled_any = any(t.get("compiled", False) for t in turn_history)
    correct_any = any(t.get("correct", False) for t in turn_history)

    return Score(
        metrics=(
            Metric("reward", reward, weight=1.0),
            Metric("best_speedup", best_speedup, weight=0.0),
            Metric("has_correct", 1.0 if has_correct else 0.0, weight=0.0),
            Metric("compiled_any", 1.0 if compiled_any else 0.0, weight=0.0),
            Metric("correct_any", 1.0 if correct_any else 0.0, weight=0.0),
            Metric("turns_used", float(turns_used), weight=0.0),
        )
    )


def kernelbench_single_turn_score_fn(sample: Sample) -> Score:
    """Score function for single-turn KernelBench (no environment feedback).

    Used when evaluating without the multi-turn environment.
    Extracts kernel from response and scores directly.

    Args:
        sample: Sample with response and ref_code in metadata

    Returns:
        Score with reward and metrics
    """

    from rollouts.dtypes import Metric, Score

    response = sample.response if hasattr(sample, "response") else ""
    metadata = sample.metadata if hasattr(sample, "metadata") else {}
    ref_code = metadata.get("ref_code", "")

    # Extract kernel code from response
    kernel_code = _extract_kernel_code(response)

    if not kernel_code or not ref_code:
        return Score(
            metrics=(
                Metric("reward", 0.0, weight=1.0),
                Metric("compiled", 0.0, weight=0.0),
                Metric("correct", 0.0, weight=0.0),
                Metric("speedup", 0.0, weight=0.0),
                Metric("has_code", 0.0 if not kernel_code else 1.0, weight=0.0),
            )
        )

    # Score via sandbox pool (sync wrapper)
    from rollouts.gpu_sandbox import SandboxPool

    pool = SandboxPool([])  # Local subprocess

    import asyncio

    async def score_async() -> dict:
        await pool.start()
        return await pool.score_one(kernel_code, ref_code, timeout=120.0)

    try:
        result = asyncio.run(score_async())
    except Exception as e:
        return Score(
            metrics=(
                Metric("reward", 0.0, weight=1.0),
                Metric("compiled", 0.0, weight=0.0),
                Metric("correct", 0.0, weight=0.0),
                Metric("speedup", 0.0, weight=0.0),
                Metric("error", 1.0, weight=0.0, metadata={"error": str(e)}),
            )
        )

    compiled = result.get("compiled", 0.0) > 0.5
    correct = result.get("correct", 0.0) > 0.5
    speedup = result.get("speedup", 0.0)

    # Kevin formula: 0.3 * correct + speedup
    reward = (0.3 + speedup) if correct else 0.0

    return Score(
        metrics=(
            Metric("reward", reward, weight=1.0),
            Metric("compiled", 1.0 if compiled else 0.0, weight=0.0),
            Metric("correct", 1.0 if correct else 0.0, weight=0.0),
            Metric("speedup", speedup, weight=0.0),
            Metric("has_code", 1.0, weight=0.0),
        )
    )


def _extract_kernel_code(response: str) -> str | None:
    """Extract kernel code from model response.

    Tries multiple patterns:
    1. ```python ... ``` code blocks
    2. <kernel> ... </kernel> tags
    3. Raw "class ModelNew" if nothing else matches
    """
    import re

    # Try ```python blocks first
    match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "class ModelNew" in code:
            return code

    # Try <kernel> tags
    match = re.search(r"<kernel>\s*(.*?)\s*</kernel>", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "class ModelNew" in code:
            return code

    # Last resort: extract from "class ModelNew" to end
    if "class ModelNew" in response:
        start = response.find("class ModelNew")
        return response[start:].strip()

    return None
