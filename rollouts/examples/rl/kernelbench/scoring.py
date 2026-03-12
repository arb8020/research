"""KernelBench scoring function.

Compiles and benchmarks generated kernels to compute rewards.

The scoring approach:
1. Extract kernel code from model response (from <kernel> tags or ```python blocks)
2. Send to SandboxPool for compilation/testing/benchmarking
3. Return Score with metrics

The SandboxPool handles the actual GPU execution, either:
- Locally via subprocess (default, for single-node setups)
- Remotely via miniray workers (for distributed scoring)

Usage:
    # Default: local subprocess scoring
    score = kernelbench_score_fn(sample)

    # With remote sandboxes:
    from rollouts.gpu_sandbox import SandboxPool, ModalSandboxConfig
    pool = SandboxPool([ModalSandboxConfig(gpu="A100", count=2)])
    await pool.start()
    configure_sandbox_pool(pool)
    # Now kernelbench_score_fn uses the pool
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rollouts.core import Score
    from rollouts.gpu_sandbox import SandboxPool
    from rollouts.training.types import Sample

logger = logging.getLogger(__name__)

# Module-level sandbox pool (configured via configure_sandbox_pool)
_sandbox_pool: SandboxPool | None = None


def configure_sandbox_pool(pool: SandboxPool | None) -> None:
    """Configure the sandbox pool for scoring.

    Args:
        pool: SandboxPool instance (or None to use local subprocess)
    """
    global _sandbox_pool
    _sandbox_pool = pool


def get_sandbox_pool() -> SandboxPool:
    """Get the configured sandbox pool, creating a default if needed."""
    global _sandbox_pool
    if _sandbox_pool is None:
        from rollouts.gpu_sandbox import SandboxPool

        _sandbox_pool = SandboxPool([])  # Empty = local subprocess
    return _sandbox_pool


def extract_kernel_code(response: str) -> str | None:
    """Extract code from <kernel> tags or ```python blocks.

    Tries multiple patterns in order of preference:
    1. <kernel>...</kernel> tags
    2. ```python...``` code blocks
    3. Raw "class ModelNew" if nothing else matches

    Args:
        response: Full model response

    Returns:
        Extracted code or None if no valid code found
    """
    # Try <kernel> tags first (our preferred format)
    match = re.search(r"<kernel>\s*(.*?)\s*</kernel>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fall back to ```python blocks
    match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Verify it has ModelNew
        if "class ModelNew" in code or "ModelNew" in code:
            return code

    # Last resort: extract from "class ModelNew" to end
    if "class ModelNew" in response:
        start = response.find("class ModelNew")
        return response[start:].strip()

    return None


async def score_kernel_async(
    kernel_code: str,
    ref_code: str,
    timeout: float = 120.0,
) -> dict:
    """Score a kernel using the sandbox pool.

    Args:
        kernel_code: Generated kernel code containing ModelNew class
        ref_code: Reference code with Model, get_inputs, get_init_inputs
        timeout: Scoring timeout in seconds

    Returns:
        Dict with: compiled, correct, speedup, reward, pass_rate, error
    """
    pool = get_sandbox_pool()

    # Ensure pool is started
    if not pool._started:
        await pool.start()

    return await pool.score_one(kernel_code, ref_code, timeout)


def score_kernel_sync(
    kernel_code: str,
    ref_code: str,
    timeout: float = 120.0,
) -> dict:
    """Synchronous wrapper for score_kernel_async.

    Creates an event loop if needed (for use in non-async contexts).
    """
    try:
        loop = asyncio.get_running_loop()
        # Already in async context - can't use run_until_complete
        # Create a new thread to run the async function
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run, score_kernel_async(kernel_code, ref_code, timeout)
            )
            return future.result()
    except RuntimeError:
        # No running loop - safe to use asyncio.run
        return asyncio.run(score_kernel_async(kernel_code, ref_code, timeout))


def kernelbench_score_fn(sample: Sample) -> Score:
    """Score function for KernelBench kernel generation.

    Extracts code from model response, sends to sandbox pool for
    compilation/testing/benchmarking.

    Reward formula: 0.2 * compiled + 1.0 * correct + speedup (if correct)

    Args:
        sample: Sample with response and metadata

    Returns:
        Score with metrics: compiled, correct, speedup, reward
    """
    from rollouts.core import Metric, Score

    # Get response and problem info
    response = sample.response if hasattr(sample, "response") else ""
    ref_code = sample.metadata.get("ref_code", "")

    # Default metrics
    compiled = 0.0
    correct = 0.0
    speedup = 0.0
    pass_rate = 0.0
    has_tags = 0.0
    error = None

    # Extract code
    kernel_code = extract_kernel_code(response)
    if kernel_code and ref_code:
        has_tags = 1.0

        # Score via sandbox pool
        result = score_kernel_sync(kernel_code, ref_code)

        compiled = result.get("compiled", 0.0)
        correct = result.get("correct", 0.0)
        speedup = result.get("speedup", 0.0)
        pass_rate = result.get("pass_rate", 0.0)
        error = result.get("error")

        if error:
            logger.debug(f"Scoring error: {error}")

    # Compute reward
    # - Failed to extract code: 0.0
    # - Failed to compile: 0.0
    # - Compiled but wrong: 0.2
    # - Correct at 1x: 1.2
    # - Correct at 2x: 2.2
    reward = 0.2 * compiled + 1.0 * correct + (speedup if correct > 0 else 0.0)

    return Score(
        metrics=(
            Metric("reward", reward, weight=1.0),
            Metric("compiled", compiled, weight=0.0),
            Metric("correct", correct, weight=0.0),
            Metric("speedup", speedup, weight=0.0),
            Metric("pass_rate", pass_rate, weight=0.0),
            Metric("has_tags", has_tags, weight=0.0),
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch scoring for efficiency
# ─────────────────────────────────────────────────────────────────────────────


async def score_batch_async(
    samples: list[Sample],
    timeout: float = 120.0,
) -> list[Score]:
    """Score a batch of samples using the sandbox pool.

    More efficient than calling kernelbench_score_fn repeatedly because
    it parallelizes across sandbox workers.

    Args:
        samples: List of Sample objects with response and metadata
        timeout: Per-sample scoring timeout

    Returns:
        List of Score objects (same order as input)
    """
    from rollouts.core import Metric, Score

    pool = get_sandbox_pool()

    # Ensure pool is started
    if not pool._started:
        await pool.start()

    # Prepare scoring requests
    requests = []
    for sample in samples:
        response = sample.response if hasattr(sample, "response") else ""
        ref_code = sample.metadata.get("ref_code", "")
        kernel_code = extract_kernel_code(response)

        if kernel_code and ref_code:
            requests.append({
                "kernel_code": kernel_code,
                "ref_code": ref_code,
                "has_tags": True,
            })
        else:
            requests.append({
                "kernel_code": "",
                "ref_code": "",
                "has_tags": False,
            })

    # Score all with valid code
    valid_requests = [
        {"kernel_code": r["kernel_code"], "ref_code": r["ref_code"]}
        for r in requests
        if r["has_tags"]
    ]

    if valid_requests:
        results = await pool.score_batch(valid_requests, timeout)
    else:
        results = []

    # Build scores, inserting results for valid requests
    scores = []
    result_idx = 0
    for req in requests:
        if req["has_tags"]:
            result = results[result_idx]
            result_idx += 1

            compiled = result.get("compiled", 0.0)
            correct = result.get("correct", 0.0)
            speedup = result.get("speedup", 0.0)
            pass_rate = result.get("pass_rate", 0.0)
            reward = 0.2 * compiled + 1.0 * correct + (speedup if correct > 0 else 0.0)

            scores.append(
                Score(
                    metrics=(
                        Metric("reward", reward, weight=1.0),
                        Metric("compiled", compiled, weight=0.0),
                        Metric("correct", correct, weight=0.0),
                        Metric("speedup", speedup, weight=0.0),
                        Metric("pass_rate", pass_rate, weight=0.0),
                        Metric("has_tags", 1.0, weight=0.0),
                    )
                )
            )
        else:
            # No valid code extracted
            scores.append(
                Score(
                    metrics=(
                        Metric("reward", 0.0, weight=1.0),
                        Metric("compiled", 0.0, weight=0.0),
                        Metric("correct", 0.0, weight=0.0),
                        Metric("speedup", 0.0, weight=0.0),
                        Metric("pass_rate", 0.0, weight=0.0),
                        Metric("has_tags", 0.0, weight=0.0),
                    )
                )
            )

    return scores
