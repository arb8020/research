#!/usr/bin/env python3
"""Stress test rate limiting across providers.

Usage:
    # Load env vars first
    export $(cat ~/wafer/.env | xargs)

    # Test OpenAI with high concurrency
    python scripts/stress_test_rate_limits.py --provider openai --samples 50 --concurrent 25

    # Test Anthropic
    python scripts/stress_test_rate_limits.py --provider anthropic --samples 50 --concurrent 25

    # Quick sanity check (low load, shouldn't hit limits)
    python scripts/stress_test_rate_limits.py --provider openai --samples 5 --concurrent 2

Expected behavior after implementation:
1. Logs show "rate_limit_discovered" with remaining/total from headers
2. As we approach limits, logs show "rate_limit_high_utilization"
3. If samples fail, "retrying_failed_samples" log appears
4. All samples eventually complete (no permanent rate limit failures)

Success criteria:
- Zero samples with error_type="provider_error" in final results
- Rate limit headers logged (after Phase 2 implementation)
- Sample retry works (after Phase 4 implementation)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add rollouts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import trio

from rollouts.agents import handle_stop_max_turns
from rollouts.dtypes import Endpoint, EvalConfig, Message, RunConfig, Score
from rollouts.evaluation import evaluate
from rollouts.logging_utils import init_rollout_logging

if TYPE_CHECKING:
    from rollouts.training.types import Sample

# Use rollouts logging setup - quiets httpx/httpcore/openai/anthropic at WARNING level
# while keeping our logs at INFO
init_rollout_logging(
    experiment_name="stress_test",
    log_level="INFO",
    # Can override specific loggers if needed:
    # logger_levels={"rollouts": "DEBUG", "httpx": "ERROR"},
)
logger = logging.getLogger(__name__)


def make_simple_dataset(n_samples: int) -> Iterator[dict[str, Any]]:
    """Generate simple math problems for stress testing.

    These are trivial problems so the model responds quickly,
    maximizing request throughput to stress rate limits.
    """
    for i in range(n_samples):
        a, b = i + 1, i + 2
        yield {
            "id": f"stress_{i:04d}",
            "question": f"What is {a} + {b}? Reply with just the number.",
            "answer": str(a + b),
        }


def simple_score_fn(sample: Sample) -> Score:
    """Simple scoring: check if answer appears in response."""
    from rollouts.dtypes import Metric

    expected = sample.ground_truth
    if sample.trajectory and sample.trajectory.messages:
        last_msg = sample.trajectory.messages[-1]
        response = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        correct = expected in response
    else:
        correct = False
        response = ""

    # Score only takes metrics tuple - reward is computed from weighted metrics
    return Score(
        metrics=(Metric(name="correct", value=1.0 if correct else 0.0, weight=1.0),),
    )


async def run_stress_test(
    provider: str,
    model: str,
    n_samples: int,
    max_concurrent: int,
    max_retries: int,
) -> dict:
    """Run stress test and return results summary."""

    # Get API key from environment
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    endpoint = Endpoint(
        provider=provider,
        model=model,
        api_key=api_key,
        max_retries=max_retries,
        max_tokens=50,  # Short responses for speed
        temperature=0.0,  # Deterministic
    )

    # Simple prepare_messages function for math problems
    def prepare_messages(sample: dict) -> list[Message]:
        return [
            Message(
                role="system",
                content="You are a helpful math assistant. Answer with just the number.",
            ),
            Message(role="user", content=sample["question"]),
        ]

    # Create RunConfig with stop condition - single turn only!
    async def noop_on_chunk(event: Any) -> None:
        pass

    run_config = RunConfig(
        on_chunk=noop_on_chunk,
        handle_stop=handle_stop_max_turns(1),  # Stop after 1 turn - critical!
    )

    config = EvalConfig(
        eval_name=f"stress_test_{provider}",
        endpoint=endpoint,
        prepare_messages=prepare_messages,
        score_fn=simple_score_fn,
        run_config=run_config,  # Pass the run config with stop condition
        max_samples=n_samples,
        max_concurrent=max_concurrent,
        show_progress=True,
        verbose=False,  # Less noise
    )

    logger.info(f"Starting stress test: provider={provider}, model={model}")
    logger.info(f"  samples={n_samples}, concurrent={max_concurrent}, max_retries={max_retries}")
    logger.info("  (progress bar should appear below - each sample takes ~2-5s)")

    import time

    start_time = time.time()

    dataset = make_simple_dataset(n_samples)
    report = await evaluate(dataset, config)

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.1f}s ({elapsed / n_samples:.1f}s per sample)")

    # Analyze results
    total = len(report.sample_results)
    provider_errors = sum(
        1 for s in report.sample_results if s.metadata.get("error_type") == "provider_error"
    )
    other_errors = sum(
        1
        for s in report.sample_results
        if s.metadata.get("error") and s.metadata.get("error_type") != "provider_error"
    )
    successful = total - provider_errors - other_errors

    summary = {
        "provider": provider,
        "model": model,
        "total_samples": total,
        "successful": successful,
        "provider_errors": provider_errors,
        "other_errors": other_errors,
        "accuracy": report.summary_metrics.get("accuracy", 0),
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test rate limiting")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="Provider to test",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use (default: gpt-4o-mini for openai, claude-3-5-haiku-latest for anthropic)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples to run",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="Max concurrent requests",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Max retries per request (SDK level)",
    )

    args = parser.parse_args()

    # Default models per provider
    if args.model is None:
        if args.provider == "openai":
            args.model = "gpt-4o-mini"
        elif args.provider == "anthropic":
            args.model = "claude-3-5-haiku-latest"

    try:
        summary = trio.run(
            run_stress_test,
            args.provider,
            args.model,
            args.samples,
            args.concurrent,
            args.max_retries,
        )

        print("\n" + "=" * 60)
        print("STRESS TEST RESULTS")
        print("=" * 60)
        print(json.dumps(summary, indent=2))

        # Success criteria
        if summary["provider_errors"] > 0:
            print(f"\n⚠️  WARNING: {summary['provider_errors']} samples failed with provider errors")
            print("   These should be retried by sample-level retry (Phase 4)")
            sys.exit(1)
        else:
            print("\n✓ All samples completed successfully")
            sys.exit(0)

    except Exception as e:
        logger.exception(f"Stress test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
