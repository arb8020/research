#!/usr/bin/env python3
"""
Needle-in-a-Haystack evaluation for RLM.

Tests RLM's ability to find specific information hidden in massive contexts.
This is a synthetic benchmark that scales from 10K to 10M+ characters.

The benchmark:
1. Generates N lines of random filler text
2. Hides a "magic number" at a random position
3. Asks the model to find it using REPL tools

Usage:
    # Run with default settings (10K lines, ~500KB context)
    python -m examples.eval.rlm.needle_haystack

    # Run with 1M lines (~50MB context)
    python -m examples.eval.rlm.needle_haystack --lines 1000000

    # Compare RLM vs baseline (full context in messages)
    python -m examples.eval.rlm.needle_haystack --compare-baseline

    # Run multiple samples for statistical significance
    python -m examples.eval.rlm.needle_haystack --samples 20

Reference: https://alexzhang13.github.io/blog/2025/rlm/
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trio

from rollouts.agents import handle_stop_max_turns, run_agent
from rollouts.dtypes import (
    Actor,
    AgentState,
    Message,
    Metric,
    RunConfig,
    Score,
    Trajectory,
)

from .base_config import (
    DatasetConfig,
    EndpointConfig,
    EvalRunConfig,
    OutputConfig,
    RLMConfig,
    RLMEvalConfig,
    RLM_TOOL_SYSTEM_PROMPT,
    get_endpoint,
    get_sub_endpoint,
    numeric_match_score,
)

logger = logging.getLogger(__name__)


# ──────────────────────── Dataset Config ────────────────────────────────────


@dataclass(frozen=True)
class NeedleHaystackDatasetConfig(DatasetConfig):
    """Needle-in-haystack specific dataset config."""

    num_lines: int = 10_000  # Lines of random text
    seed: int = 42
    max_samples: int | None = 10  # Number of different needles to test


@dataclass(frozen=True)
class NeedleHaystackConfig(RLMEvalConfig):
    """Needle-in-haystack evaluation config."""

    dataset: NeedleHaystackDatasetConfig = field(
        default_factory=NeedleHaystackDatasetConfig
    )
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(experiment_name="needle_haystack")
    )


# ──────────────────────── Dataset Generation ────────────────────────────────


def generate_haystack(
    num_lines: int,
    magic_number: int,
    seed: int | None = None,
) -> tuple[str, int]:
    """Generate haystack with hidden needle.

    Args:
        num_lines: Number of lines of random filler
        magic_number: The number to hide
        seed: Random seed

    Returns:
        Tuple of (context_string, needle_position)
    """
    if seed is not None:
        random.seed(seed)

    words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
        "adipiscing", "elit", "sed", "do", "eiusmod", "tempor",
        "incididunt", "ut", "labore", "et", "dolore", "magna",
    ]

    lines = []
    for i in range(num_lines):
        num_words = random.randint(5, 15)
        line = " ".join(random.choice(words) for _ in range(num_words))
        lines.append(f"[{i:08d}] {line}")

    # Insert needle in middle third
    insert_pos = random.randint(num_lines // 3, 2 * num_lines // 3)
    lines[insert_pos] = f"[{insert_pos:08d}] SECRET: The magic number is {magic_number}. Remember this."

    return "\n".join(lines), insert_pos


def generate_samples(config: NeedleHaystackDatasetConfig) -> list[dict[str, Any]]:
    """Generate evaluation samples."""
    samples = []
    rng = random.Random(config.seed)

    n_samples = config.max_samples or 10

    for i in range(n_samples):
        magic_number = rng.randint(1000, 999999)
        sample_seed = config.seed + i if config.seed else None

        context, position = generate_haystack(
            num_lines=config.num_lines,
            magic_number=magic_number,
            seed=sample_seed,
        )

        samples.append({
            "id": f"needle_{i:04d}",
            "context": context,
            "magic_number": magic_number,
            "position": position,
            "num_lines": config.num_lines,
            "context_chars": len(context),
        })

    return samples


# ──────────────────────── Evaluation Logic ──────────────────────────────────


async def evaluate_sample_rlm(
    sample: dict[str, Any],
    config: NeedleHaystackConfig,
) -> dict[str, Any]:
    """Evaluate a single sample using RLM."""
    from rollouts.environments.repl import MessageParsingREPLEnvironment, REPLEnvironment

    endpoint = get_endpoint(config.endpoint)
    sub_endpoint = get_sub_endpoint(config.sub_endpoint)

    context = sample["context"]
    magic_number = sample["magic_number"]

    # Create environment
    if config.rlm.use_tool_calling:
        environment = REPLEnvironment(
            context=context,
            sub_endpoint=sub_endpoint,
            recursive=config.rlm.recursive,
            max_depth=config.rlm.max_depth,
        )
    else:
        environment = MessageParsingREPLEnvironment(
            context=context,
            sub_endpoint=sub_endpoint,
            recursive=config.rlm.recursive,
            max_depth=config.rlm.max_depth,
        )

    # Create trajectory (context NOT in messages - that's the point of RLM)
    trajectory = Trajectory(
        messages=[
            Message(role="system", content=RLM_TOOL_SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    "Find the magic number hidden in the context. "
                    "The context is very large, so use the REPL to search efficiently. "
                    "Use final_answer to submit the number when you find it."
                ),
            ),
        ]
    )

    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint,
        tools=environment.get_tools(),
    )

    state = AgentState(actor=actor, environment=environment)

    # Run
    async def silent_handler(_: object) -> None:
        await trio.lowlevel.checkpoint()

    run_config = RunConfig(
        on_chunk=silent_handler,
        handle_stop=handle_stop_max_turns(config.run.max_turns),
    )

    states = await run_agent(state, run_config)

    # Get result from the final state's environment (run_agent may create new instances)
    final_answer = None
    if states and states[-1].environment:
        final_answer = states[-1].environment._final_answer
    score = numeric_match_score(final_answer, magic_number)

    return {
        "sample_id": sample["id"],
        "magic_number": magic_number,
        "predicted": final_answer,
        "correct": score.metrics[0].value == 1.0,
        "num_turns": len(states),
        "context_chars": sample["context_chars"],
    }


async def evaluate_sample_baseline(
    sample: dict[str, Any],
    config: NeedleHaystackConfig,
) -> dict[str, Any]:
    """Evaluate a single sample using baseline (full context in messages)."""
    endpoint = get_endpoint(config.endpoint)

    context = sample["context"]
    magic_number = sample["magic_number"]

    # Baseline: put full context in messages (will likely truncate/fail for large contexts)
    trajectory = Trajectory(
        messages=[
            Message(
                role="system",
                content="You are a helpful assistant. Find information in the provided context.",
            ),
            Message(
                role="user",
                content=f"Find the magic number in this context:\n\n{context}\n\nWhat is the magic number?",
            ),
        ]
    )

    actor = Actor(trajectory=trajectory, endpoint=endpoint, tools=[])

    state = AgentState(actor=actor, environment=None)

    async def silent_handler(_: object) -> None:
        await trio.lowlevel.checkpoint()

    run_config = RunConfig(
        on_chunk=silent_handler,
        handle_stop=handle_stop_max_turns(1),  # Single turn for baseline
    )

    try:
        states = await run_agent(state, run_config)

        # Extract answer from last message
        final_answer = None
        for msg in reversed(states[-1].actor.trajectory.messages):
            if msg.role == "assistant" and msg.content:
                final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        score = numeric_match_score(final_answer, magic_number)

        return {
            "sample_id": sample["id"],
            "magic_number": magic_number,
            "predicted": final_answer[:200] if final_answer else None,  # Truncate for logging
            "correct": score.metrics[0].value == 1.0,
            "num_turns": len(states),
            "context_chars": sample["context_chars"],
            "error": None,
        }
    except Exception as e:
        return {
            "sample_id": sample["id"],
            "magic_number": magic_number,
            "predicted": None,
            "correct": False,
            "num_turns": 0,
            "context_chars": sample["context_chars"],
            "error": str(e),
        }


async def run_evaluation(config: NeedleHaystackConfig) -> dict[str, Any]:
    """Run full needle-in-haystack evaluation."""
    from rollouts._logging import setup_logging

    setup_logging(level="INFO", use_color=True)

    logger.info("=" * 60)
    logger.info(f"Needle-in-Haystack Evaluation")
    logger.info("=" * 60)
    logger.info(f"Lines per context: {config.dataset.num_lines:,}")
    logger.info(f"Samples: {config.dataset.max_samples}")
    logger.info(f"RLM enabled: {config.rlm.enabled}")
    logger.info(f"Model: {config.endpoint.provider}/{config.endpoint.model}")

    # Generate samples
    samples = generate_samples(config.dataset)
    logger.info(f"Generated {len(samples)} samples")
    logger.info(f"Context size: ~{samples[0]['context_chars']:,} chars each")

    # Run evaluation
    results = []

    async def eval_one(sample: dict) -> dict:
        if config.rlm.enabled:
            return await evaluate_sample_rlm(sample, config)
        else:
            return await evaluate_sample_baseline(sample, config)

    # Run sequentially (simpler, easier to debug)
    for sample in samples:
        result = await eval_one(sample)
        results.append(result)
        status = "✓" if result["correct"] else "✗"
        logger.info(f"  {status} {result['sample_id']}: {result['predicted']} (expected {result['magic_number']})")

    # Compute metrics
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    avg_turns = sum(r["num_turns"] for r in results) / total if total > 0 else 0.0

    logger.info("=" * 60)
    logger.info(f"Results: {correct}/{total} correct ({accuracy:.1%})")
    logger.info(f"Average turns: {avg_turns:.1f}")
    logger.info("=" * 60)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_turns": avg_turns,
        "results": results,
    }


# ──────────────────────── CLI ────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Needle-in-Haystack RLM Evaluation")

    parser.add_argument(
        "--lines",
        type=int,
        default=10_000,
        help="Lines per context (default: 10000, ~500KB)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to evaluate (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Model to use",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai"],
        help="API provider",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run baseline (full context in messages) for comparison",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run baseline, skip RLM",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum turns for RLM (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    async def run() -> None:
        # RLM evaluation
        if not args.baseline_only:
            rlm_config = NeedleHaystackConfig(
                endpoint=EndpointConfig(provider=args.provider, model=args.model),
                rlm=RLMConfig(enabled=True),
                dataset=NeedleHaystackDatasetConfig(
                    num_lines=args.lines,
                    max_samples=args.samples,
                    seed=args.seed,
                ),
                run=EvalRunConfig(max_turns=args.max_turns),
            )
            logger.info("\n>>> Running RLM evaluation...")
            rlm_results = await run_evaluation(rlm_config)

        # Baseline evaluation
        if args.compare_baseline or args.baseline_only:
            baseline_config = NeedleHaystackConfig(
                endpoint=EndpointConfig(provider=args.provider, model=args.model),
                rlm=RLMConfig(enabled=False),
                dataset=NeedleHaystackDatasetConfig(
                    num_lines=args.lines,
                    max_samples=args.samples,
                    seed=args.seed,
                ),
                run=EvalRunConfig(max_turns=1),
            )
            logger.info("\n>>> Running baseline evaluation...")
            baseline_results = await run_evaluation(baseline_config)

            if not args.baseline_only:
                logger.info("\n>>> Comparison:")
                logger.info(f"  RLM:      {rlm_results['accuracy']:.1%}")
                logger.info(f"  Baseline: {baseline_results['accuracy']:.1%}")

    trio.run(run)


if __name__ == "__main__":
    main()
