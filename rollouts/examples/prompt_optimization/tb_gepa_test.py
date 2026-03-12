"""GEPA optimization test for terminal-bench.

Tests that GEPA can optimize prompts for terminal-bench tasks using pure functions.

This is a small-scale test:
- 2 training tasks
- 1 validation task
- 4 evaluations total

Run:
    python -m examples.prompt_optimization.tb_gepa_test
"""

import logging
import os
from functools import partial

import trio

from rollouts.core import Endpoint
from rollouts.prompt_optimization import (
    GEPAConfig,
    TerminalBenchConfig,
    evaluate_terminal_bench,
    make_terminal_bench_reflective,
    run_gepa,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Simple initial prompt
SEED_PROMPT = """You are a terminal agent solving tasks in a Linux Docker container.

Available tools:
- send_keys: Send keystrokes (end with \\n to execute)
- capture_terminal: See terminal output
- task_complete: Signal when done

Strategy:
1. Explore with ls, cat, pwd
2. Identify the problem
3. Fix it
4. Verify the fix
5. Call task_complete"""


async def main() -> None:
    """Run GEPA optimization on terminal-bench tasks."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("No ANTHROPIC_API_KEY set")
        return

    # Setup endpoints
    agent_endpoint = Endpoint(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        api_key=api_key,
        max_tokens=4096,
    )

    reflection_endpoint = Endpoint(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        api_key=api_key,
        max_tokens=4096,
        temperature=0.7,
    )

    # Setup config (frozen dataclass)
    adapter_config = TerminalBenchConfig(
        endpoint=agent_endpoint,
        max_turns=15,
        max_concurrent=2,
    )

    # Small dataset for testing
    trainset = [
        {"task_id": "fix-permissions"},
        {"task_id": "log-summary"},
    ]
    valset = [
        {"task_id": "fix-permissions"},
    ]

    logger.info("Starting GEPA optimization")
    logger.info(f"Train tasks: {[t['task_id'] for t in trainset]}")
    logger.info(f"Val tasks: {[t['task_id'] for t in valset]}")

    # Run GEPA with pure functions
    gepa_config = GEPAConfig(
        max_evaluations=4,  # Very small for testing
        minibatch_size=1,
    )

    result = await run_gepa(
        seed_candidate={"instruction_prompt": SEED_PROMPT},
        dataset=trainset,
        evaluate_fn=partial(evaluate_terminal_bench, adapter_config),
        make_reflective_fn=make_terminal_bench_reflective,
        config=gepa_config,
        reflection_endpoint=reflection_endpoint,
        valset=valset,
        seed=42,
    )

    # Report results
    print()
    print("=" * 60)
    logger.info("GEPA optimization complete")
    logger.info(f"Total evaluations: {result.total_evaluations}")
    logger.info(f"Best validation score: {result.best_score:.3f}")
    logger.info(f"Iterations: {len(result.history)}")
    print()
    print("Optimized prompt:")
    print("-" * 40)
    print(result.best_candidate.get("instruction_prompt", "")[:500])
    if len(result.best_candidate.get("instruction_prompt", "")) > 500:
        print("... [truncated]")
    print("-" * 40)


if __name__ == "__main__":
    trio.run(main)
