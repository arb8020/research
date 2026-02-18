#!/usr/bin/env python3
"""Run the functional extractor eval.

Usage:
    python run.py smoke    # Quick smoke test (1 task, SmolLM2-135M)
    python run.py full     # Full evaluation (all 8 models)

    # With overrides
    python run.py smoke --model claude-opus-4-5-20251101
    python run.py full --max-concurrent 8
"""

import argparse
import sys
from pathlib import Path

# Add rollouts to path
ROLLOUTS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROLLOUTS_ROOT))

from rollouts.eval_runner import run_eval_from_spec

from eval import get_spec, load_tasks


def main():
    parser = argparse.ArgumentParser(description="Run functional extractor eval")
    parser.add_argument("mode", choices=["smoke", "full"], help="Eval mode")
    parser.add_argument("--model", type=str, help="Override model")
    parser.add_argument("--provider", type=str, help="Override provider")
    parser.add_argument("--max-turns", type=int, help="Override max turns")
    parser.add_argument("--max-concurrent", type=int, help="Override max concurrent")
    parser.add_argument("--limit", type=int, help="Override task limit")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--verbose", action="store_true", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Load config for mode
    if args.mode == "smoke":
        from configs.smoke import endpoint, output, run
        from configs.smoke import tasks_override
        tasks = tasks_override
    else:
        from configs.full import endpoint, output, run
        tasks = load_tasks(None)

    # Apply limit from config
    if run.limit is not None:
        tasks = tasks[:run.limit]

    # Handle verbose/quiet
    verbose = run.verbose
    if args.verbose:
        verbose = True
    if args.quiet:
        verbose = False

    print(f"\n{'='*60}")
    print(f"Functional Extractor Eval - {args.mode.upper()} mode")
    print(f"{'='*60}")
    print(f"Tasks: {len(tasks)}")
    print(f"Model: {args.model or endpoint.model}")
    print(f"Max turns: {args.max_turns or run.max_turns}")
    print(f"Max concurrent: {args.max_concurrent or run.max_concurrent}")
    print(f"{'='*60}\n")

    # Run eval
    result = run_eval_from_spec(
        spec=get_spec(),
        tasks=tasks,
        endpoint=endpoint,
        run=run,
        output=output,
        # Overrides
        model=args.model,
        provider=args.provider,
        max_turns=args.max_turns,
        max_concurrent=args.max_concurrent,
        limit=args.limit,
        verbose=verbose,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"{'='*60}")
    print(f"Total samples: {result.get('total_samples', result.get('total', 0))}")
    print(f"Success rate: {result.get('success_rate', 0):.1%}")
    print(f"Mean passed: {result.get('mean_passed', 0):.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
