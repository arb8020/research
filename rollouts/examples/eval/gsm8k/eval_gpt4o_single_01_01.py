#!/usr/bin/env python3
"""GSM8K Single-Turn Eval - GPT-4o-mini.

Naming: eval_gpt4o_single_01_01.py
- eval: evaluation
- gpt4o_single: GPT-4o single-turn (no tools)
- 01: experiment ID
- 01: parent ID (base config)

Model answers directly with \\boxed{} format.

Usage:
    python examples/eval/gsm8k/eval_gpt4o_single_01_01.py
    python examples/eval/gsm8k/eval_gpt4o_single_01_01.py --tui  # With TUI

Requires:
    OPENAI_API_KEY environment variable
"""

import argparse

from base_config import (
    DatasetConfig,
    EndpointConfig,
    EvalRunConfig,
    GSM8KConfig,
    OutputConfig,
    evaluate_gsm8k,
    run_with_tui,
)

config = GSM8KConfig(
    endpoint=EndpointConfig(
        provider="openai",
        model="gpt-4o-mini",
    ),
    dataset=DatasetConfig(
        max_samples=8,  # Start small
    ),
    run=EvalRunConfig(
        max_concurrent=4,
        use_tools=False,  # Single-turn
    ),
    output=OutputConfig(
        experiment_name="gsm8k_gpt4o_mini_single",
    ),
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tui", action="store_true", help="Show TUI monitor")
    args = parser.parse_args()

    if args.tui:
        metrics = run_with_tui(config)
    else:
        metrics = evaluate_gsm8k(config)

    print(f"\nAccuracy: {metrics.get('accuracy', 0):.1%}")
