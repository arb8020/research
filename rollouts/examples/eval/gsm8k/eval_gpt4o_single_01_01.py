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

Requires:
    OPENAI_API_KEY environment variable
"""

from base_config import (
    DatasetConfig,
    EndpointConfig,
    EvalConfig,
    EvalRunConfig,
    OutputConfig,
    evaluate_gsm8k,
)

config = EvalConfig(
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
    metrics = evaluate_gsm8k(config)
    print(f"\nAccuracy: {metrics.get('accuracy', 0):.1%}")
