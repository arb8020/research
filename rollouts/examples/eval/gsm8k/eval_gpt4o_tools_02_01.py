#!/usr/bin/env python3
"""GSM8K Multi-Turn Eval - GPT-4o-mini with Calculator.

Naming: eval_gpt4o_tools_02_01.py
- eval: evaluation
- gpt4o_tools: GPT-4o with calculator tools
- 02: experiment ID
- 01: parent ID (base config)

Model uses calculator tools to solve problems.

Usage:
    python examples/eval/gsm8k/eval_gpt4o_tools_02_01.py

Requires:
    OPENAI_API_KEY environment variable
"""

from base_config import (
    DatasetConfig,
    EndpointConfig,
    EvalRunConfig,
    GSM8KConfig,
    OutputConfig,
    evaluate_gsm8k,
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
        max_turns=10,
        use_tools=True,  # Multi-turn with calculator
    ),
    output=OutputConfig(
        experiment_name="gsm8k_gpt4o_mini_tools",
    ),
)


if __name__ == "__main__":
    metrics = evaluate_gsm8k(config)
    print(f"\nMean reward: {metrics.get('mean_reward', 0):.1%}")
