"""KernelBench multi-turn RL training config.

Multi-turn RL training for kernel optimization using the Kevin approach.
Model iteratively generates kernels, gets execution feedback, and refines.

Based on: Kevin: Multi-Turn RL for Generating CUDA Kernels
https://arxiv.org/abs/2507.11948

Run with:
    python examples/rl/kernelbench/multi_turn_config.py --modal

Key differences from single-turn:
- Uses KernelBenchMultiTurnEnvironment (no tools, parses code from responses)
- max_turns=8 (model can iterate)
- Reward includes discounted future rewards (Kevin-style)
"""

from __future__ import annotations

from typing import Any

import trio

from rollouts.training.grpo import GRPOConfig, grpo_train

from .dataset import load_kernelbench_prompts
from .resources import KernelBenchRolloutResources, KernelBenchScoringResources
from .scoring import KEVIN_MULTI_TURN_REWARD_WEIGHTS


def train(
    config: GRPOConfig | None = None,
    num_samples: int = 50,
    levels: list[int] | None = None,
    backend: str = "cuda",
    max_turns: int = 8,
    sandbox_configs: list[Any] | None = None,
) -> dict[str, Any]:
    """Run KernelBench multi-turn RL training.

    Args:
        config: Training config. If None, uses defaults.
        num_samples: Number of problems to load from dataset.
        levels: KernelBench levels to use (default: [1, 2] for Kevin setup).
        backend: Kernel backend ("cuda" or "hip").
        max_turns: Maximum turns per problem (Kevin used 4-8).
        sandbox_configs: Optional GPU sandbox configs for kernel evaluation.

    Returns:
        Dict with metrics_history.

    Example:
        # Basic training (Level 1-2, 50 problems, 8 turns)
        python examples/rl/kernelbench/multi_turn_config.py --modal

        # Custom config
        from examples.rl.kernelbench.multi_turn_config import train
        results = train(num_samples=100, levels=[1, 2], max_turns=8)
    """
    if levels is None:
        levels = [1, 2]  # Kevin used Level 1 and 2

    if config is None:
        from rollouts.training.grpo import (
            CheckpointConfig,
            GRPOOutputConfig,
            InferenceConfig,
            ModelConfig,
            RolloutConfig,
            TrainerConfig,
        )

        config = GRPOConfig(
            output=GRPOOutputConfig(experiment_name="kernelbench_multi_turn"),
            # Kevin used QwQ-32B, but we start with smaller model
            model=ModelConfig(name="Nanbeige/Nanbeige4.1-3B"),
            trainer=TrainerConfig(
                lr=1e-6,  # Conservative LR for code generation
                num_minibatches=4,
                # Kevin used 2 gradient steps per batch (1 on-policy, 1 off-policy)
                # This is handled by the trainer internally
            ),
            rollout=RolloutConfig(
                batch_size=4,  # Number of problems per step
                n_samples_per_prompt=4,  # Parallel trajectories per problem
                temperature=0.9,  # Kevin used 0.9
                max_turns=max_turns,  # Multi-turn: up to 8 turns
                max_seq_len=16384,  # Kevin used 16K (then extended to 22K)
                max_tokens=4096,  # Per-turn generation limit
            ),
            inference=InferenceConfig(
                mem_fraction=0.5,  # Leave room for kernel compilation
            ),
            checkpoint=CheckpointConfig(
                num_steps=50,  # Kevin trained up to 40 steps (80 gradient steps)
                checkpoint_every=10,
                sync_weights_every=1,  # On-policy
            ),
        )

    # Load KernelBench problems
    prompts = load_kernelbench_prompts(
        levels=levels,
        max_samples=num_samples,
        backend=backend,
    )

    print(f"Loaded {len(prompts)} KernelBench problems from levels {levels}")
    print(f"Multi-turn setup: max_turns={max_turns}")

    rollout_resources = KernelBenchRolloutResources.from_sandbox_configs(
        sandbox_configs,
        backend=backend,
        max_turns=max_turns,
    )
    scoring_resources = KernelBenchScoringResources.metadata_only(
        reward_weights=KEVIN_MULTI_TURN_REWARD_WEIGHTS,
    )
    trio.run(rollout_resources.start)
    try:
        return grpo_train(
            config=config,
            prompts=prompts,
            sample_scorer=scoring_resources.scorer,
            environment_cls=None,  # We use environment_factory instead
            environment_factory=rollout_resources,
        )
    finally:
        trio.run(rollout_resources.stop)


# Default config for direct execution
from dataclasses import replace

from examples.rl.kernelbench.grpo_01_01 import config as base_config

# Multi-turn configuration
config = replace(
    base_config,
    output=replace(
        base_config.output,
        experiment_name="kernelbench_multi_turn_01",
    ),
    rollout=replace(
        base_config.rollout,
        max_turns=8,  # Multi-turn: allow up to 8 iterations
        temperature=0.9,  # Higher temp for exploration (Kevin used 0.9)
        max_seq_len=16384,  # Longer context for multi-turn history
        max_tokens=4096,  # Per-turn limit
    ),
    trainer=replace(
        base_config.trainer,
        lr=1e-6,
    ),
)

if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
