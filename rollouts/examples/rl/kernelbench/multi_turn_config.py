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

from rollouts.dtypes import Metric, Score
from rollouts.environments.kernelbench_multi import KernelBenchMultiTurnEnvironment
from rollouts.training.grpo import GRPOConfig, grpo_train

from .dataset import load_kernelbench_prompts


def kernelbench_multi_turn_score_fn(sample: Any) -> Score:
    """Score function for multi-turn KernelBench training.

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

    # Additional metrics
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


def train(
    config: GRPOConfig | None = None,
    num_samples: int = 50,
    levels: list[int] | None = None,
    backend: str = "cuda",
    max_turns: int = 8,
) -> dict[str, Any]:
    """Run KernelBench multi-turn RL training.

    Args:
        config: Training config. If None, uses defaults.
        num_samples: Number of problems to load from dataset.
        levels: KernelBench levels to use (default: [1, 2] for Kevin setup).
        backend: Kernel backend ("cuda" or "hip").
        max_turns: Maximum turns per problem (Kevin used 4-8).

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

    # Create environment factory that passes ref_code to each environment
    def environment_factory(sample_data: dict[str, Any]) -> KernelBenchMultiTurnEnvironment:
        return KernelBenchMultiTurnEnvironment(
            ref_code=sample_data.get("ref_code", ""),
            backend=backend,
            max_turns=max_turns,
        )

    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=kernelbench_multi_turn_score_fn,
        environment_cls=None,  # We use environment_factory instead
        environment_factory=environment_factory,
    )


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
