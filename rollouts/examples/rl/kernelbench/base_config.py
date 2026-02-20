"""KernelBench RL training config.

GRPO training for GPU kernel optimization using KernelBench problems.
Based on the reverse_text example pattern.

Key differences from reverse_text:
1. Kernels require GPU compilation and benchmarking for scoring
2. Responses are longer (kernel code vs short strings)
3. Reward includes speedup metric (unbounded upside)
"""

from __future__ import annotations

from typing import Any

from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.grpo import GRPOConfig, grpo_train

from .dataset import load_kernelbench_prompts
from .scoring import kernelbench_score_fn

# Re-export score function for external use
__all__ = ["train", "kernelbench_score_fn", "load_kernelbench_prompts"]


def train(
    config: GRPOConfig | None = None,
    num_samples: int = 50,
    levels: list[int] | None = None,
    backend: str = "CUDA",
) -> dict[str, Any]:
    """Run KernelBench RL training.

    Args:
        config: Training config. If None, uses defaults.
        num_samples: Number of problems to load from dataset.
        levels: KernelBench levels to use (default: [1] for easiest).
        backend: Kernel backend ("CUDA" or "HIP").

    Returns:
        Dict with metrics_history.

    Example:
        # Basic training (Level 1, 50 problems)
        python examples/rl/kernelbench/grpo_01_01.py --modal

        # Custom config
        from examples.rl.kernelbench.base_config import train
        from rollouts.training.grpo import GRPOConfig, ModelConfig

        config = GRPOConfig(model=ModelConfig(name="Nanbeige/Nanbeige4.1-3B"))
        results = train(config=config, num_samples=100, levels=[1, 2])
    """
    if levels is None:
        levels = [1]

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
            output=GRPOOutputConfig(experiment_name="kernelbench_grpo"),
            # Small model for fast iteration
            # Alternative: "zai-org/GLM-4.7-Flash" for larger capacity
            model=ModelConfig(name="Nanbeige/Nanbeige4.1-3B"),
            trainer=TrainerConfig(
                lr=1e-6,  # Conservative LR for code generation
                num_minibatches=4,  # Small batches for long sequences
            ),
            rollout=RolloutConfig(
                batch_size=4,  # Fewer prompts due to expensive scoring
                n_samples_per_prompt=4,  # Fewer samples due to scoring cost
                temperature=0.7,  # Moderate exploration
                max_seq_len=4096,  # Kernels can be long
                max_tokens=2048,  # Allow long responses
            ),
            inference=InferenceConfig(
                mem_fraction=0.5,  # Leave room for kernel compilation
            ),
            checkpoint=CheckpointConfig(
                num_steps=50,
                checkpoint_every=10,
                sync_weights_every=1,  # On-policy
            ),
        )

    # Load KernelBench problems as prompts
    prompts = load_kernelbench_prompts(
        levels=levels,
        max_samples=num_samples,
        backend=backend,
    )

    print(f"Loaded {len(prompts)} KernelBench problems from levels {levels}")

    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=kernelbench_score_fn,
        environment_cls=BasicEnvironment,  # No tools - single-turn generation
    )
