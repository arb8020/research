"""KernelBench base configuration and train function.

Provides the core train() function used by all GRPO config files.
"""

from __future__ import annotations

from typing import Any

from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.grpo import GRPOConfig, grpo_train

from .dataset import load_kernelbench_prompts
from .scoring import kernelbench_score_fn


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
        python examples/kernelbench/grpo_level1.py --modal
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
            model=ModelConfig(name="Nanbeige/Nanbeige4.1-3B"),
            trainer=TrainerConfig(lr=1e-6, num_minibatches=4),
            rollout=RolloutConfig(
                batch_size=4,
                n_samples_per_prompt=4,
                temperature=0.7,
                max_seq_len=4096,
                max_tokens=2048,
            ),
            inference=InferenceConfig(mem_fraction=0.5),
            checkpoint=CheckpointConfig(num_steps=50, checkpoint_every=10),
        )

    # Load KernelBench problems as prompts
    prompts = load_kernelbench_prompts(
        levels=levels,
        max_samples=num_samples,
        backend=backend,
    )

    print(f"Loaded {len(prompts)} KernelBench problems from levels {levels}")

    # Single-turn training (model generates kernel, we score it)
    # Multi-turn training requires environment_factory support in grpo_train
    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=kernelbench_score_fn,
        environment_cls=BasicEnvironment,
    )
