"""Training infrastructure for rollouts framework.

Includes:
- Training loops (SFT, RL)
- Dataset loading and preparation
- Rollout generation for RL
- Training backends (PyTorch, etc.)
- Metrics logging

Note: This module uses lazy imports for torch-dependent components
to allow importing Sample/types without torch installed.
"""

# Types first - these don't need torch
from rollouts.training.types import (
    RLTrainingConfig,
    RolloutBatch,
    RolloutConfig,
    Sample,
    SFTTrainingConfig,
    Status,
    TrainerConfig,
)


def __getattr__(name: str):
    """Lazy import torch-dependent modules."""
    # Backends (need torch)
    if name == "PyTorchTrainingBackend":
        from rollouts.training.backends import PyTorchTrainingBackend

        return PyTorchTrainingBackend
    if name == "TrainingBackend":
        from rollouts.training.backends.protocol import TrainingBackend

        return TrainingBackend

    # Training loops (need torch)
    if name == "run_sft_training":
        from rollouts.training.loops import run_sft_training

        return run_sft_training
    if name == "run_rl_training":
        from rollouts.training.loops import run_rl_training

        return run_rl_training

    # Datasets
    if name == "DataBuffer":
        from rollouts.training.datasets import DataBuffer

        return DataBuffer
    if name == "load_sft_dataset":
        from rollouts.training.datasets import load_sft_dataset

        return load_sft_dataset

    # Filters
    if name in (
        "check_any_success",
        "check_min_reward",
        "check_quality_and_diversity",
        "check_reasonable_length",
        "check_response_diversity",
        "check_reward_nonzero_std",
        "make_length_filter",
        "make_threshold_filter",
    ):
        from rollouts.training import filters

        return getattr(filters, name)

    # Metrics
    if name == "JSONLLogger":
        from rollouts.training.metrics import JSONLLogger

        return JSONLLogger
    if name == "MetricsLogger":
        from rollouts.training.metrics import MetricsLogger

        return MetricsLogger

    # Rollout generation
    if name == "AsyncRolloutManager":
        from rollouts.training.rollout_gen import AsyncRolloutManager

        return AsyncRolloutManager
    if name == "generate_rollout_batches":
        from rollouts.training.rollout_gen import generate_rollout_batches

        return generate_rollout_batches

    # Agent integration
    if name == "agent_rollout_to_sample":
        from rollouts.training.agent_integration import agent_rollout_to_sample

        return agent_rollout_to_sample
    if name == "generate_rollout_batch":
        from rollouts.training.agent_integration import generate_rollout_batch

        return generate_rollout_batch
    if name == "trajectory_to_sample":
        from rollouts.training.agent_integration import trajectory_to_sample

        return trajectory_to_sample

    # Loss functions
    if name in (
        "pretrain_loss",
        "sft_loss",
        "grpo_loss",
        "grpo_loss_clipped",
        "grpo_loss_masked",
        "ppo_loss",
        "LossOutput",
    ):
        from rollouts.training import losses

        return getattr(losses, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Loops
    "run_sft_training",
    "run_rl_training",
    # Datasets
    "DataBuffer",
    "load_sft_dataset",
    # Rollout generation
    "generate_rollout_batches",
    "AsyncRolloutManager",
    # Backends
    "PyTorchTrainingBackend",
    "TrainingBackend",
    # Metrics
    "MetricsLogger",
    "JSONLLogger",
    # Types
    "Sample",
    "Status",
    "SFTTrainingConfig",
    "RLTrainingConfig",
    "RolloutConfig",
    "RolloutBatch",
    "TrainerConfig",
    # Filters (SLIME-style)
    "check_reward_nonzero_std",
    "check_min_reward",
    "check_response_diversity",
    "check_reasonable_length",
    "check_any_success",
    "check_quality_and_diversity",
    "make_threshold_filter",
    "make_length_filter",
    # Agent integration
    "agent_rollout_to_sample",
    "generate_rollout_batch",
    "trajectory_to_sample",
    # Loss functions
    "pretrain_loss",
    "sft_loss",
    "grpo_loss",
    "grpo_loss_clipped",
    "grpo_loss_masked",
    "ppo_loss",
    "LossOutput",
]
