"""Training infrastructure for rollouts framework.

Includes:
- Training loops (SFT, RL)
- Dataset loading and preparation
- Rollout generation for RL
- Training backends (PyTorch, etc.)
- Metrics logging
"""

# Training loops
# Backends
from rollouts.training.backends import PyTorchTrainingBackend
from rollouts.training.backends.protocol import TrainingBackend

# Datasets
from rollouts.training.datasets import DataBuffer, load_sft_dataset
from rollouts.training.loops import run_rl_training, run_sft_training

# Metrics
from rollouts.training.metrics import JSONLLogger, MetricsLogger

# Rollout generation
from rollouts.training.rollout_gen import (
    AsyncRolloutManager,
    generate_rollout_batches,
)

# Types and utilities
from rollouts.training.types import (
    RLTrainingConfig,
    RolloutBatch,
    RolloutConfig,
    Sample,
    SFTTrainingConfig,
)

# Filters (SLIME-style)
from rollouts.training.filters import (
    check_reward_nonzero_std,
    check_min_reward,
    check_response_diversity,
    check_reasonable_length,
    check_any_success,
    check_quality_and_diversity,
    make_threshold_filter,
    make_length_filter,
)

# Agent integration
from rollouts.training.agent_integration import (
    agent_rollout_to_sample,
    generate_rollout_batch,
    trajectory_to_sample,
)

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
    "SFTTrainingConfig",
    "RLTrainingConfig",
    "RolloutConfig",
    "RolloutBatch",
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
]
