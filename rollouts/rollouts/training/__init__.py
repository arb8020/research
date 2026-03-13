"""Training infrastructure for rollouts framework.

Includes:
- Training loops (SFT, RL)
- Dataset loading and preparation
- Rollout generation for RL
- Training backends (PyTorch, etc.)
- Metrics logging

Note: This module uses lazy imports for torch-dependent components
to allow importing training row types without torch installed.
"""

# Types first - these don't need torch
from ..training.contracts import (
    AdmissionPolicy,
    ForwardProducts,
    ModelInput,
    OverloadPolicy,
    PipelineRuntimeState,
    PrecisionPolicy,
    StalenessPolicy,
    StepResult,
    TrainableParameterPolicy,
    TrainingDatum,
    TrainingRuntimeState,
    VersionedRolloutBatch,
    WeightPublication,
    WeightSyncPolicy,
    WeightVersion,
    WeightVisibilityPolicy,
)
from ..training.lowering import (
    MegatronLowering,
    ParallelIntent,
    RealizationPlan,
    TorchTitanLowering,
)
from ..training.types import (
    AttemptRow,
    ProblemRow,
    RLTrainingConfig,
    RolloutBatch,
    RolloutConfig,
    SFTTrainingConfig,
    Status,
    TrainerConfig,
    TrainingSample,
)


def __getattr__(name: str) -> object:
    """Lazy import torch-dependent modules."""
    # Backends (need torch)
    if name == "PyTorchTrainingBackend":
        from ..training.backends import PyTorchTrainingBackend

        return PyTorchTrainingBackend
    if name == "TrainingBackend":
        from ..training.backends.protocol import TrainingBackend

        return TrainingBackend

    # Training loops (need torch)
    if name == "run_sft_training":
        from ..training.loops import run_sft_training

        return run_sft_training
    if name == "run_distill_training":
        from ..training.loops import run_distill_training

        return run_distill_training
    if name == "run_moe_sft_training":
        from ..training.loops import run_moe_sft_training

        return run_moe_sft_training
    if name == "run_rl_training":
        from ..training.loops import run_rl_training

        return run_rl_training

    # Datasets
    if name == "DataBuffer":
        from ..training.datasets import DataBuffer

        return DataBuffer
    if name == "load_sft_dataset":
        from ..training.datasets import load_sft_dataset

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
        from ..training import filters

        return getattr(filters, name)

    # Metrics
    if name == "JSONLLogger":
        from ..training.metrics import JSONLLogger

        return JSONLLogger
    if name == "MetricsLogger":
        from ..training.metrics import MetricsLogger

        return MetricsLogger

    # Rollout generation
    if name == "AsyncRolloutManager":
        from ..training.rollout_gen import AsyncRolloutManager

        return AsyncRolloutManager
    if name == "generate_rollout_batches":
        from ..training.rollout_gen import generate_rollout_batches

        return generate_rollout_batches

    # Agent integration
    if name == "agent_rollout_to_sample":
        from ..training.agent_integration import agent_rollout_to_sample

        return agent_rollout_to_sample
    if name == "generate_rollout_batch":
        from ..training.agent_integration import generate_rollout_batch

        return generate_rollout_batch
    if name == "trajectory_to_sample":
        from ..training.agent_integration import trajectory_to_sample

        return trajectory_to_sample
    if name == "trajectory_to_samples":
        from ..training.agent_integration import trajectory_to_samples

        return trajectory_to_samples

    # Loss functions
    if name in (
        "pretrain_loss",
        "sft_loss",
        "grpo_loss",
        "grpo_loss_clipped",
        "grpo_loss_masked",
        "ppo_loss",
        "LossOutput",
        "compute_group_advantages",
    ):
        from ..training import losses

        return getattr(losses, name)
    if name in (
        "distillation_contract_loss",
        "legacy_supervised_batch_to_training_datum",
        "moe_supervised_contract_loss",
        "rl_training_batch_to_datum",
        "training_sample_to_distill_datum",
        "training_sample_to_supervised_datum",
        "rl_contract_loss",
        "supervised_contract_loss",
    ):
        from ..training import contract_witnesses

        return getattr(contract_witnesses, name)

    # GRPO training
    if name == "GRPOConfig":
        from ..training.grpo import GRPOConfig

        return GRPOConfig
    if name == "grpo_train":
        from ..training.grpo import grpo_train

        return grpo_train

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Loops
    "run_sft_training",
    "run_distill_training",
    "run_moe_sft_training",
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
    "ProblemRow",
    "AttemptRow",
    "TrainingSample",
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
    "trajectory_to_samples",
    # Loss functions
    "pretrain_loss",
    "sft_loss",
    "grpo_loss",
    "grpo_loss_clipped",
    "grpo_loss_masked",
    "ppo_loss",
    "LossOutput",
    "compute_group_advantages",
    # Core training contracts
    "AdmissionPolicy",
    "ModelInput",
    "OverloadPolicy",
    "PipelineRuntimeState",
    "PrecisionPolicy",
    "StalenessPolicy",
    "TrainingDatum",
    "ForwardProducts",
    "StepResult",
    "TrainingRuntimeState",
    "TrainableParameterPolicy",
    "WeightPublication",
    "WeightVisibilityPolicy",
    "WeightSyncPolicy",
    "WeightVersion",
    "VersionedRolloutBatch",
    "MegatronLowering",
    "ParallelIntent",
    "RealizationPlan",
    "TorchTitanLowering",
    # Contract witness helpers
    "distillation_contract_loss",
    "legacy_supervised_batch_to_training_datum",
    "moe_supervised_contract_loss",
    "rl_training_batch_to_datum",
    "training_sample_to_distill_datum",
    "training_sample_to_supervised_datum",
    "rl_contract_loss",
    "supervised_contract_loss",
    # GRPO training
    "GRPOConfig",
    "grpo_train",
]
