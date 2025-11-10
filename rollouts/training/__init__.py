"""Training system - uses rollouts agent framework.

Functional, composable training infrastructure inspired by SLIME's architecture
and Casey Muratori's API design principles.

This is a SEPARATE package from rollouts/ (the agent framework).
Training uses the agent framework but is not part of it.

Key design principles:
- Minimal stateful components (only DataBuffer)
- Pure functions for all transformations
- User-provided functions (SLIME-style)
- No hidden coupling or retention
- Explicit control flow (no callbacks)

Usage:
    from rollouts import Agent, Endpoint  # Agent framework
    from training import DataBuffer       # Training system
"""

from training.data_buffer import DataBuffer, load_prompts_from_jsonl, load_prompts_from_list
from training.types import Sample, RolloutConfig, RolloutBatch, TrainingConfig, RLTrainingConfig
from training.sft import (
    compute_loss_mask,
    tokenize_conversation,
    prepare_sft_sample,
    example_sft_rollout_fn,
    export_samples_to_jsonl,
    load_samples_from_jsonl,
    export_samples_to_huggingface_format,
)
from training.rollout_generation import (
    generate_rollout_batches,
    convert_to_batch,
    apply_sample_transforms,
    extract_sample_fields,
    compute_response_lengths,
    build_batch_metadata,
)
from training.rollout_manager import RolloutManager  # Deprecated
from training.async_rollout_manager import (
    AsyncRolloutManager,
    generate_rollout_batch,
)
from training.weight_sync import (
    InferenceEngine,
    SGLangEngine,
    VLLMEngine,
    update_sglang_weights_from_disk,
    update_vllm_weights_from_disk,
    sync_weights_to_engines,
)
from training.sft_loop import (
    run_sft_training,
    collate_batch,
    prepare_sft_batch,
)
from training.rl_loop import (
    run_rl_training,
    compute_reward,
    prepare_grpo_batch,
    compute_advantages,
)
from training.rl_losses import (
    grpo_loss,
    ppo_loss,
)

__all__ = [
    # Data management
    "DataBuffer",
    "load_prompts_from_jsonl",
    "load_prompts_from_list",
    # Types
    "Sample",
    "RolloutConfig",
    "RolloutBatch",
    "TrainingConfig",
    "RLTrainingConfig",
    # SFT functions
    "compute_loss_mask",
    "tokenize_conversation",
    "prepare_sft_sample",
    "example_sft_rollout_fn",
    # JSONL export (D2)
    "export_samples_to_jsonl",
    "load_samples_from_jsonl",
    "export_samples_to_huggingface_format",
    # Rollout generation (pure functions)
    "generate_rollout_batches",
    "convert_to_batch",
    "apply_sample_transforms",
    "extract_sample_fields",
    "compute_response_lengths",
    "build_batch_metadata",
    # Orchestration (deprecated)
    "RolloutManager",  # DEPRECATED: Use generate_rollout_batches instead
    # Async orchestration (D4)
    "AsyncRolloutManager",
    "generate_rollout_batch",
    # Weight sync (D5)
    "InferenceEngine",
    "SGLangEngine",
    "VLLMEngine",
    "update_sglang_weights_from_disk",
    "update_vllm_weights_from_disk",
    "sync_weights_to_engines",
    # SFT training loop (Phase 2)
    "run_sft_training",
    "collate_batch",
    "prepare_sft_batch",
    # RL training loop (Phase 3)
    "run_rl_training",
    "compute_reward",
    "prepare_grpo_batch",
    "compute_advantages",
    # RL losses (Phase 3)
    "grpo_loss",
    "ppo_loss",
]
