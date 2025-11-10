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
from training.types import Sample, RolloutConfig, RolloutBatch, TrainingConfig
from training.sft import (
    compute_loss_mask,
    tokenize_conversation,
    prepare_sft_sample,
    example_sft_rollout_fn,
    export_samples_to_jsonl,
    load_samples_from_jsonl,
    export_samples_to_huggingface_format,
)
from training.rollout_manager import RolloutManager, convert_to_batch
from training.async_rollout_manager import (
    AsyncRolloutManager,
    generate_rollout_batch,
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
    # SFT functions
    "compute_loss_mask",
    "tokenize_conversation",
    "prepare_sft_sample",
    "example_sft_rollout_fn",
    # JSONL export (D2)
    "export_samples_to_jsonl",
    "load_samples_from_jsonl",
    "export_samples_to_huggingface_format",
    # Orchestration
    "RolloutManager",
    "convert_to_batch",
    # Async orchestration (D4)
    "AsyncRolloutManager",
    "generate_rollout_batch",
]
