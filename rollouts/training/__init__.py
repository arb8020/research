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
)
from training.rollout_manager import RolloutManager, convert_to_batch

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
    # Orchestration
    "RolloutManager",
    "convert_to_batch",
]
