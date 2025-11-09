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

from training.data_buffer import DataBuffer, load_prompts_from_jsonl
from training.types import Sample, RolloutConfig, RolloutBatch, TrainingConfig

__all__ = [
    "DataBuffer",
    "load_prompts_from_jsonl",
    "Sample",
    "RolloutConfig",
    "RolloutBatch",
    "TrainingConfig",
]
