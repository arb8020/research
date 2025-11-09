"""Training system for rollouts.

Functional, composable training infrastructure inspired by SLIME's architecture
and Casey Muratori's API design principles.

Key design principles:
- Minimal stateful components (only DataBuffer)
- Pure functions for all transformations
- User-provided functions (SLIME-style)
- No hidden coupling or retention
- Explicit control flow (no callbacks)
"""

from rollouts.training.data_buffer import DataBuffer, load_prompts_from_jsonl

__all__ = [
    "DataBuffer",
    "load_prompts_from_jsonl",
]
