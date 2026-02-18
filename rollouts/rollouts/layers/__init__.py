"""Shared neural network primitives.

Pure functions for use by both pretrain and inference.
"""

from .norm import rms_norm
from .rope import apply_rotary_pos_emb, compute_rope_embeddings, rotate_half

__all__ = [
    "rms_norm",
    "rotate_half",
    "apply_rotary_pos_emb",
    "compute_rope_embeddings",
]
