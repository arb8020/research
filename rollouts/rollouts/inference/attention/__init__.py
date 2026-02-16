"""Attention backends for inference.

Provides pluggable attention implementations:
- FlashAttention: fast fused attention with paged KV cache
- Reference: simple PyTorch implementation for testing
"""

from .backend import AttentionBackend, AttentionMetadata
from .flash import FlashAttentionBackend
from .reference import ReferenceAttentionBackend

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "FlashAttentionBackend",
    "ReferenceAttentionBackend",
]
