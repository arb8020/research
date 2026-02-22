"""Attention backends for inference.

Provides pluggable attention implementations:
- FlashAttention: fast fused attention with paged KV cache
- FlashInfer: optimized decode kernels (TODO)
- Reference: simple PyTorch implementation for testing
"""

from .backend import AttentionBackend, AttentionMetadata
from .flash import FlashAttentionBackend
from .flashinfer import FlashInferBackend, is_flashinfer_available
from .reference import ReferenceAttentionBackend

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "FlashAttentionBackend",
    "FlashInferBackend",
    "is_flashinfer_available",
    "ReferenceAttentionBackend",
]
