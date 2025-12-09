"""Attention with paged KV cache.

Architecture:
- CacheConfig: frozen dataclass with cache dimensions
- AttentionBackend: protocol for swappable implementations
- FlexAttentionBackend: PyTorch FlexAttention implementation
- Attention: thin nn.Module wrapper for PyTorch compatibility

Why this structure (following vLLM/SGLang patterns)?
- Single cache allocation shared across all layers
- Easy to swap backends (FlexAttention -> FlashInfer)
- Clear separation: config vs compute vs PyTorch integration
"""

from rollouts.inference.attention.config import CacheConfig
from rollouts.inference.attention.protocol import AttentionBackend
from rollouts.inference.attention.flex_backend import FlexAttentionBackend
from rollouts.inference.attention.layer import Attention
from rollouts.inference.attention.mask import create_causal_block_mask

__all__ = [
    "CacheConfig",
    "AttentionBackend",
    "FlexAttentionBackend",
    "Attention",
    "create_causal_block_mask",
]
