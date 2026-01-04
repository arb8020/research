"""Attention with paged KV cache.

Architecture:
- CacheConfig: frozen dataclass with cache dimensions + sliding_window
- AttentionBackend: protocol for swappable implementations
- FlexAttentionBackend: PyTorch FlexAttention implementation
- Attention: thin nn.Module wrapper for PyTorch compatibility
- Masks: composable block masks for causal, sliding window, document attention

Why this structure (following vLLM/SGLang patterns)?
- Single cache allocation shared across all layers
- Easy to swap backends (FlexAttention -> FlashInfer)
- Clear separation: config vs compute vs PyTorch integration
"""

from ...inference.attention.config import CacheConfig
from ...inference.attention.flex_backend import FlexAttentionBackend
from ...inference.attention.layer import Attention
from ...inference.attention.mask import (
    create_attention_mask,
    create_causal_block_mask,
    create_sliding_window_causal_mask,
)
from ...inference.attention.protocol import AttentionBackend

__all__ = [
    "CacheConfig",
    "AttentionBackend",
    "FlexAttentionBackend",
    "Attention",
    "create_attention_mask",
    "create_causal_block_mask",
    "create_sliding_window_causal_mask",
]
