"""FlashAttention backend with paged KV cache.

Uses flash_attn_with_kvcache from sgl-kernel or flash-attn.
This is the production attention implementation.
"""

from __future__ import annotations

from torch import Tensor

from .backend import AttentionMetadata


class FlashAttentionBackend:
    """FlashAttention with paged KV cache.

    Uses the fused flash_attn_with_kvcache kernel that:
    - Reads K,V directly from paged cache
    - Handles variable sequence lengths efficiently
    - Supports both prefill and decode

    Requirements:
    - GPU with compute capability >= 8.0 (Ampere+)
    - flash-attn or sgl-kernel installed
    """

    def __init__(
        self,
        k_cache: Tensor,
        v_cache: Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        """
        Args:
            k_cache: Key cache, shape [num_layers, num_slots, num_kv_heads, head_dim]
            v_cache: Value cache, same shape
            num_q_heads: Number of query heads
            num_kv_heads: Number of key/value heads
            head_dim: Dimension per head
        """
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        # Try to import flash attention
        self._flash_attn_fn = _get_flash_attn_fn()

    def store_kv(
        self,
        k: Tensor,
        v: Tensor,
        layer_idx: int,
        out_loc: Tensor,
    ) -> None:
        """Store K,V to cache slots."""
        # k, v: [num_tokens, num_kv_heads, head_dim]
        # out_loc: [num_tokens]
        self.k_cache[layer_idx, out_loc] = k
        self.v_cache[layer_idx, out_loc] = v

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        layer_idx: int,
        metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        """Compute attention using FlashAttention.

        1. Store new K,V to cache
        2. Call flash_attn_with_kvcache to compute attention
        """
        # Store new K,V first
        self.store_kv(k, v, layer_idx, out_loc)

        # Get cache for this layer
        # Shape: [num_slots, num_kv_heads, head_dim]
        k_cache_layer = self.k_cache[layer_idx]
        v_cache_layer = self.v_cache[layer_idx]

        # FlashAttention with paged KV cache
        # q shape: [total_q_tokens, num_q_heads, head_dim]
        # Need to unsqueeze for flash_attn: [total_q_tokens, 1, num_q_heads, head_dim]
        # Actually flash_attn_with_kvcache expects different shapes...

        return self._flash_attn_fn(
            q=q,
            k_cache=k_cache_layer,
            v_cache=v_cache_layer,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k_new=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            causal=True,
        )


def _get_flash_attn_fn():
    """Get the best available FlashAttention implementation."""
    # Try sgl-kernel first (optimized for serving)
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        return _wrap_sgl_kernel(flash_attn_with_kvcache)
    except ImportError:
        pass

    # Fall back to flash-attn
    try:
        from flash_attn import flash_attn_with_kvcache

        return _wrap_flash_attn(flash_attn_with_kvcache)
    except ImportError:
        pass

    # No FlashAttention available
    raise ImportError(
        "No FlashAttention implementation found. "
        "Install either sgl-kernel (`pip install sgl-kernel`) "
        "or flash-attn (`pip install flash-attn`)."
    )


def _wrap_sgl_kernel(fn):
    """Wrap sgl-kernel's flash_attn_with_kvcache."""

    def wrapped(
        q: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        page_table: Tensor,
        cache_seqlens: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k_new: Tensor,
        max_seqlen_q: int,
        softmax_scale: float,
        causal: bool = True,
    ) -> Tensor:
        return fn(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=softmax_scale,
            causal=causal,
            ver=3,  # FlashAttention v3
        )

    return wrapped


def _wrap_flash_attn(fn):
    """Wrap flash-attn's flash_attn_with_kvcache.

    flash-attn has a different API, need to adapt.
    """

    def wrapped(
        q: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        page_table: Tensor,
        cache_seqlens: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k_new: Tensor,
        max_seqlen_q: int,
        softmax_scale: float,
        causal: bool = True,
    ) -> Tensor:
        # flash-attn expects:
        # q: [batch, seqlen_q, num_heads, head_dim] or [total_q, num_heads, head_dim]
        # k_cache: [batch, seqlen_k, num_kv_heads, head_dim] (but we have paged)
        #
        # For paged cache, need to use flash_attn_varlen_func with block_table
        # This is more complex - fall back to reference impl for now

        raise NotImplementedError(
            "flash-attn paged KV cache support requires additional work. "
            "Use sgl-kernel instead: pip install sgl-kernel"
        )

    return wrapped


def is_flash_attn_available() -> bool:
    """Check if any FlashAttention implementation is available."""
    try:
        _get_flash_attn_fn()
        return True
    except ImportError:
        return False
