"""Reference attention implementation.

Simple PyTorch implementation for correctness testing.
Not optimized — use FlashAttention for production.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .backend import AttentionMetadata


class ReferenceAttentionBackend:
    """Reference attention using standard PyTorch ops.

    This implementation:
    - Uses our KVCachePool for storage
    - Computes attention via matmul (no FlashAttention)
    - Handles variable sequence lengths via looping

    Useful for:
    - Numerical correctness testing
    - Running without flash-attn installed
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
        self.scale = 1.0 / math.sqrt(head_dim)

        # GQA: how many query heads per KV head
        assert num_q_heads % num_kv_heads == 0
        self.num_q_per_kv = num_q_heads // num_kv_heads

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
        """Compute attention with cached K,V.

        1. Store new K,V to cache
        2. For each sequence, gather all K,V from cache
        3. Compute attention output
        """
        # Store new K,V
        self.store_kv(k, v, layer_idx, out_loc)

        # Process each sequence separately
        outputs = []
        q_offset = 0

        for seq_idx in range(metadata.batch_size):
            # Sequence lengths
            q_len = int(metadata.cu_seqlens_q[seq_idx + 1] - metadata.cu_seqlens_q[seq_idx])
            kv_len = int(metadata.cache_seqlens[seq_idx].item())

            # Get query for this sequence
            seq_q = q[q_offset : q_offset + q_len]  # [q_len, num_q_heads, head_dim]

            # Get all K,V from cache using page table
            slots = metadata.page_table[seq_idx, :kv_len]  # [kv_len]
            seq_k = self.k_cache[layer_idx, slots]  # [kv_len, num_kv_heads, head_dim]
            seq_v = self.v_cache[layer_idx, slots]  # [kv_len, num_kv_heads, head_dim]

            # Compute attention for this sequence
            seq_out = self._attention(seq_q, seq_k, seq_v, q_len, kv_len)
            outputs.append(seq_out)

            q_offset += q_len

        return torch.cat(outputs, dim=0)

    def _attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_len: int,
        kv_len: int,
    ) -> Tensor:
        """Standard scaled dot-product attention with GQA and causal mask.

        Args:
            q: [q_len, num_q_heads, head_dim]
            k: [kv_len, num_kv_heads, head_dim]
            v: [kv_len, num_kv_heads, head_dim]

        Returns:
            [q_len, num_q_heads, head_dim]
        """
        # Expand K,V for GQA: [kv_len, num_kv_heads, head_dim] -> [kv_len, num_q_heads, head_dim]
        if self.num_q_per_kv > 1:
            k = k.repeat_interleave(self.num_q_per_kv, dim=1)
            v = v.repeat_interleave(self.num_q_per_kv, dim=1)

        # Reshape for batched matmul
        # q: [num_q_heads, q_len, head_dim]
        # k: [num_q_heads, kv_len, head_dim]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # Attention scores: [num_q_heads, q_len, kv_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask: query at position i can only attend to keys at positions <= i
        # For prefill: queries are at positions [kv_len - q_len, kv_len)
        # Key positions are [0, kv_len)
        q_positions = torch.arange(kv_len - q_len, kv_len, device=q.device)
        k_positions = torch.arange(kv_len, device=q.device)
        mask = q_positions.unsqueeze(1) < k_positions.unsqueeze(0)  # [q_len, kv_len]
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        # Softmax and weighted sum
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # [num_q_heads, q_len, head_dim]

        # Reshape back: [q_len, num_q_heads, head_dim]
        return out.transpose(0, 1)
