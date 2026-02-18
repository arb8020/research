"""Attention backend protocol.

Defines the interface that all attention implementations must follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


@dataclass
class AttentionMetadata:
    """Metadata for attention computation.

    This is the "flattened batch" view where all sequences are concatenated.
    Variable-length sequences are handled via cu_seqlens (cumulative sequence lengths).

    Fields:
        cu_seqlens_q: Cumulative query sequence lengths, shape [batch_size + 1]
            Example: [0, 5, 8, 15] means 3 sequences with lengths 5, 3, 7
        cu_seqlens_k: Cumulative key sequence lengths (total including cached)
        cache_seqlens: Number of cached tokens per sequence, shape [batch_size]
        max_seqlen_q: Maximum query sequence length in batch
        max_seqlen_k: Maximum key sequence length in batch
        page_table: Maps (seq_idx, position) -> cache slot, shape [batch_size, max_seqlen_k]
    """

    cu_seqlens_q: Tensor  # [batch_size + 1], int32
    cu_seqlens_k: Tensor  # [batch_size + 1], int32
    cache_seqlens: Tensor  # [batch_size], int32
    max_seqlen_q: int
    max_seqlen_k: int
    page_table: Tensor  # [batch_size, max_seqlen_k], int32

    def __post_init__(self) -> None:
        batch_size = len(self.cache_seqlens)
        assert len(self.cu_seqlens_q) == batch_size + 1
        assert len(self.cu_seqlens_k) == batch_size + 1
        assert self.page_table.shape[0] == batch_size

    @property
    def batch_size(self) -> int:
        return len(self.cache_seqlens)

    @property
    def total_q_tokens(self) -> int:
        return int(self.cu_seqlens_q[-1].item())


class AttentionBackend(Protocol):
    """Protocol for attention backends.

    Each backend handles:
    1. Storing K,V to cache
    2. Computing attention output

    The backend is stateless — all state is in the KV cache and metadata.
    """

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        layer_idx: int,
        metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        """Compute attention and store K,V to cache.

        Args:
            q: Query tensor, shape [total_q_tokens, num_q_heads, head_dim]
            k: Key tensor, shape [total_q_tokens, num_kv_heads, head_dim]
            v: Value tensor, shape [total_q_tokens, num_kv_heads, head_dim]
            layer_idx: Which layer (for KV cache indexing)
            metadata: Attention metadata (sequence lengths, page table)
            out_loc: Cache slots for new K,V, shape [total_q_tokens]

        Returns:
            Output tensor, shape [total_q_tokens, num_q_heads, head_dim]
        """
        ...

    def store_kv(
        self,
        k: Tensor,
        v: Tensor,
        layer_idx: int,
        out_loc: Tensor,
    ) -> None:
        """Store K,V to cache slots.

        Args:
            k: Key tensor, shape [num_tokens, num_kv_heads, head_dim]
            v: Value tensor, shape [num_tokens, num_kv_heads, head_dim]
            layer_idx: Which layer
            out_loc: Cache slots, shape [num_tokens]
        """
        ...


def build_attention_metadata(
    cached_lens: list[int],
    extend_lens: list[int],
    page_table: Tensor,
    device: torch.device,
    table_indices: list[int] | None = None,
) -> AttentionMetadata:
    """Build AttentionMetadata from per-request info.

    Args:
        cached_lens: Number of cached tokens per request
        extend_lens: Number of new tokens per request
        page_table: Full page table, shape [max_batch, max_seq_len]
        device: Target device
        table_indices: Per-request page table row indices in batch order.
            If None, uses rows [0, batch_size), for compatibility.

    Returns:
        AttentionMetadata ready for attention computation
    """
    batch_size = len(cached_lens)
    assert len(extend_lens) == batch_size

    # Query sequence lengths = extend lengths (new tokens)
    seqlens_q = extend_lens
    # Key sequence lengths = cached + extend (all tokens)
    seqlens_k = [c + e for c, e in zip(cached_lens, extend_lens, strict=False)]

    # Cumulative sequence lengths
    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor(seqlens_k, dtype=torch.int32, device=device)

    max_seqlen_q = max(seqlens_q) if seqlens_q else 0
    max_seqlen_k = max(seqlens_k) if seqlens_k else 0

    if table_indices is None:
        batch_page_table = page_table[:batch_size, :max_seqlen_k]
    else:
        assert len(table_indices) == batch_size
        index_tensor = torch.tensor(table_indices, dtype=torch.long, device=page_table.device)
        batch_page_table = page_table.index_select(0, index_tensor)[:, :max_seqlen_k]

    return AttentionMetadata(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        page_table=batch_page_table.contiguous(),
    )
