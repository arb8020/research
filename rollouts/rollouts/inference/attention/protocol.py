"""AttentionBackend protocol - interface for swappable implementations."""

from typing import Protocol

from torch import Tensor

from ...inference.types import InferenceContext


class AttentionBackend(Protocol):
    """Protocol for attention backends.

    Why a protocol?
    - Structural subtyping: implementations don't need to inherit
    - Easy to swap: FlexAttention -> FlashInfer -> custom CUDA
    - Testable: mock implementations for unit tests

    Backends own the KV cache tensors and compute attention.
    The forward() method takes layer_idx to index into the shared cache.
    """

    def forward(
        self,
        q: Tensor,  # [batch, seq_len, num_heads, head_dim]
        k: Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
        v: Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
        layer_idx: int,
        ctx: InferenceContext,
    ) -> Tensor:
        """Compute attention with KV cache.

        Args:
            q, k, v: Query, key, value tensors
            layer_idx: Which layer (for indexing into shared cache)
            ctx: Inference context with cache slot mapping

        Returns:
            Attention output: [batch, seq_len, num_heads, head_dim]
        """
        ...
