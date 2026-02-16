"""Base model class for inference."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from ..attention.backend import AttentionBackend, AttentionMetadata


class BaseModel(Protocol):
    """Protocol for inference models.

    All models must implement forward() that takes batch info explicitly.
    No global context — everything passed as arguments.
    """

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        """Run forward pass and return logits.

        Args:
            input_ids: Token IDs, shape [total_tokens]
            positions: Position indices, shape [total_tokens]
            attn_backend: Attention backend for computing attention
            attn_metadata: Metadata (sequence lengths, page table)
            out_loc: KV cache slots for new tokens, shape [total_tokens]

        Returns:
            Logits tensor, shape [total_tokens, vocab_size]
        """
        ...
