"""Attention nn.Module wrapper - thin layer for PyTorch compatibility."""

import torch.nn as nn
from torch import Tensor

from rollouts.inference.attention.protocol import AttentionBackend
from rollouts.inference.types import InferenceContext


class Attention(nn.Module):
    """Thin nn.Module wrapper around AttentionBackend.

    Why nn.Module?
    - PyTorch expects nn.Module for model layers
    - Enables standard model.to(device), model.eval(), etc.
    - Plays nice with torch.compile

    Why thin?
    - All compute logic lives in backend
    - This just holds layer_idx and delegates to backend
    - No trainable parameters (no weights here)
    """

    def __init__(self, layer_idx: int, backend: AttentionBackend) -> None:
        """Initialize attention layer.

        Args:
            layer_idx: Which layer this is (for indexing into backend's cache)
            backend: Shared backend that owns KV cache
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.backend = backend

    def forward(
        self,
        q: Tensor,  # [batch, seq_len, num_heads, head_dim]
        k: Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
        v: Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
        ctx: InferenceContext,
    ) -> Tensor:
        """Compute attention. Delegates to backend.

        Args:
            q, k, v: Query, key, value tensors
            ctx: Inference context with cache slot mapping

        Returns:
            Attention output: [batch, seq_len, num_heads, head_dim]
        """
        return self.backend.forward(q, k, v, self.layer_idx, ctx)
