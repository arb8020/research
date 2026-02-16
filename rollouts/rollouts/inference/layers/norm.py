"""Normalization layers for inference."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight

    Used by Llama, Qwen, and most modern LLMs.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization."""
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight).to(dtype)

    def forward_with_residual(self, x: Tensor, residual: Tensor | None) -> tuple[Tensor, Tensor]:
        """Apply RMS norm and update residual.

        This is the fused version used in transformer blocks:
        1. Add x to residual (if residual exists)
        2. Normalize
        3. Return (normalized, new_residual)
        """
        if residual is not None:
            x = x + residual
        residual = x
        return self.forward(x), residual
