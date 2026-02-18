"""Normalization layers."""

import torch
from torch import Tensor


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight.

    Args:
        x: Input tensor [..., dim]
        weight: Scale parameter [dim]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor with same shape as x
    """
    assert x.ndim >= 2, f"x must be at least 2D, got {x.shape}"
    assert weight.ndim == 1, f"weight must be 1D, got {weight.shape}"
    assert x.shape[-1] == weight.shape[0], (
        f"dim mismatch: x={x.shape[-1]}, weight={weight.shape[0]}"
    )

    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return weight * x_normed.to(x.dtype)
