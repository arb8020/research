"""Activation functions for inference."""

from __future__ import annotations

import torch
from torch import Tensor


def silu_and_mul(x: Tensor) -> Tensor:
    """SiLU activation with gated multiplication.

    Input shape: [..., 2 * hidden_size]
    Output shape: [..., hidden_size]

    Splits input in half: gate, up
    Returns: silu(gate) * up
    """
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def gelu_and_mul(x: Tensor) -> Tensor:
    """GELU activation with gated multiplication.

    Input shape: [..., 2 * hidden_size]
    Output shape: [..., hidden_size]
    """
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.gelu(gate) * up
