"""Rotary Position Embedding (RoPE) for inference.

RoPE applies rotation to Q and K based on position, enabling
relative position encoding without explicit position embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class RotaryConfig:
    """Configuration for rotary embeddings."""

    head_dim: int
    rotary_dim: int  # Usually == head_dim, but can be partial
    max_position: int = 8192
    base: float = 10000.0
    # Scaling for extended context (e.g., Llama 3.1)
    scaling_type: str | None = None
    scaling_factor: float = 1.0


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding.

    Precomputes cos/sin tables for positions [0, max_position).
    Apply with apply_rotary_pos_emb().
    """

    def __init__(self, config: RotaryConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.rotary_dim = config.rotary_dim

        # Compute inverse frequencies
        inv_freq = self._compute_inv_freq(config)
        self.register_buffer("inv_freq", inv_freq.to(device))

        # Precompute cos/sin for all positions
        self._precompute_cache(config.max_position, device)

    def _compute_inv_freq(self, config: RotaryConfig) -> Tensor:
        """Compute inverse frequencies for RoPE."""
        # Standard RoPE: inv_freq = 1 / (base ^ (2i / dim))
        dim = config.rotary_dim
        inv_freq = 1.0 / (config.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        # Apply scaling if configured
        if config.scaling_type == "linear":
            inv_freq = inv_freq / config.scaling_factor
        elif config.scaling_type == "dynamic":
            # Dynamic NTK scaling (used by some models)
            base = config.base * (
                (config.scaling_factor * config.max_position / config.max_position)
                ** (dim / (dim - 2))
            )
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        return inv_freq

    def _precompute_cache(self, max_position: int, device: torch.device) -> None:
        """Precompute cos/sin cache for all positions."""
        positions = torch.arange(max_position, device=device, dtype=torch.float32)
        # [max_position, rotary_dim/2]
        freqs = torch.outer(positions, self.inv_freq)
        # [max_position, rotary_dim]
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cache", emb.cos().to(torch.bfloat16))
        self.register_buffer("sin_cache", emb.sin().to(torch.bfloat16))

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Get cos/sin for given positions.

        Args:
            positions: Position indices, shape [num_tokens]

        Returns:
            (cos, sin) each shape [num_tokens, rotary_dim]
        """
        cos = self.cos_cache[positions]
        sin = self.sin_cache[positions]
        return cos, sin


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embedding to Q and K.

    Args:
        q: Query tensor, shape [num_tokens, num_heads, head_dim]
        k: Key tensor, shape [num_tokens, num_kv_heads, head_dim]
        cos: Cosine values, shape [num_tokens, rotary_dim]
        sin: Sine values, shape [num_tokens, rotary_dim]

    Returns:
        (q_rotated, k_rotated) with same shapes as inputs
    """
    rotary_dim = cos.shape[-1]

    # Split into rotary and pass-through parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Reshape cos/sin for broadcasting: [num_tokens, 1, rotary_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Apply rotation
    q_rotated = _rotate_half(q_rot, cos, sin)
    k_rotated = _rotate_half(k_rot, cos, sin)

    # Concatenate with pass-through
    if q_pass.shape[-1] > 0:
        q_rotated = torch.cat([q_rotated, q_pass], dim=-1)
        k_rotated = torch.cat([k_rotated, k_pass], dim=-1)

    return q_rotated, k_rotated


def _rotate_half(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate using complex multiplication trick.

    x = [x1, x2]
    rotate(x) = [x1*cos - x2*sin, x2*cos + x1*sin]
    """
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat(
        [
            x1 * cos[..., : x1.shape[-1]] - x2 * sin[..., : x1.shape[-1]],
            x2 * cos[..., : x2.shape[-1]] + x1 * sin[..., : x2.shape[-1]],
        ],
        dim=-1,
    )
    return rotated
