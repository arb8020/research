#!/usr/bin/env python3
"""Model configuration for GPT-2."""

from dataclasses import dataclass
from typing import Optional
from jax import Array


@dataclass(frozen=True)
class GPT2Config:
    """Configuration for GPT-2 model."""
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_positions: int = 1024
    layer_norm_epsilon: float = 1e-5
    use_cache: bool = True
    training: bool = False
    freqs_cis: Optional[Array] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
