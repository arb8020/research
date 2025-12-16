"""Cache and attention configuration - frozen dataclass."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for KV cache and attention. Immutable after creation.

    Passed at backend initialization. Defines cache tensor dimensions
    and attention behavior (e.g., sliding window).

    Why frozen?
    - Cache dimensions don't change after init
    - Hashable for debugging/logging
    - Thread-safe
    """

    # Model dimensions
    num_layers: int
    num_kv_heads: int
    head_dim: int

    # Cache dimensions
    num_blocks: int = 1024
    block_size: int = 16

    # Attention configuration
    sliding_window: int | None = None  # None = full attention, else window size

    # Hardware
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"

    @property
    def total_slots(self) -> int:
        """Total cache slots (num_blocks * block_size)."""
        return self.num_blocks * self.block_size

    @property
    def cache_shape(self) -> tuple[int, int, int, int]:
        """Shape of each cache tensor: [num_layers, total_slots, num_kv_heads, head_dim]."""
        return (self.num_layers, self.total_slots, self.num_kv_heads, self.head_dim)

    def __hash__(self) -> int:
        # dtype isn't hashable by default, convert to string
        return hash((
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.num_blocks,
            self.block_size,
            self.sliding_window,
            str(self.dtype),
            self.device,
        ))
