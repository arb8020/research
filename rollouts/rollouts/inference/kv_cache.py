"""KV cache for inference engine.

Design principles:
- KVCachePool is a class because it owns GPU memory
- Pure functions for cache operations where possible
- Immutable view types for cache state per-request

The cache stores K,V tensors in a flat pool indexed by slot.
Each slot holds one token's K,V across all layers.

Layout:
- k_cache: [num_layers, num_slots, num_heads, head_dim]
- v_cache: [num_layers, num_slots, num_heads, head_dim]

For each request, we track which slots hold its K,V values.
`out_loc` from scheduler tells us where to write new K,V.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CacheConfig:
    """KV cache configuration. Immutable."""

    num_layers: int
    num_heads: int
    head_dim: int
    num_slots: int  # Total slots in pool
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        assert self.num_layers > 0
        assert self.num_heads > 0
        assert self.head_dim > 0
        assert self.num_slots > 0


class KVCachePool:
    """Pool of K,V cache slots.

    Why a class?
    - Owns GPU memory (the K,V tensors)
    - Stateful: tracks which slots are allocated

    Slots are allocated per-token. A sequence of N tokens uses N slots.
    """

    def __init__(self, config: CacheConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        # Allocate K,V tensors: [num_layers, num_slots, num_heads, head_dim]
        shape = (config.num_layers, config.num_slots, config.num_heads, config.head_dim)
        self.k_cache = torch.zeros(shape, dtype=config.dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=config.dtype, device=device)

        # Slot allocation: simple bump allocator (reset per generation)
        self.next_slot = 0

    @property
    def num_free_slots(self) -> int:
        """Number of free slots available."""
        return self.config.num_slots - self.next_slot

    def allocate_slots(self, n: int) -> Tensor:
        """Allocate n contiguous slots. Returns slot indices.

        Args:
            n: Number of slots to allocate

        Returns:
            Tensor of slot indices, shape [n], dtype int32
        """
        assert n > 0
        assert n <= self.num_free_slots, f"not enough slots: need {n}, have {self.num_free_slots}"

        slots = torch.arange(
            self.next_slot,
            self.next_slot + n,
            dtype=torch.int32,
            device=self.device,
        )
        self.next_slot += n
        return slots

    def write_kv(
        self,
        layer_idx: int,
        slots: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> None:
        """Write K,V to cache slots.

        Args:
            layer_idx: Which layer
            slots: Slot indices, shape [num_tokens]
            k: Key tensor, shape [num_tokens, num_heads, head_dim]
            v: Value tensor, shape [num_tokens, num_heads, head_dim]
        """
        assert 0 <= layer_idx < self.config.num_layers
        assert len(slots) == len(k) == len(v)

        self.k_cache[layer_idx, slots] = k
        self.v_cache[layer_idx, slots] = v

    def read_kv(
        self,
        layer_idx: int,
        slots: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Read K,V from cache slots.

        Args:
            layer_idx: Which layer
            slots: Slot indices, shape [num_tokens]

        Returns:
            (k, v) tensors, each shape [num_tokens, num_heads, head_dim]
        """
        assert 0 <= layer_idx < self.config.num_layers

        k = self.k_cache[layer_idx, slots]
        v = self.v_cache[layer_idx, slots]
        return k, v

    def reset(self) -> None:
        """Reset allocator (free all slots). Does not zero tensors."""
        self.next_slot = 0


@dataclass(frozen=True)
class RequestCache:
    """Cache state for a single request. Immutable.

    Tracks which slots hold this request's K,V values.
    """

    uid: int
    slots: Tensor  # Slot indices for this request's cached tokens, shape [cached_len]

    def __post_init__(self) -> None:
        assert self.slots.dtype == torch.int32
        assert self.slots.device.type in ("cpu", "cuda")

    @property
    def cached_len(self) -> int:
        """Number of tokens with K,V in cache."""
        return len(self.slots)


def empty_request_cache(uid: int, device: torch.device) -> RequestCache:
    """Create empty cache state for a new request."""
    return RequestCache(
        uid=uid,
        slots=torch.tensor([], dtype=torch.int32, device=device),
    )


def extend_request_cache(cache: RequestCache, new_slots: Tensor) -> RequestCache:
    """Return new RequestCache with slots appended.

    Called after forward pass writes K,V to new_slots.
    """
    assert new_slots.dtype == torch.int32

    combined = torch.cat([cache.slots.to(new_slots.device), new_slots])
    return RequestCache(uid=cache.uid, slots=combined)


# ═══════════════════════════════════════════════════════════════════════════════
# HUGGINGFACE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


def gather_past_key_values(
    pool: KVCachePool,
    slots: Tensor,
) -> tuple[tuple[Tensor, Tensor], ...]:
    """Gather past_key_values in HuggingFace format.

    HuggingFace expects: tuple of (key, value) per layer
    Each key/value: [batch_size, num_heads, seq_len, head_dim]

    We're processing one request at a time, so batch_size=1.
    """
    if len(slots) == 0:
        return None

    past_kv: list[tuple[Tensor, Tensor]] = []

    for layer_idx in range(pool.config.num_layers):
        k, v = pool.read_kv(layer_idx, slots)
        # k, v are [seq_len, num_heads, head_dim]
        # HF wants [batch, num_heads, seq_len, head_dim]
        k = k.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
        v = v.transpose(0, 1).unsqueeze(0)
        past_kv.append((k, v))

    return tuple(past_kv)


def store_new_key_values(
    pool: KVCachePool,
    slots: Tensor,
    past_key_values: tuple[tuple[Tensor, Tensor], ...],
    cached_len: int,
) -> None:
    """Store new K,V from HuggingFace format into pool.

    HuggingFace returns K,V for the FULL sequence (cached + new tokens).
    We extract only the new tokens (positions [cached_len:]) and store them.

    Args:
        pool: KV cache pool
        slots: Where to store NEW tokens' K,V, shape [num_new_tokens]
        past_key_values: HF format K,V for full sequence
        cached_len: Number of already-cached tokens (to skip)
    """
    assert past_key_values is not None
    assert len(past_key_values) == pool.config.num_layers

    for layer_idx, (k, v) in enumerate(past_key_values):
        # k, v are [1, num_heads, full_seq_len, head_dim]
        # Extract new tokens only: [cached_len:]
        k_new = k[:, :, cached_len:, :]  # [1, num_heads, num_new, head_dim]
        v_new = v[:, :, cached_len:, :]

        # Reshape for storage: [num_new, num_heads, head_dim]
        k_new = k_new.squeeze(0).transpose(0, 1)
        v_new = v_new.squeeze(0).transpose(0, 1)

        assert len(k_new) == len(slots), f"mismatch: {len(k_new)} new tokens, {len(slots)} slots"

        pool.write_kv(layer_idx, slots, k_new, v_new)
