#!/usr/bin/env python3
"""KV cache management for efficient inference.

Demonstrates the functional programming pattern for "updating" immutable JAX arrays.
JAX arrays are immutable, so all updates create new arrays and return a new KVCache.

Shape suffixes (Shazeer convention):
  L: num_layers
  T: seq_len (time/sequence dimension)
  K: num_kv_heads (key-value heads)
  H: head_dim (d_model // n_heads, dimension per attention head)
"""

import dataclasses

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class KVCache:
    """Container for key and value caches."""
    k_LTKH: jax.Array  # (num_layers, max_seq_len, num_kv_heads, head_dim)
    v_LTKH: jax.Array  # (num_layers, max_seq_len, num_kv_heads, head_dim)


def init_kv_cache(n_layers: int, max_seq_len: int, n_kv_heads: int, head_dim: int, dtype=jnp.float32) -> KVCache:
    """Initialize empty KV cache with zero-filled arrays."""
    assert n_layers > 0
    assert max_seq_len > 0
    assert n_kv_heads > 0
    assert head_dim > 0

    shape_LTKH = (n_layers, max_seq_len, n_kv_heads, head_dim)
    return KVCache(
        k_LTKH=jnp.zeros(shape_LTKH, dtype=dtype),
        v_LTKH=jnp.zeros(shape_LTKH, dtype=dtype),
    )


def update_kv_cache(cache: KVCache, layer_idx: int, start_pos: int, k_TKH: jax.Array, v_TKH: jax.Array) -> KVCache:
    """Update KV cache with new key/value tensors.

    JAX arrays are immutable - this creates new arrays and returns a new KVCache.
    Input shapes: k_TKH, v_TKH are (seq_len, num_kv_heads, head_dim)
    """
    assert layer_idx >= 0
    assert layer_idx < cache.k_LTKH.shape[0]
    assert start_pos >= 0
    assert k_TKH.shape == v_TKH.shape
    assert len(k_TKH.shape) == 3

    seq_len = k_TKH.shape[0]
    max_seq_len = cache.k_LTKH.shape[1]
    assert start_pos + seq_len <= max_seq_len

    # JAX functional array update: .at[].set() returns NEW array
    new_k_LTKH = cache.k_LTKH.at[layer_idx, start_pos:start_pos + seq_len].set(k_TKH)
    new_v_LTKH = cache.v_LTKH.at[layer_idx, start_pos:start_pos + seq_len].set(v_TKH)

    return KVCache(k_LTKH=new_k_LTKH, v_LTKH=new_v_LTKH)


def get_kv_cache(cache: KVCache, layer_idx: int, seq_len: int) -> tuple[jax.Array, jax.Array]:
    """Retrieve cached keys and values for a layer.

    Returns: (k_TKH, v_TKH) each of shape (seq_len, num_kv_heads, head_dim)
    """
    assert layer_idx >= 0
    assert layer_idx < cache.k_LTKH.shape[0]
    assert seq_len > 0
    assert seq_len <= cache.k_LTKH.shape[1]

    k_TKH = cache.k_LTKH[layer_idx, :seq_len]
    v_TKH = cache.v_LTKH[layer_idx, :seq_len]
    return k_TKH, v_TKH


if __name__ == "__main__":
    print("Creating KV cache...")
    cache = init_kv_cache(n_layers=12, max_seq_len=2048, n_kv_heads=8, head_dim=64)
    print(f"  k_LTKH shape: {cache.k_LTKH.shape}")
    print(f"  v_LTKH shape: {cache.v_LTKH.shape}")

    # Prefill: add keys/values for first 10 tokens
    print("\nPrefill: adding tokens 0-9 to layer 0")
    prefill_len = 10
    k_TKH = jnp.ones((prefill_len, 8, 64))
    v_TKH = jnp.ones((prefill_len, 8, 64)) * 2
    cache = update_kv_cache(cache, layer_idx=0, start_pos=0, k_TKH=k_TKH, v_TKH=v_TKH)

    k_retrieved_TKH, v_retrieved_TKH = get_kv_cache(cache, layer_idx=0, seq_len=prefill_len)
    assert k_retrieved_TKH.shape == (prefill_len, 8, 64)
    assert v_retrieved_TKH.shape == (prefill_len, 8, 64)
    assert k_retrieved_TKH[0, 0, 0] == 1.0
    assert v_retrieved_TKH[0, 0, 0] == 2.0
    print(f"  Retrieved shape: {k_retrieved_TKH.shape}")

    # Generation: add one token at a time
    print("\nGeneration: adding tokens 10-12 one at a time")
    for pos in range(10, 13):
        k_new_TKH = jnp.ones((1, 8, 64)) * pos
        v_new_TKH = jnp.ones((1, 8, 64)) * pos * 2
        cache = update_kv_cache(cache, layer_idx=0, start_pos=pos, k_TKH=k_new_TKH, v_TKH=v_new_TKH)
        print(f"  Token {pos} added")

    k_all_TKH, v_all_TKH = get_kv_cache(cache, layer_idx=0, seq_len=13)
    assert k_all_TKH[10, 0, 0] == 10.0
    assert v_all_TKH[12, 0, 0] == 24.0
    print(f"\nVerified: k[10]={k_all_TKH[10, 0, 0]}, v[12]={v_all_TKH[12, 0, 0]}")
    print("All tests passed!")
