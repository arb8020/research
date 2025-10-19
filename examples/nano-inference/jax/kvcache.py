#!/usr/bin/env python3
"""KV cache management for efficient inference.

Demonstrates the functional programming pattern for "updating" immutable JAX arrays.
JAX arrays are immutable, so all updates create new arrays and return a new KVCache.

Shape suffixes (Shazeer convention):
  L: num_layers
  T: seq_len (time/sequence dimension)
  K: num_kv_heads (key-value heads)
  D: head_dim (d_model per head)
"""

import dataclasses
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class KVCache:
    """Container for key and value caches."""
    k_LTKD: jax.Array  # (num_layers, max_seq_len, num_kv_heads, head_dim)
    v_LTKD: jax.Array  # (num_layers, max_seq_len, num_kv_heads, head_dim)


def init_kv_cache(n_layers: int, max_seq_len: int, n_kv_heads: int, head_dim: int, dtype=jnp.float32) -> KVCache:
    """Initialize empty KV cache with zero-filled arrays."""
    assert n_layers > 0
    assert max_seq_len > 0
    assert n_kv_heads > 0
    assert head_dim > 0

    shape_LTKD = (n_layers, max_seq_len, n_kv_heads, head_dim)
    return KVCache(
        k_LTKD=jnp.zeros(shape_LTKD, dtype=dtype),
        v_LTKD=jnp.zeros(shape_LTKD, dtype=dtype),
    )


def update_kv_cache(cache: KVCache, layer_idx: int, start_pos: int, k_TKD: jax.Array, v_TKD: jax.Array) -> KVCache:
    """Update KV cache with new key/value tensors.

    JAX arrays are immutable - this creates new arrays and returns a new KVCache.
    Input shapes: k_TKD, v_TKD are (seq_len, num_kv_heads, head_dim)
    """
    assert layer_idx >= 0
    assert layer_idx < cache.k_LTKD.shape[0]
    assert start_pos >= 0
    assert k_TKD.shape == v_TKD.shape
    assert len(k_TKD.shape) == 3

    seq_len = k_TKD.shape[0]
    max_seq_len = cache.k_LTKD.shape[1]
    assert start_pos + seq_len <= max_seq_len

    # JAX functional array update: .at[].set() returns NEW array
    new_k_LTKD = cache.k_LTKD.at[layer_idx, start_pos:start_pos + seq_len].set(k_TKD)
    new_v_LTKD = cache.v_LTKD.at[layer_idx, start_pos:start_pos + seq_len].set(v_TKD)

    return KVCache(k_LTKD=new_k_LTKD, v_LTKD=new_v_LTKD)


def get_kv_cache(cache: KVCache, layer_idx: int, seq_len: int) -> tuple[jax.Array, jax.Array]:
    """Retrieve cached keys and values for a layer.

    Returns: (k_TKD, v_TKD) each of shape (seq_len, num_kv_heads, head_dim)
    """
    assert layer_idx >= 0
    assert layer_idx < cache.k_LTKD.shape[0]
    assert seq_len > 0
    assert seq_len <= cache.k_LTKD.shape[1]

    k_TKD = cache.k_LTKD[layer_idx, :seq_len]
    v_TKD = cache.v_LTKD[layer_idx, :seq_len]
    return k_TKD, v_TKD


if __name__ == "__main__":
    print("Creating KV cache...")
    cache = init_kv_cache(n_layers=12, max_seq_len=2048, n_kv_heads=8, head_dim=64)
    print(f"  k_LTKD shape: {cache.k_LTKD.shape}")
    print(f"  v_LTKD shape: {cache.v_LTKD.shape}")

    # Prefill: add keys/values for first 10 tokens
    print("\nPrefill: adding tokens 0-9 to layer 0")
    prefill_len = 10
    k_TKD = jnp.ones((prefill_len, 8, 64))
    v_TKD = jnp.ones((prefill_len, 8, 64)) * 2
    cache = update_kv_cache(cache, layer_idx=0, start_pos=0, k_TKD=k_TKD, v_TKD=v_TKD)

    k_retrieved_TKD, v_retrieved_TKD = get_kv_cache(cache, layer_idx=0, seq_len=prefill_len)
    assert k_retrieved_TKD.shape == (prefill_len, 8, 64)
    assert v_retrieved_TKD.shape == (prefill_len, 8, 64)
    assert k_retrieved_TKD[0, 0, 0] == 1.0
    assert v_retrieved_TKD[0, 0, 0] == 2.0
    print(f"  Retrieved shape: {k_retrieved_TKD.shape}")

    # Generation: add one token at a time
    print("\nGeneration: adding tokens 10-12 one at a time")
    for pos in range(10, 13):
        k_new_TKD = jnp.ones((1, 8, 64)) * pos
        v_new_TKD = jnp.ones((1, 8, 64)) * pos * 2
        cache = update_kv_cache(cache, layer_idx=0, start_pos=pos, k_TKD=k_new_TKD, v_TKD=v_new_TKD)
        print(f"  Token {pos} added")

    k_all_TKD, v_all_TKD = get_kv_cache(cache, layer_idx=0, seq_len=13)
    assert k_all_TKD[10, 0, 0] == 10.0
    assert v_all_TKD[12, 0, 0] == 24.0
    print(f"\nVerified: k[10]={k_all_TKD[10, 0, 0]}, v[12]={v_all_TKD[12, 0, 0]}")
    print("All tests passed!")
