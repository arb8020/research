# JetStream KV Cache Implementation Reference

Reference documentation for implementing KV caching in JAX, based on Google's JetStream inference engine.

## Overview

JetStream implements KV caching using a **two-phase architecture** (prefill + decode) with **functional cache threading** that fits JAX's immutable programming model.

**Key files in originals/JetStream:**
- `experimental/jax/inference/nn/attention.py` - Core attention with KV cache
- `experimental/jax/inference/runtime/kv_cache.py` - Cache storage management
- `experimental/jax/inference/model/llama.py` - Integration in model forward pass

## Core Data Structures

### KVCache (Simple Version)

```python
@dataclasses.dataclass
class KVCache:
    k: jax.Array  # shape: (num_kv_heads, num_pages, page_size, head_dim)
    v: jax.Array  # shape: (num_kv_heads, num_pages, page_size, head_dim)
```

**For learning/simple implementation:**
```python
@dataclasses.dataclass
class KVCache:
    k_LTKH: jax.Array  # (num_layers, max_seq_len, num_kv_heads, head_dim)
    v_LTKH: jax.Array  # (num_layers, max_seq_len, num_kv_heads, head_dim)
```

### AttentionMetadata

Controls routing between prefill and decode phases:

```python
@dataclasses.dataclass
class AttentionMetadata:
    # Prefill phase
    prefill_length: jax.Array          # actual prefill length without padding
    prefill_pos: jax.Array             # positions during prefill
    prefill_page_table: jax.Array      # maps logical to physical pages

    # Decode phase
    generate_pos: jax.Array            # current position per sequence
    generate_page_table: jax.Array     # page mapping for generation
```

**For simple implementation:** Just track `start_pos: int` and `seq_len: int`.

## Two-Phase Architecture

### Phase 1: Prefill (Process All Prompt Tokens)

**What happens:**
1. Compute Q, K, V for **entire prompt** at once
2. Write **all K/V** to cache
3. Compute attention over full prompt
4. Return output + populated cache

**Code pattern:**
```python
def _prefill(q, k, v, kv_cache, attn_metadata):
    # Input shapes:
    # q: (seq_len, num_heads, head_dim)
    # k, v: (seq_len, num_kv_heads, head_dim)

    # Step 1: Write K/V to cache
    kv_cache = _write_prefill_kv_to_kv_cache(
        k, v, kv_cache,
        attn_metadata.prefill_length,
        attn_metadata.prefill_page_table,
    )

    # Step 2: Compute attention using cached K/V
    output = prefill_attention_kernel(
        q,
        kv_cache.k,
        kv_cache.v,
        attn_metadata.prefill_length,
        attn_metadata.prefill_page_table,
    )

    return output, kv_cache
```

**Writing K/V to cache (prefill):**
```python
def _write_prefill_kv_to_kv_cache(k, v, kv_cache, unpadded_len, page_table):
    """
    k, v shape: (padded_seq_len, num_kv_heads, head_dim)
    Returns: updated kv_cache
    """
    # Transpose to (num_kv_heads, seq_len, head_dim)
    k = k.transpose((1, 0, 2))
    v = v.transpose((1, 0, 2))

    # Reshape into pages: (num_kv_heads, num_pages, page_size, head_dim)
    k = k.reshape((num_kv_heads, -1, page_size, head_dim))
    v = v.reshape((num_kv_heads, -1, page_size, head_dim))

    # Calculate active pages
    num_active_pages = math.ceil(unpadded_len / page_size)

    # Write each page to cache using page_table mapping
    for page_idx in range(num_active_pages):
        physical_page = page_table[page_idx]
        kv_cache.k = kv_cache.k.at[:, physical_page, :, :].set(k[:, page_idx, :, :])
        kv_cache.v = kv_cache.v.at[:, physical_page, :, :].set(v[:, page_idx, :, :])

    return kv_cache
```

### Phase 2: Decode (Incremental Token Generation)

**What happens:**
1. Compute Q, K, V for **only the new token**
2. Update cache with **just this token's K/V**
3. Attend to **all cached tokens** (full context)
4. Return output + updated cache

**Code pattern:**
```python
def _generate(q, k, v, kv_cache, attn_metadata):
    # Input shapes:
    # q: (batch_size, num_heads, head_dim)  [often batch_size=1]
    # k, v: (batch_size, num_kv_heads, head_dim)

    # Step 1: Update cache with new K/V at current position
    kv_cache = _write_generate_kv_to_kv_cache(
        k, v, kv_cache,
        attn_metadata.generate_pos,
        attn_metadata.generate_page_table,
    )

    # Step 2: Compute attention - query attends to ALL cached K/V
    output = decode_attention_kernel(
        q,
        kv_cache.k,
        kv_cache.v,
        attn_metadata.generate_pos,
        attn_metadata.generate_page_table,
    )

    return output, kv_cache
```

**Writing K/V to cache (decode):**
```python
def _write_generate_kv_to_kv_cache(k, v, kv_cache, pos, page_table):
    """
    k, v shape: (num_tokens, num_kv_heads, head_dim)  [typically num_tokens=1]
    pos: current position for each sequence in batch
    Returns: updated kv_cache
    """
    # Transpose to (num_kv_heads, num_tokens, head_dim)
    k = k.transpose((1, 0, 2))
    v = v.transpose((1, 0, 2))

    # Calculate which page and offset within page
    page_idx, offset = jnp.divmod(pos, page_size)

    # Look up physical page from page table
    physical_page = page_table[page_idx]

    # Update cache at [head, physical_page, offset, :]
    kv_cache.k = kv_cache.k.at[:, physical_page, offset, :].set(k)
    kv_cache.v = kv_cache.v.at[:, physical_page, offset, :].set(v)

    return kv_cache
```

## Functional Cache Threading Pattern

Cache flows through the network as input/output (immutable updates):

### At Attention Layer Level

```python
def attention_layer(
    hidden_states,
    positions,
    kv_cache: KVCache,
    attn_metadata: AttentionMetadata,
) -> tuple[jax.Array, KVCache]:

    # Project to Q, K, V
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    # Reshape to (num_tokens, num_heads, head_dim)
    q = q.reshape((q.shape[0], -1, self.head_dim))
    k = k.reshape((k.shape[0], -1, self.head_dim))
    v = v.reshape((v.shape[0], -1, self.head_dim))

    # Apply rotary embeddings (position-dependent)
    q = self.rotary_emb(q, positions, self.rope_theta)
    k = self.rotary_emb(k, positions, self.rope_theta)

    # Core attention with cache - returns updated cache
    output, kv_cache = self.attn(
        q, k, v,
        kv_cache,
        attn_metadata,
    )

    # Project output
    output = self.o_proj(output)

    return output, kv_cache  # Cache threads to next layer!
```

### At Decoder Layer Level

```python
def decoder_layer(
    hidden_states,
    positions,
    kv_cache: KVCache,
    attn_metadata: AttentionMetadata,
) -> tuple[jax.Array, KVCache]:

    # Pre-attention norm
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self-attention with cache update
    hidden_states, kv_cache = self.self_attn(
        hidden_states,
        positions,
        kv_cache,
        attn_metadata,
    )

    # Residual connection
    hidden_states = residual + hidden_states

    # FFN block
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.ffw(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, kv_cache
```

### At Model Level

```python
def model_forward(
    input_ids,
    positions,
    kv_caches: list[KVCache],  # One cache per layer
    attn_metadata: AttentionMetadata,
) -> tuple[jax.Array, list[KVCache]]:

    # Embeddings
    hidden_states = self.embed_tokens(input_ids)

    # Process through all decoder layers
    updated_kv_caches = []
    for layer_idx, layer in enumerate(self.layers):
        hidden_states, updated_cache = layer(
            hidden_states,
            positions,
            kv_caches[layer_idx],
            attn_metadata,
        )
        updated_kv_caches.append(updated_cache)

    # Final norm + LM head
    hidden_states = self.norm(hidden_states)
    logits = self.lm_head(hidden_states)

    return logits, updated_kv_caches
```

## Metadata-Driven Control Flow

Single attention function handles both phases:

```python
def attention_call(
    q, k, v,
    kv_cache: KVCache,
    attn_metadata: AttentionMetadata,
):
    # Route based on metadata shape
    if len(attn_metadata.generate_pos.shape) == 0:
        # Pure prefill: generate_pos is scalar (empty)
        return self._prefill(q, k, v, kv_cache, attn_metadata)

    elif len(attn_metadata.prefill_pos.shape) == 0:
        # Pure decode: prefill_pos is scalar (empty)
        return self._generate(q, k, v, kv_cache, attn_metadata)

    else:
        # Mixed: some sequences prefilling, some decoding
        return self._mixed_prefill_generate(q, k, v, kv_cache, attn_metadata)
```

**Simple alternative:** Pass explicit `use_cache: bool` and `start_pos: int`.

## Paged Memory Layout

JetStream uses PagedAttention for efficient memory management:

**Advantages:**
- Non-contiguous memory allocation
- Dynamic page allocation/deallocation
- Memory sharing between sequences (prefix caching)
- Efficient batching of variable-length sequences

**Cache shape:** `(num_kv_heads, num_pages, page_size, head_dim)`

**Page table mapping:**
- Logical page index → Physical page index
- Computed as: `page_idx, offset = divmod(position, page_size)`

**For simple implementation:**
- Use contiguous memory: `(num_layers, max_seq_len, num_kv_heads, head_dim)`
- Direct indexing: `cache[layer_idx, position, :, :]`
- Simpler but less memory-efficient

## Key Design Patterns

### Pattern 1: Position-Based Indexing

No need to track sequence length explicitly:

```python
# During decode, compute cache location from position
page_idx, offset = divmod(current_position, page_size)
physical_page = page_table[page_idx]

# Update cache at computed location
kv_cache.k = kv_cache.k.at[:, physical_page, offset, :].set(new_k)
```

### Pattern 2: Immutable Updates with .at[].set()

JAX arrays are immutable - create new arrays:

```python
# Wrong: This doesn't work in JAX
kv_cache.k[layer_idx, pos] = new_k

# Correct: Creates new array
kv_cache.k = kv_cache.k.at[layer_idx, pos].set(new_k)
```

### Pattern 3: Cache per Layer

Each transformer layer has its own K/V cache:

```python
# Initialize cache for all layers
kv_caches = [init_kv_cache(...) for _ in range(num_layers)]

# Thread through layers
for layer_idx in range(num_layers):
    hidden_states, kv_caches[layer_idx] = transformer_block(
        hidden_states,
        kv_caches[layer_idx],
        layer_idx
    )
```

### Pattern 4: Separate K and V

Always keep keys and values separate (don't concatenate):

```python
# Good: Separate arrays
@dataclasses.dataclass
class KVCache:
    k: jax.Array
    v: jax.Array

# Bad: Concatenated (harder to work with)
@dataclasses.dataclass
class KVCache:
    kv: jax.Array  # Concatenated K and V
```

## Attention Computation with Cache

### Without Cache (Your Current Implementation)

```python
# Compute full Q, K, V for all tokens
q_BNTH = compute_q(x_BTD)  # (batch, n_heads, seq_len, head_dim)
k_BNTH = compute_k(x_BTD)  # (batch, n_heads, seq_len, head_dim)
v_BNTH = compute_v(x_BTD)  # (batch, n_heads, seq_len, head_dim)

# Attention scores over full sequence
scores_BNTT = q_BNTH @ k_BNTH.transpose()  # (batch, n_heads, seq_len, seq_len)
attn_weights_BNTT = softmax(scores_BNTT)
output_BNTH = attn_weights_BNTT @ v_BNTH
```

### With Cache (Decode Phase)

```python
# Compute Q, K, V for ONLY new token
q_BN1H = compute_q(x_new_B1D)  # (batch, n_heads, 1, head_dim)
k_BN1H = compute_k(x_new_B1D)  # (batch, n_heads, 1, head_dim)
v_BN1H = compute_v(x_new_B1D)  # (batch, n_heads, 1, head_dim)

# Update cache with new K/V
kv_cache.k[:, :, pos, :] = k_BN1H
kv_cache.v[:, :, pos, :] = v_BN1H

# Retrieve ALL cached K/V (including new token)
k_all_BNTH = kv_cache.k[:, :, :pos+1, :]  # (batch, n_heads, pos+1, head_dim)
v_all_BNTH = kv_cache.v[:, :, :pos+1, :]  # (batch, n_heads, pos+1, head_dim)

# Attention: new query attends to all cached keys
scores_BN1T = q_BN1H @ k_all_BNTH.transpose()  # (batch, n_heads, 1, pos+1)
attn_weights_BN1T = softmax(scores_BN1T)
output_BN1H = attn_weights_BN1T @ v_all_BNTH
```

**Key difference:** Query is 1 token, but Key/Value are full context!

## Testing Example

### Prefill Test
```python
# Setup
prefill_len = 16
page_size = 8
num_pages = 2
cache = init_kv_cache(num_layers=12, num_kv_heads=8, head_dim=64, num_pages=num_pages, page_size=page_size)

# Compute Q, K, V for prompt
q = jnp.ones((prefill_len, num_heads, head_dim))
k = jnp.ones((prefill_len, num_kv_heads, head_dim))
v = jnp.ones((prefill_len, num_kv_heads, head_dim))

# Prefill attention
output, cache = prefill_attention(q, k, v, cache, page_table)

# Verify cache populated
assert cache.k.shape == (num_kv_heads, num_pages, page_size, head_dim)
assert not jnp.allclose(cache.k, 0)  # Should have data now
```

### Decode Test
```python
# Generate next token
current_pos = prefill_len
q_new = jnp.ones((1, num_heads, head_dim))
k_new = jnp.ones((1, num_kv_heads, head_dim))
v_new = jnp.ones((1, num_kv_heads, head_dim))

# Decode attention
output, cache = decode_attention(q_new, k_new, v_new, cache, current_pos, page_table)

# Verify cache updated
page_idx, offset = divmod(current_pos, page_size)
physical_page = page_table[page_idx]
assert jnp.allclose(cache.k[:, physical_page, offset, :], k_new[0])
```

## Summary: What You Need to Implement

For a **simple learning implementation** (without paging):

1. **Modify attention function signature:**
   ```python
   def multi_head_attention(
       x_BTD, weights, layer_idx, config,
       kv_cache: Optional[KVCache] = None,
       start_pos: int = 0
   ) -> tuple[Array, Optional[KVCache]]
   ```

2. **Cache update logic:**
   - If `kv_cache is None`: Compute fresh K/V (prefill mode)
   - Else: Compute K/V for new tokens, update cache, retrieve full cached K/V

3. **Thread cache through model:**
   ```python
   for layer_idx in range(n_layers):
       x_BTD, kv_cache = transformer_block(x_BTD, weights, layer_idx, config, kv_cache, start_pos)
   ```

4. **Test both phases:**
   - Prefill: Process prompt, populate cache
   - Decode: Generate tokens one-by-one using cache

## References

**JetStream source files:**
- `originals/JetStream/experimental/jax/inference/nn/attention.py:29-254`
- `originals/JetStream/experimental/jax/inference/runtime/kv_cache.py:31-167`
- `originals/JetStream/experimental/jax/inference/model/llama.py:150-185`
- `originals/JetStream/experimental/jax/inference/kernel/attention_ops.py`

**Your existing implementation:**
- `backends/jax/kvcache.py` - Already has functional cache structure!
- `backends/jax/layers.py:152` - `multi_head_attention()` needs cache integration
- `backends/jax/model.py:81` - `gpt2_forward()` needs cache threading
