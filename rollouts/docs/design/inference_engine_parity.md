# Inference Engine: mini-sglang Parity Plan

## Goal
Full parity with mini-sglang including:
- Numerical equivalence (same outputs given same inputs)
- Feature parity (all major features)
- Ability to run mini-sglang's test suite or equivalent

## Architecture Comparison

### mini-sglang Architecture
```
LLM (offline mode)
└── Scheduler
    ├── Engine
    │   ├── Model (custom Llama/Qwen impl)
    │   ├── KVCache (MHAKVCache)
    │   ├── AttnBackend (FlashAttention)
    │   ├── Sampler
    │   └── GraphRunner (CUDA graphs)
    ├── CacheManager (RadixCacheManager)
    ├── TableManager (page table)
    ├── PrefillManager
    └── DecodeManager
```

### Our Current Architecture
```
InferenceEngine
├── model (HuggingFace)
├── kv_pool (KVCachePool)
├── scheduler_state (SchedulerState)
└── request_caches (RequestCache per request)
```

## Key Differences

| Feature | mini-sglang | Ours |
|---------|-------------|------|
| Model | Custom impl, reads batch from global ctx | HuggingFace |
| Attention | FlashAttention with paged KV | HuggingFace past_key_values |
| KV Storage | Kernel-based store_cache | Python tensor indexing |
| Batching | Padded batches with masks | One request at a time |
| CUDA Graphs | Yes, for decode | No |
| Overlap | Dual-stream scheduling | No |
| Prefix Cache | Radix tree | No |
| Chunked Prefill | Yes | No |

## Implementation Phases

### Phase 1: Custom Model + FlashAttention
**Files to create:**
- `rollouts/rollouts/inference/models/base.py` — base model class
- `rollouts/rollouts/inference/models/llama.py` — Llama impl
- `rollouts/rollouts/inference/attention/backend.py` — attention backend protocol
- `rollouts/rollouts/inference/attention/flash.py` — FlashAttention impl

**Key insight:** mini-sglang uses global context (`get_global_ctx().batch`) to pass batch info.
We'll pass explicitly instead: `model.forward(batch)` where batch contains input_ids, positions, out_loc.

**Dependencies:**
- `flash-attn` or `sgl-kernel` for FlashAttention
- Need to handle: prefill (variable seq lens) vs decode (seq len = 1)

### Phase 2: Batched Forward
**Changes to scheduler:**
- Build padded input_ids tensor
- Build positions tensor
- Build attention metadata (cu_seqlens, etc.)

**Attention metadata (from mini-sglang):**
```python
@dataclass
class FAMetadata:
    cu_seqlens_k: Tensor  # cumulative seq lens for keys
    cu_seqlens_q: Tensor  # cumulative seq lens for queries
    cache_seqlens: Tensor  # current cached length per seq
    max_seqlen_k: int
    max_seqlen_q: int
    page_table: Tensor    # [batch, max_seq] -> slot indices
```

### Phase 3: CUDA Graphs
**Pattern from mini-sglang:**
1. Pre-allocate capture buffers for each batch size [1,2,4,8,16,...]
2. Capture graph with dummy inputs
3. On replay: copy real data to buffers, replay graph

**Requirements:**
- Static tensor shapes within each graph
- Pad batch to next power of 2 (or fixed sizes)
- Decode only (prefill has variable shapes)

### Phase 4: Overlap Scheduling
**Two CUDA streams:**
- `scheduler_stream`: CPU-side work, memory allocation
- `engine_stream`: GPU compute

**Loop:**
```python
while True:
    # On scheduler_stream: prepare batch N+1
    forward_input = schedule_next_batch()

    # On engine_stream: run batch N+1
    with engine_stream:
        engine_stream.wait_stream(scheduler_stream)
        output = forward(forward_input)

    # On scheduler_stream: process batch N results
    process_last_results(last_output)
    last_output = output
```

### Phase 5: Radix Cache
**Data structure:**
- Radix tree where each node holds K,V for a chunk of tokens
- Nodes keyed by token IDs
- Reference counting for eviction

**Operations:**
- `match_prefix(token_ids)` → find longest cached prefix
- `insert_prefix(token_ids, kv_indices)` → cache completed sequence
- `evict(size)` → free LRU leaves

### Phase 6: Chunked Prefill
**For long prompts:**
1. Split into chunks of `max_extend_tokens`
2. Process chunk by chunk
3. Only sample after final chunk

**ChunkedReq:** marker that request shouldn't be sampled yet

### Phase 7: Numerical Equivalence Tests
**Test matrix:**
- Model: SmolLM2-135M, Llama-3.1-8B
- Modes: greedy, temperature, top-k, top-p
- Cases: single request, batched, long context, prefix cache hit

**Comparisons:**
1. Our engine vs HuggingFace generate (baseline)
2. Our engine vs mini-sglang (feature parity)
3. Logits comparison (numerical precision)

## File Structure (Proposed)

```
rollouts/rollouts/inference/
├── __init__.py
├── core.py              # Req, Batch, SamplingParams (existing)
├── scheduler.py         # SchedulerState, schedule_step (existing)
├── engine.py            # InferenceEngine (refactor)
├── kv_cache.py          # KVCachePool (existing, extend)
├── models/
│   ├── __init__.py
│   ├── base.py          # BaseModel protocol
│   ├── config.py        # ModelConfig
│   ├── llama.py         # LlamaForCausalLM
│   └── weight.py        # Weight loading
├── attention/
│   ├── __init__.py
│   ├── backend.py       # AttentionBackend protocol
│   ├── flash.py         # FlashAttention impl
│   └── metadata.py      # AttentionMetadata
├── layers/
│   ├── __init__.py
│   ├── linear.py        # ColumnParallel, RowParallel
│   ├── norm.py          # RMSNorm
│   ├── rotary.py        # RoPE
│   └── activation.py    # SiLU, etc.
├── sampling.py          # Sampler
├── graph.py             # GraphRunner (CUDA graphs)
├── overlap.py           # Overlap scheduler
├── radix.py             # RadixCacheManager
└── tests/
    ├── test_engine.py   # (existing)
    ├── test_equivalence.py
    └── test_attention.py
```

## Dependencies

```toml
[project.optional-dependencies]
inference = [
    "flash-attn>=2.5.0",
    "sgl-kernel>=0.1.0",  # for optimized FA
    "triton>=2.2.0",
]
```

## Implementation Status (2026-02-15)

All phases implemented:

| Phase | Status | Files |
|-------|--------|-------|
| 1. FlashAttention | ✅ Done | `attention/flash.py`, `attention/reference.py` |
| 2. Custom Model | ✅ Done | `models/llama.py`, `layers/*` |
| 3. Batched Forward | ✅ Done | `engine_v2.py`, `attention/backend.py` |
| 4. CUDA Graphs | ✅ Done | `graph.py` |
| 5. Overlap Scheduling | ✅ Done | `overlap.py` |
| 6. Radix Cache | ✅ Done | `radix.py` |
| 7. Chunked Prefill | ✅ Done | `chunked_prefill.py` |
| 8. Equivalence Tests | ✅ Done | `tests/test_*.py` |

### Key Design Decisions

1. **No global context** - Unlike mini-sglang which uses `get_global_ctx()`,
   we pass all state explicitly through function parameters.

2. **Two engine variants**:
   - `InferenceEngine` (engine.py): HuggingFace backend, simple, compatible
   - `InferenceEngineV2` (engine_v2.py): Custom model, FlashAttention, fast

3. **Attention backend abstraction** - Protocol-based design allows swapping
   FlashAttention, reference impl, or future optimized kernels.

4. **Immutable state** - Scheduler state, request cache, etc. are frozen
   dataclasses with pure function transitions.

### Testing

Run tests:
```bash
# Equivalence tests
python -m rollouts.inference.tests.test_equivalence

# mini-sglang parity tests
python -m rollouts.inference.tests.test_mini_sglang_parity

# All tests
python -m pytest rollouts/rollouts/inference/tests/
```

### Remaining Work

1. **Tensor Parallelism** - Not implemented, single-GPU only
2. **MoE Support** - No mixture-of-experts layers
3. **Qwen Model** - Only Llama implemented
4. **Production Hardening** - Error handling, logging, metrics

## Open Questions

1. **Use sgl-kernel or flash-attn?**
   - sgl-kernel has `flash_attn_with_kvcache` optimized for serving
   - flash-attn is more widely available
   - Decision: support both, prefer sgl-kernel when available

2. **Tensor parallelism?**
   - mini-sglang has it, but adds complexity
   - Decision: defer to later phase, focus on single-GPU first

3. **Model weight loading?**
   - HuggingFace safetensors vs custom
   - Decision: use HuggingFace for loading, our model for inference
