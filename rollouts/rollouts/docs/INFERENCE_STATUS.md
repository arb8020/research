# Inference Engine Status

## Completed

### Sliding Window Attention (SWA) - VERIFIED
- FlexAttention implementation with BlockMask
- Tested against reference explicit-mask implementation
- Max diff: 8.34e-07 (within fp tolerance)
- Full causal mode also verified (diff: 1.19e-06)

Key files:
- `rollouts/inference/attention/mask.py` - `create_sliding_window_causal_mask()`, `create_attention_mask()`
- `rollouts/inference/attention/flex_backend.py` - FlexAttentionBackend
- `rollouts/tools/functional_extractor/debug_swa.py` - Test script

### Package Structure
- Moved `tools/` into `rollouts/tools/` for clean imports
- Remote execution uses `uv pip install -e .` (no PYTHONPATH hacks)

## Architecture

```
rollouts/inference/
├── attention/
│   ├── config.py      # CacheConfig
│   ├── protocol.py    # AttentionBackend protocol
│   ├── flex_backend.py # FlexAttention implementation
│   ├── mask.py        # BlockMask creation (causal, SWA)
│   └── layer.py       # nn.Module wrapper
├── cache/
│   └── paged.py       # PagedKVCache
├── context.py         # InferenceContext
├── engine.py          # InferenceEngine
├── scheduler.py       # Request scheduling
└── types.py           # Request/Response types
```

## Next Steps

1. **Paged KV Cache** - Implement block allocation, copy-on-write
2. **Continuous Batching** - Scheduler for prefill/decode mixing
3. **Model Integration** - Wire up with Qwen3 functional model
4. **Benchmarks** - Compare against vLLM/SGLang

## Running Tests

```bash
# SWA test (provisions GPU automatically)
uv run python -m rollouts.tools.functional_extractor.debug_swa

# Reuse existing GPU
uv run python -m rollouts.tools.functional_extractor.debug_swa --gpu-id <id>
```
