# Inference Engine Design

## Goal

Build a full inference engine to replace SGLang/vLLM for RL training and serving. Must be easy to reason about and extend, avoiding the accumulated complexity of existing implementations.

## Core Insight

An inference engine is fundamentally a scheduler + executor loop:

```python
while has_work():
    batch = schedule()       # decide what to run
    outputs = forward(batch) # run the model
    process(outputs)         # update state, emit tokens
```

Everything else (CUDA graphs, overlap, radix cache, TP, MoE, spec decode) plugs into this loop.

## Architecture Layers

### Distributed Layer (foundation)
- Tensor parallelism: split weight matrices across GPUs, all-reduce to combine
- Expert parallelism: route MoE experts across GPUs
- Primitives: all-reduce, all-gather (NCCL)
- Higher layers call these without knowing the details

### Model Layer
- Architecture-specific forward passes (Llama, Qwen, Mistral, etc.)
- Fused operations: QKV projection, gate+up FFN, RMSNorm+residual
- Weight loading: safetensors, TP sharding on load, HF name remapping

### Cache Layer
- **KV Cache Pool**: owns GPU memory, tracks allocated slots
- **Page Table**: maps (request, position) → cache slot (enables paged allocation)
- **Radix Tree**: tracks completed sequences for prefix matching/sharing

### Scheduler + Executor
- **Scheduler**: decides what to run (prefill queue, decode set, memory budget)
- **Executor**: runs model forward, attention backend, CUDA graphs, sampling
- Separation enables overlap scheduling (CPU schedules N+1 while GPU runs N)

### Engine Loop
- Orchestrates the while loop
- Overlap scheduling lives here (two CUDA streams)
- Simple—all complexity in the layers it calls

### API Layer
- HTTP server (OpenAI-compatible) for serving
- Python API for batch processing
- RL integration: tokens-in/tokens-out with logprobs

## Key Concepts

### Continuous Batching
Requests join/leave dynamically instead of fixed batches:
```
[req1, req2, req3] → forward → req2 finishes → [req1, req3, req4] → forward
```

Requires: variable-length attention, per-request memory management, dynamic scheduling.

### Prefill vs Decode
- **Prefill**: process N prompt tokens → generate 1 token, cache N K,V pairs (compute-bound)
- **Decode**: process 1 token → generate 1 token, cache 1 K,V pair (memory-bound)

### Paged KV Cache
Like virtual memory—allocate in pages on demand:
- Pool of pages, each holds K,V for N tokens
- Page table maps (request, position) → page
- Free pages when request completes

### Radix Cache (prefix sharing)
Tree structure for cached prefixes:
```
"You are a helpful" → [cached K,V slots 0-10]
├── " assistant" → [slots 11-13]
└── " pirate" → [slots 14-16]
```
New requests match against tree, skip redundant K,V computation.

### Attention Metadata
Variable-length batching via flattening:
```
Sequences: [5 tokens], [3 tokens], [7 tokens]
Flattened: [15 tokens total]
cu_seqlens: [0, 5, 8, 15]  # cumulative lengths
```

### CUDA Graphs
Capture decode operations, replay with minimal CPU overhead:
- Capture once per batch size (1, 2, 4, 8, ...)
- Pad actual batch to nearest captured size
- Constraint: fixed shapes, decode only

### Overlap Scheduling
Hide CPU latency with two streams:
```
CPU: [schedule N] [process N-1 + schedule N+1] [process N + schedule N+2]
GPU:              [forward N]                   [forward N+1]
```

## Design Principles

1. **Requests are frozen dataclasses** - state transitions are pure functions
2. **Engine is the loop** - simple, obvious, all control flow here
3. **Optimizations are strategies** - plug in without changing the loop
4. **Resources are classes** - KV cache pool, model own GPU memory
5. **Scheduler is pure** - takes state in, returns (batch, new_state)

## Current State (engine_v2.py)

### Working
- Basic generation loop
- Token input (RL use case)
- Batched generation
- Chunked prefill
- Logprobs (just added)
- Scaffolding for CUDA graphs, overlap, radix cache

### Needs GPU Testing
- CUDA graphs
- Overlap scheduling
- Radix cache
- FlashAttention backend (sgl-kernel)

### Missing for RL
- ~~Logprobs~~ ✓ Done
- Logprobs for given tokens (compute without sampling, for importance ratios)
- ~~Weight hot-reload~~ ✓ Done (reload_weights, reload_weights_from_path)
- ~~NCCL weight sync~~ ✓ Done (init_weight_sync, receive_weights_from_nccl)

### Missing for Scale
- Tensor parallelism (next priority)
- MoE support
- Multi-architecture (currently Llama-focused)

### Missing for Production
- HTTP server
- Structured output (XGrammar)
- Quantization (FP8)

## Feature Priority

### Tier 1: Core Correctness (RL integration)
- [x] Continuous batching
- [x] Paged KV cache
- [x] Variable-length attention
- [x] Token-level I/O
- [x] Chunked prefill
- [x] Logprobs

### Tier 2: Performance
- [ ] CUDA graphs (scaffolded, needs GPU test)
- [ ] Overlap scheduling (scaffolded, needs GPU test)
- [ ] Radix/prefix caching (scaffolded, needs GPU test)
- [ ] FlashAttention backend (scaffolded, needs sgl-kernel)
- [ ] Fused kernels

### Tier 3: Scale
- [ ] Tensor parallelism
- [ ] MoE support
- [ ] Multi-architecture (Qwen, Mistral, etc.)
- [ ] Vocab parallel embedding

### Tier 4: Advanced
- [ ] Speculative decoding
- [ ] Disaggregated prefill/decode (not needed for RL v1)
- [ ] Structured output
- [ ] Quantization
- [ ] Multi-modal

### Tier 5: Production
- [ ] HTTP server (OpenAI-compatible)
- [ ] Streaming responses
- [ ] Load balancing

## RL-Specific Requirements

Standard inference: `prompt → tokens`
RL inference: `prompt_tokens → (output_tokens, logprobs, maybe values)`

Key differences:
1. **Input is tokens, not strings** - skip tokenizer round-trips
2. **Need logprobs** - for policy gradient
3. **May need old_logprobs** - for importance sampling (PPO/GRPO)
4. **Weight updates during serving** - policy improves, reload weights

## Speculative Decoding Placement (TBD)

Two options:
1. Inside engine: draft model managed internally
2. Outside engine: RL loop provides draft tokens

For RL, outside might make sense—the policy *is* a draft model candidate. Could verify policy samples with reference model in one forward pass.

## Not Needed for v1

- **Disaggregated prefill/decode**: latency optimization for serving, RL doesn't have latency sensitivity
- **HTTP server**: RL uses Python API directly
- **Multi-modal**: focus on text first
