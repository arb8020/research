# Inference Engine Implementation Brief

This is a reference document of design decisions that production inference
engines (vLLM, SGLang, etc.) converged on. Not everything here is relevant
to the early steps of the progression — treat it as a map of decisions you
will eventually have to make, so you make them consciously rather than by
accident.

Items you don't understand yet: leave them. They will make sense once you
have the context from the relevant progression step.

---

## 1. Upfront Model Analysis Phase

At startup, before serving any requests:

- Autotune GEMM kernels against actual model weight shapes. Benchmark
  cuBLAS/cublasLt with 32+ algorithm candidates, cache results to disk keyed
  by (shape, dtype, hardware). Never pay this cost at runtime.
- Generate JIT-specialized PTX for M=1 decode paths with model dimensions as
  compile-time constants. Cache to disk. This is 2-7x over generic kernels
  for single-token decode.
- Validate all required kernels are present for the selected execution path.
  Hard crash if anything is missing. No silent fallbacks.
- Detect quantization config from model directory (quantize_config.json).
  Don't require the user to specify it.
- Pre-allocate every buffer that touches the scheduler or forward pass hot
  path. No runtime tensor construction. This includes penalty tensors,
  sampling buffers, page tables, decode slots — everything.

---

## 2. Request Metadata Schema

Every request object carries these fields from entry to exit. Adding them
later requires touching the scheduler, batcher, and output pipeline
simultaneously. Design the request object with all of these fields upfront
even if most are unused in early steps.

**Identity and routing:**
- `request_id: str`
- `lora_adapter_id: Optional[str]` — also serves as cache namespace key
- `cache_namespace: Optional[str]` — for non-LoRA namespace isolation
- `priority: int`
- `routing_key: Optional[str]`

**Input:**
- `input_token_ids: List[int]`
- `input_embeds: Optional[Tensor]` — pre-computed embeddings if bypassing tokenizer
- `multimodal_inputs: Optional[MultimodalInputs]` — typed, with per-item modality, hash, kv_length

**Chunked prefill tracking** — load-bearing for activation harvesting and
prefix cache correctness:
- `chunk_index: int` — which chunk of this request's prefill we're processing
- `absolute_token_offset: int` — absolute position of this chunk's first token
- `is_chunked: bool`

**KV cache state:**
- `kv_committed_len: int` — tokens with finalized KV written
- `kv_allocated_len: int` — tokens with KV allocated (may exceed committed for speculative decode)
- `prefix_cache_node: Optional[TreeNode]` — last matched node in radix tree
- `cached_token_count: int` — tokens served from cache (for metrics)
- `req_pool_idx: int` — slot in req_to_token_pool

**Sampling:**
- `temperature, top_p, top_k, min_p, repetition_penalty, frequency_penalty, presence_penalty`
- `max_new_tokens, min_new_tokens`
- `stop_token_ids: List[int]`
- `stop_strings: List[str]`
- `sampling_seed: Optional[int]`
- `ignore_eos: bool`
- `logit_bias: Optional[Dict[int, float]]`

**Logprobs** — needed for RL training:
- `return_logprob: bool`
- `top_logprobs_k: int`
- `logprob_start_len: int` — only compute logprobs from this position
- `return_input_logprobs: bool`
- `specific_token_ids_logprob: Optional[List[int]]`

**Activation harvesting** — must be first-class, not bolted on:
- `capture_hidden_mode: Enum[NULL, LAST, FULL]`
- `capture_layer_indices: Optional[List[int]]` — which layers to capture; None means all
- `return_hidden_states_before_norm: bool`

**Speculative decode:**
- `spec_accepted_tokens: int`
- `spec_verify_count: int`
- `kv_overallocated_freed: bool`

**Output tracking:**
- `output_token_ids: List[int]` — accumulated
- `send_token_offset: int` — streaming cursor
- `finished_reason: Optional[Enum]`
- `decoded_text: str` — incremental

**Constrained decoding:**
- `json_schema: Optional[str]`
- `regex: Optional[str]`
- `ebnf: Optional[str]`

**Expert routing (MoE interpretability / RL):**
- `return_routed_experts: bool`
- `routed_experts: Optional[Tensor]`

**Disaggregation:**
- `routed_dp_rank: Optional[int]`
- `disagg_kv_sender: Optional[KVSender]`
- `start_send_idx: int`

**Timing/observability:**
- `arrival_time: float`
- `first_token_time: Optional[float]`
- `time_in_queue: float`

---

## 3. KV Cache and Page Table

- Paged KV cache, configurable page size (default 16 or 32). Do not force page_size=1.
- Radix tree prefix cache with two separate lock types: `lock_ref` (eviction)
  and `prefill_lock_ref` (mutation during active prefill). Single refcount is
  a correctness bug under concurrent prefills on shared nodes.
- O(1) evictable and reclaimable page counters maintained incrementally on
  lock/unlock/insert/evict. Never traverse the tree to answer "can I fit
  this request?"
- Typed cache tokens with `cache_key()` and `kv_length()`. One logical token
  can consume N KV positions. Handle this at the cache layer, not by
  flattening in the tokenizer.
- Cache namespace isolation per LoRA adapter and per cache-salt. Different
  namespaces never share KV even with identical token sequences.
- CPU-authoritative page table: allocation decisions happen in CPU Python
  with numpy-backed page table. Indexed H2D copy of only dirty rows before
  each forward pass. No GPU readbacks during scheduling.

---

## 4. Kernels

- Fused top-p/top-k sampling kernel. Never do this in Python at high concurrency.
- Fused RMSNorm + per-token FP8 quantize (one kernel, not two).
- Fused SiLU×mul + per-token FP8 quantize (one kernel, not two).
- Register all custom ops as `torch.ops` extensions for CUDA graph and
  `torch.compile` compatibility.
- CUDA graphs captured per discrete batch size (1 through max_decode_batch,
  at least 32 sizes). Pad to nearest captured size.
- Piecewise graph capture for MoE models — MoE routing is data-dependent and
  breaks whole-graph capture. Split at routing boundaries.
- Kernel dispatch is explicit and env-var controlled. No silent automatic
  path selection.

---

## 5. No Synchronous Work on the Forward Pass Critical Path

- Scheduler runs in a separate process. It never blocks the GPU forward pass.
- Tokenization runs in a separate process. Same.
- Output processing (detokenization, logprob assembly, streaming) runs in a
  separate process.
- IPC between processes via ZMQ or equivalent. GPU tensor sharing via CUDA IPC.
- Activation harvesting pipeline: during forward pass, one non-blocking D2D
  copy into a pre-staged GPU buffer. All subsequent work (H2D DMA, sequence
  reconstruction from flattened batch, disk write) happens in sidecar threads
  synchronized via CUDA events. The forward pass never waits on any of this.
- Async scheduling: schedule the next batch while the current forward pass is
  executing. This requires the scheduler to be one step ahead — design for it
  from day one, not as an add-on.
- All per-step data structures are preallocated tensors written into at
  runtime. No list→tensor conversions, no pin_memory() calls on the critical
  path.

---

## 6. Secondary GPU Workloads

Any co-resident GPU workload (codec decoder, vision encoder, reward model,
SAE) must go through a batching layer:

- Accumulate work across requests with a configurable timeout and max batch size.
- Single batched invocation instead of per-request calls.
- Batch along both the within-request dimension (e.g. audio chunks) and the
  across-request dimension.
- `torch.compile` the batched path.
- Single CPU→GPU copy per batch, not one per request.

---

## 7. Parallelism

- Tensor parallelism via NCCL all-reduce. Capture activations after all-reduce
  (the aggregated representation), not within shards.
- Expert parallelism for MoE: requires all-to-all for routing, not all-reduce.
  Design the communication layer to support both.
- Pipeline parallelism: optional, only needed for models that don't fit in TP=8.
- For RL training weight sync: expose an HTTP weight-update endpoint on the
  inference server. Accept a NCCL broadcast from training ranks. Hold executor
  lock only during `load_weights()`, not during inference forward. Track weight
  version per request for staleness detection.

---

## 8. Quantization

- Detect from model directory at startup.
- FP8 E4M3 weight quantization with per-tensor weight scales, per-token
  activation scales.
- FP8 KV cache, per-layer k/v scales stored alongside weights.
- Fused quantization kernels (section 4 above) — the unfused chain wastes
  memory bandwidth.
- Support Marlin for INT4 W4A16 (best available INT4 kernel).
- Hardware-gate FP8 paths on SM89+. Hard error if FP8 requested on
  incompatible hardware.
