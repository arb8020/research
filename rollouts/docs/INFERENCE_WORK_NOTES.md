# Inference Engine Work Notes

## What We Did This Session

### 1. Integrated engine_v2 as RL Training Backend

Made our native inference engine (`engine_v2`) work as a backend for GRPO training, replacing SGLang.

**Files created/modified:**
- `rollouts/inference/server.py` - Added weight sync endpoints:
  - `/update_weights_from_disk` - disk-based weight reload
  - `/init_weights_update_group` - NCCL group setup
  - `/update_weights_from_distributed` - NCCL broadcast
  - `/destroy_weights_update_group` - cleanup

- `rollouts/training/weight_sync.py` - Added `EngineV2Engine` class:
  - Same interface as `SGLangEngine`/`VLLMEngine`
  - `launch()`, `wait_until_ready()`, `update_weights_from_checkpoint()`, `shutdown()`
  - Runs `python -m rollouts.inference.server` in tmux

- `rollouts/training/grpo.py` - Added `backend="engine_v2"` option in `_create_inference_engines()`

- `rollouts/training/configs.py` - Updated `InferenceConfig.backend` comment

- `examples/rl/reverse_text/grpo_engine_v2_01.py` - Test config using engine_v2

**Commits:**
- `a6b0c6cb` - Add engine_v2 as inference backend for RL training
- `7769cded` - Fix max_batch_size to handle n_samples_per_prompt
- `71c3fa5a` - Fix tensor truthiness check
- `6bd583eb` - Fix async concurrency (semaphore + run_in_executor)
- `a8235ac2` - Fix tensor serialization for JSON

**Verified working:** Ran RL training on RunPod with engine_v2, confirmed via logs showing:
- `"Backend: engine_v2"`
- `"Launching engine_v2 on GPU 0..."`
- `"Using FlashInfer attention backend"` (our engine's log)
- All `/v1/chat/completions` requests returning 200 OK

### 2. Benchmarking Design

Created design doc: `docs/inference_benchmark_design.md`

**Key ideas:**
- Modal GPU snapshots for fast iteration (~2s restore vs 20s cold boot)
- Wide event logging (JSONL) for aggregate analysis
- Standard workloads (random, sharegpt, burst) matching SGLang/vLLM
- Multi-backend comparison (sglang, vllm, engine_v2)

**Profiling approach (from research):**
- PyTorch Profiler → Perfetto UI (for GPU kernel analysis)
- Nsight Systems (nsys) → deep hardware profiling
- Wide events (our addition) → jq/SQL queries for aggregate metrics

**Reference:** Silares/SAIL Baseten optimization article shows the methodology:
1. Profile at max sustainable load
2. Identify bottleneck via torch profiler / nsys
3. Fix, re-profile, repeat
4. Key bottlenecks found: pin_memory, batching, async scheduling, penalties

### 3. Resources to Study

- `/tmp/kestrel/` - m87-labs/kestrel inference engine (cloned)
- Silares article: https://www.silares.com/targets/target-1-baseten
  - Shows iterative profiling methodology
  - 9x throughput improvement through system-level optimizations
  - Key insight: "the hardest part is rarely the fix itself, but recognizing where to look"

## Recent Progress (Feb 2026)

### Benchmark Runner Working

Built Modal-based benchmark runner with proper observability:

**Changes:**
- `rollouts/inference/benchmark/runner.py` - JSONL wide event logging
  - Events: `benchmark_init`, `server_start`, `server_log`, `server_ready`, `progress`, `benchmark_done`
  - Streams server logs in real-time
  - Progress updates every 10%

- Switched from `pip_install()` to `uv_pip_install()` for image builds
  - 58s build time vs 5+ minutes with pip
  - Image caching works (second run ~8s)

**Benchmark results (engine_v2, A100, Qwen3-0.6B, 20 prompts):**
- Throughput: 0.7 req/s, 23 tok/s
- TTFT p50: 102.6ms
- E2E p50: 1156.1ms
- High p95/p99 due to CUDA graph capture during benchmark

**Commands:**
```bash
# Run benchmark
python examples/benchmark/run_comparison.py --backend engine_v2
python examples/benchmark/run_comparison.py --backend sglang

# Test Modal image caching
modal run examples/benchmark/test_modal_simple.py
```

### TODO

1. **Run sglang benchmark** - get comparison numbers
2. **Profile engine_v2** - find why it's slower than sglang
   - Add torch profiler integration for Perfetto traces
   - Look at CUDA graph capture overhead
   - Check batch scheduling efficiency
3. **Study Kestrel** - `/tmp/kestrel/` - look at their scheduling approach
4. **Modal GPU snapshots** - implement for faster iteration (design in `docs/inference_benchmark_design.md`)

## Key Files

```
rollouts/inference/
├── engine_v2.py          # Main inference engine
├── server.py             # HTTP server with weight sync
├── scheduler.py          # Request scheduling
├── attention/
│   └── flashinfer.py     # FlashInfer backend
├── models/
│   └── qwen.py           # Qwen2/3/MoE support
└── layers/
    └── moe.py            # MoE implementation

docs/
├── inference_benchmark_design.md  # Benchmark design doc
└── INFERENCE_WORK_NOTES.md        # This file
```

## Commands

```bash
# Run RL training with engine_v2
python -m argus run --config examples/rl/reverse_text/grpo_engine_v2_01.py \
  --provision --provider runpod

# Run engine_v2 server standalone
python -m rollouts.inference.server --model Qwen/Qwen3-0.6B --port 30000

# Check running pods
broker list

# Terminate pod
broker terminate <instance_id> runpod
```
