# Inference Benchmark Design

Compare inference backends (SGLang, vLLM, engine_v2) with fast iteration via Modal GPU snapshots.

## Goals

1. **Fast iteration** - Snapshot after model load, restore in ~2s instead of 20s+ cold boot
2. **Apples-to-apples comparison** - Same workload, same hardware, same metrics
3. **Wide event logging** - Rich per-step telemetry for identifying bottlenecks
4. **Standard workloads** - Match SGLang/vLLM benchmark formats for external comparison

## Architecture

```
rollouts/inference/benchmark/
├── __init__.py
├── runner.py          # Modal sandbox orchestration with snapshots
├── workloads.py       # Standard workloads (random, sharegpt, burst)
├── metrics.py         # Wide event collector
├── backends/
│   ├── __init__.py
│   ├── sglang.py      # SGLang server wrapper
│   ├── vllm.py        # vLLM server wrapper
│   └── engine_v2.py   # Our engine wrapper
└── compare.py         # Multi-backend comparison CLI
```

## Modal GPU Snapshots

Modal's GPU memory snapshots checkpoint the full GPU state after model loading. This transforms 20s+ cold starts into ~2s restores.

```python
import modal

app = modal.App("inference-benchmark")

# Image with deps
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "transformers", "sglang[all]", "flashinfer"
)

@app.cls(gpu="A100", image=image)
class InferenceBenchmark:
    @modal.enter(snap=True)  # Snapshot after this method
    def load_model(self):
        # This runs once, then gets snapshotted
        from rollouts.inference.engine_v2 import InferenceEngineV2, EngineConfig

        self.engine = InferenceEngineV2(EngineConfig(
            model_path="Qwen/Qwen3-0.6B",
            max_batch_size=256,
        ))
        # GPU memory state is now snapshotted

    @modal.method()
    def run_workload(self, workload: dict) -> dict:
        # Restores from snapshot in ~2s, runs workload
        return self._benchmark(workload)
```

## Wide Event Logging

Instead of scattered log lines, emit one rich event per inference step:

```python
@dataclass
class InferenceStepEvent:
    # Timing
    timestamp: float
    step_id: int
    step_duration_ms: float
    prefill_duration_ms: float | None
    decode_duration_ms: float | None

    # Batch info
    batch_size: int
    prefill_tokens: int
    decode_tokens: int
    total_tokens: int

    # Engine state
    kv_cache_pages_used: int
    kv_cache_pages_total: int
    kv_cache_hit_rate: float
    cuda_graph_hit: bool
    attention_backend: str

    # Request breakdown
    requests_prefilling: int
    requests_decoding: int
    requests_finished: int

    # Memory
    gpu_memory_used_mb: float
    gpu_memory_reserved_mb: float
```

Emit as JSONL for post-hoc analysis:
```bash
# Find slow steps
cat events.jsonl | jq 'select(.step_duration_ms > 100)'

# Analyze by batch size
cat events.jsonl | jq -s 'group_by(.batch_size) | map({batch: .[0].batch_size, avg_ms: (map(.step_duration_ms) | add / length)})'
```

## Standard Workloads

Match SGLang/vLLM benchmark formats:

### Random
Fixed input/output lengths, uniform request rate:
```python
workload = {
    "type": "random",
    "num_prompts": 1000,
    "input_len": 512,
    "output_len": 128,
    "request_rate": "inf",  # Burst mode
}
```

### ShareGPT
Real conversation distribution from ShareGPT dataset:
```python
workload = {
    "type": "sharegpt",
    "num_prompts": 1000,
    "request_rate": 10.0,  # 10 req/s
}
```

### Burst
High concurrency stress test:
```python
workload = {
    "type": "burst",
    "num_prompts": 500,
    "input_len": 256,
    "output_len": 64,
    "concurrency": 128,
}
```

## Metrics Output

Standard JSON output matching vLLM/SGLang format:

```json
{
    "backend": "engine_v2",
    "model": "Qwen/Qwen3-0.6B",
    "gpu": "A100",
    "workload": {"type": "random", "num_prompts": 1000, "input_len": 512, "output_len": 128},

    "throughput": {
        "requests_per_second": 142.5,
        "tokens_per_second": 18240,
        "output_tokens_per_second": 15360
    },

    "latency": {
        "ttft_ms": {"mean": 12.3, "p50": 11.0, "p95": 18.5, "p99": 25.2},
        "tpot_ms": {"mean": 8.1, "p50": 7.8, "p95": 12.0, "p99": 15.5},
        "e2e_ms": {"mean": 145.2, "p50": 138.0, "p95": 195.0, "p99": 245.0}
    },

    "resource": {
        "gpu_memory_peak_mb": 18432,
        "gpu_utilization_mean": 0.92
    },

    "events_file": "events.jsonl"
}
```

## CLI Usage

### Single backend benchmark
```bash
# First run creates snapshot (slow)
python -m rollouts.inference.benchmark \
    --model Qwen/Qwen3-0.6B \
    --backend engine_v2 \
    --gpu A100 \
    --workload random \
    --num-prompts 1000 \
    --output results/engine_v2.json

# Subsequent runs restore from snapshot (fast)
python -m rollouts.inference.benchmark \
    --model Qwen/Qwen3-0.6B \
    --backend engine_v2 \
    --workload sharegpt \
    --num-prompts 1000
```

### Compare backends
```bash
python -m rollouts.inference.benchmark.compare \
    --model Qwen/Qwen3-0.6B \
    --backends sglang,engine_v2 \
    --gpu A100 \
    --workload random \
    --num-prompts 1000 \
    --output comparison.json
```

### Quick iteration on engine changes
```bash
# Invalidate snapshot after code change
python -m rollouts.inference.benchmark \
    --model Qwen/Qwen3-0.6B \
    --backend engine_v2 \
    --rebuild-snapshot \
    --workload random \
    --num-prompts 100
```

## Implementation Plan

### Phase 1: Core infrastructure
- [ ] `metrics.py` - Wide event dataclass + JSONL emitter
- [ ] `workloads.py` - Random, ShareGPT, burst workload generators
- [ ] `backends/engine_v2.py` - Wrap our engine with event emission

### Phase 2: Modal integration
- [ ] `runner.py` - Modal sandbox with GPU snapshot support
- [ ] Snapshot creation/restore logic
- [ ] Code sync for iterative development

### Phase 3: Multi-backend comparison
- [ ] `backends/sglang.py` - SGLang server wrapper
- [ ] `backends/vllm.py` - vLLM server wrapper
- [ ] `compare.py` - Run same workload across backends

### Phase 4: Analysis tooling
- [ ] Summary statistics computation
- [ ] Comparison report generation
- [ ] Flamegraph-style visualization of step timing

## Instrumenting engine_v2

Add wide event emission to the engine step loop:

```python
# In engine_v2.py step() method
def step(self) -> list[RequestOutput]:
    step_start = time.perf_counter()

    # ... existing step logic ...

    # Emit wide event
    if self._event_collector:
        self._event_collector.emit(InferenceStepEvent(
            timestamp=time.time(),
            step_id=self._step_count,
            step_duration_ms=(time.perf_counter() - step_start) * 1000,
            prefill_duration_ms=prefill_time * 1000 if prefill_time else None,
            decode_duration_ms=decode_time * 1000 if decode_time else None,
            batch_size=len(self._active_requests),
            # ... all other fields ...
        ))

    return finished
```

## Why Wide Events vs Traces

Traces (OpenTelemetry spans) show request flow. Wide events show what happened at each step.

For inference optimization, we need to answer:
- "Why was this step slow?" → Need all context in one place
- "What's the batch size distribution?" → Need to aggregate across steps
- "Does CUDA graph caching help?" → Need to correlate cache hits with latency

Wide events make these queries trivial SQL/jq operations.

## References

- [Modal GPU Memory Snapshots](https://modal.com/blog/gpu-mem-snapshots)
- [Modal Memory Snapshot Docs](https://modal.com/docs/guide/memory-snapshot)
- [SGLang bench_serving](https://docs.sglang.io/developer_guide/bench_serving.html)
- [vLLM benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)
- [Wide Event Logging Philosophy](https://loggingsucks.com/)
