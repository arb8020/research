# Pretraining Playground Design

> Modular pretraining infrastructure for architecture experimentation, targeting matx prep.

## Resources Consulted

### Repositories
- `/Users/chiraagbalu/research/nmoe` - B200-targeted MoE training, metrics/experiment patterns
- `/Users/chiraagbalu/research/rollouts/rollouts/training` - Loss functions, PyTorch backend, GRPO
- `/Users/chiraagbalu/research/rollouts/rollouts/tools/functional_extractor/llama_functional.py` - Functional Llama implementation
- `/Users/chiraagbalu/research/miniray` - Multi-GPU orchestration (fork+socketpair, NCCL setup)
- `/Users/chiraagbalu/research/broker` - Compute provisioning (Modal, RunPod, LambdaLabs)
- `/tmp/modded-nanogpt` - Keller Jordan's speedrun repo (45 world records)

### Documentation
- `/Users/chiraagbalu/research/docs/code_style/` - Logging recommendations, wide events
- `/Users/chiraagbalu/research/nmoe/nmoe/metrics.py` - Parquet-per-step metrics
- `/Users/chiraagbalu/research/nmoe/nmoe/experiments.py` - SQLite experiment tracking
- `/Users/chiraagbalu/research/nmoe/nmoe/research/lab.py` - Multi-seed experiment harness

### External
- https://github.com/KellerJordan/modded-nanogpt - Speedrun patterns

---

## Goals

1. **Architecture experimentation** - Easy to swap layers, attention variants, MoE configurations
2. **Minimal dependencies** - torch, numpy, tiktoken only
3. **Functional models** - Pure functions with weight dicts, not nn.Module inheritance
4. **Reproducible** - Config fingerprinting, git hash capture, deterministic sharding
5. **8xGPU ready** - NCCL distributed training on Modal (expandable to 2x8x later)
6. **QAT support** - Quantization-aware training for chip company prep

---

## Directory Structure

```
pretrain/
├── pyproject.toml              # Minimal deps: torch, numpy, tiktoken
├── configs/
│   ├── tiny.toml               # Single-GPU testing (256 dim, 6 layers)
│   ├── small.toml              # 8xGPU training
│   └── tasks.toml              # Data sources, tokenizer contract
│
├── pretrain/
│   ├── __init__.py
│   │
│   ├── models/                 # Functional model implementations
│   │   ├── llama.py            # Dense Llama-style (from functional_extractor)
│   │   ├── moe.py              # Mixture of Experts
│   │   ├── attention/
│   │   │   ├── mha.py          # Multi-head attention
│   │   │   ├── gqa.py          # Grouped query attention
│   │   │   ├── mla.py          # Multi-head latent attention (DeepSeek)
│   │   │   └── rope.py         # Rotary embeddings
│   │   └── mlp.py              # SwiGLU, ReLU², etc.
│   │
│   ├── train.py                # Main training loop
│   ├── data.py                 # Shard loading, distributed generator
│   ├── optim.py                # AdamW (Muon later)
│   ├── distributed.py          # NCCL setup, reduce-scatter/all-gather
│   ├── metrics.py              # Parquet logging
│   ├── config.py               # Dataclass configs + fingerprinting
│   ├── experiments.py          # SQLite experiment tracking
│   │
│   └── kernels/                # Optional Triton kernels
│       ├── fused_attention.py
│       └── fp8_matmul.py
│
├── scripts/
│   ├── run_local.py            # torchrun --nproc_per_node=8
│   ├── run_modal.py            # Modal sandbox provisioning
│   └── prep_data.py            # Tokenize datasets to shards
│
└── research/
    └── lab.py                  # Experiment harness (multi-seed, comparison)
```

---

## Dependencies

```toml
[project]
name = "pretrain"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4.0",
    "numpy",
    "tiktoken",
]

[project.optional-dependencies]
research = [
    "matplotlib",
    "pandas",
    "duckdb",      # Query parquet metrics
]
dev = [
    "pytest",
    "ruff",
]
```

**Explicitly NOT using:**
- transformers (use tiktoken for tokenizer, functional models for architecture)
- peft (no LoRA abstraction - implement directly if needed)
- accelerate (manual FSDP/DDP)
- trio (sync training loop)
- datasets (direct shard loading)

---

## Key Patterns

### 1. Functional Models (from llama_functional.py)

```python
def transformer_forward(
    input_ids: Tensor,           # [batch, seq_len]
    weights: dict[str, Tensor],  # All model weights
    config: ModelConfig,         # Frozen dataclass
) -> Tensor:                     # [batch, seq_len, vocab_size]
    """Pure function - no hidden state, no nn.Module."""

    hidden = F.embedding(input_ids, weights["embed.weight"])
    cos, sin = compute_rope(seq_len, config.head_dim, config.rope_theta)

    for i in range(config.n_layers):
        hidden = transformer_layer(hidden, weights, i, cos, sin, config)

    hidden = rms_norm(hidden, weights["norm.weight"])
    logits = F.linear(hidden, weights["lm_head.weight"])
    return logits
```

**Why functional:**
- Easy to swap components (different attention, different MLP)
- Explicit weight management (good for sharding, checkpointing)
- torch.compile friendly (no object overhead)
- Matches how you'd think about it on a chip

### 2. Generator-Based Data Loading (from modded-nanogpt)

```python
def distributed_data_generator(
    shard_pattern: str,
    batch_tokens: int,
    max_seq_len: int,
    rank: int,
    world_size: int,
):
    """Generator that yields batches, receives new params via .send()"""
    shard = load_next_shard(shard_pattern, rank, world_size)

    while True:
        batch = shard.next_batch(batch_tokens, max_seq_len)

        # Yield batch, receive new hyperparams (or None)
        new_params = yield batch

        if new_params is not None:
            batch_tokens, max_seq_len = new_params

        if shard.exhausted():
            shard = load_next_shard(...)
```

**Why generator:**
- Supports dynamic batch size/seq len changes mid-training
- No DataLoader overhead
- Explicit control over prefetching (async thread for next shard)

### 3. Explicit Distributed (from modded-nanogpt + miniray)

```python
def setup_distributed():
    """Initialize NCCL process group."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

def all_reduce_grads(model_weights: dict[str, Tensor]):
    """Manual gradient sync - no FSDP magic."""
    for name, param in model_weights.items():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
```

**For later (100B+ scale):**
- reduce-scatter for sharded optimizer state
- all-gather for forward pass
- Expert parallelism for MoE

### 4. Parquet Metrics (from nmoe)

```python
class MetricsWriter:
    """Write metrics to parquet files, queryable with DuckDB."""

    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir = run_dir
        self.run_id = run_id
        self._buffer = []  # In-memory buffer

    def log(self, step: int, **metrics):
        """Buffer metrics for this step."""
        self._buffer.append({
            "run": self.run_id,
            "step": step,
            "ts_ms": time.time_ns() // 1_000_000,
            **metrics,
        })

    def flush(self, step: int):
        """Atomic write to parquet."""
        df = pd.DataFrame(self._buffer)
        path = self.run_dir / f"step_{step:08d}.parquet"
        df.to_parquet(path)
        self._buffer.clear()
```

**Why parquet:**
- No W&B/tensorboard dependency
- Query with DuckDB: `SELECT * FROM 'metrics/*.parquet' WHERE loss < 2.0`
- Atomic writes (no corruption on crash)

### 5. Config Fingerprinting (from nmoe)

```python
@dataclass(frozen=True)
class TrainConfig:
    # Model
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int = 50304

    # Training
    batch_size: int = 8
    lr: float = 3e-4
    steps: int = 10000

    def fingerprint(self) -> str:
        """Stable hash for resume checks."""
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        s = json.dumps(d, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:16]
```

---

## Training Loop Sketch

```python
def train(config: TrainConfig, resume_from: Path | None = None):
    # Setup
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Model (functional)
    weights = init_weights(config, device)
    if resume_from:
        weights = load_checkpoint(resume_from)

    # Optimizer
    optimizer = torch.optim.AdamW(weights.values(), lr=config.lr)

    # Data
    data_gen = distributed_data_generator(
        config.data_pattern, config.batch_size, config.max_seq_len,
        rank, world_size,
    )

    # Metrics
    metrics = MetricsWriter(config.output_dir, config.run_id)

    # Training
    for step in range(config.steps):
        batch = next(data_gen)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward
        logits = transformer_forward(batch["input_ids"], weights, config)
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            batch["labels"].view(-1),
        )

        # Backward
        loss.backward()
        all_reduce_grads(weights)

        # Optimize
        torch.nn.utils.clip_grad_norm_(weights.values(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Log
        if rank == 0 and step % 10 == 0:
            metrics.log(step, loss=loss.item())
            print(f"step={step} loss={loss.item():.4f}")

        # Checkpoint
        if step % config.checkpoint_every == 0:
            save_checkpoint(weights, optimizer, step, config.output_dir)

    metrics.flush(step)
```

---

## Compute Strategy

### Phase 1: Local Development
- Single GPU testing with tiny config
- `python -m pretrain.train --config configs/tiny.toml`

### Phase 2: 8xGPU on Modal
- H100 x8 @ $31.60/hr
- `python scripts/run_modal.py --config configs/small.toml --gpu H100 --gpu-count 8`

### Phase 3: 2x8x (if needed for 100B)
- miniray TCP workers for multi-node
- NCCL across nodes via broker SSH provisioning

---

## Implementation Order

1. **Core training loop** (train.py, config.py)
   - Minimal: forward, backward, optimize, checkpoint
   - Test with random data first

2. **Functional Llama** (models/llama.py)
   - Copy from rollouts/tools/functional_extractor
   - Verify against HuggingFace

3. **Data pipeline** (data.py)
   - Shard loading from .npy files
   - Distributed generator

4. **Metrics** (metrics.py)
   - Parquet writer
   - Basic logging

5. **Distributed** (distributed.py)
   - NCCL setup
   - Gradient sync

6. **Modal runner** (scripts/run_modal.py)
   - Provision 8xH100
   - Run training

7. **Attention variants** (models/attention/)
   - GQA, MLA for experimentation

8. **MoE** (models/moe.py)
   - Router, expert parallelism

9. **QAT** (quantization-aware training)
   - Fake quantization during training

---

## Open Questions

1. **Tokenizer**: tiktoken (o200k) or train custom?
2. **Data**: FineWeb-Edu, or custom mix?
3. **First architecture**: Start with dense Llama, or jump to MoE?
4. **Parallelism**: Data parallel only to start, or tensor parallel?
5. **Checkpoint format**: PyTorch native, or safetensors?

---

## Related Files

After implementation, key files will be:
- `pretrain/pretrain/train.py` - Main training loop
- `pretrain/pretrain/models/llama.py` - Functional Llama
- `pretrain/pretrain/config.py` - Configuration
- `pretrain/scripts/run_modal.py` - Cloud runner
