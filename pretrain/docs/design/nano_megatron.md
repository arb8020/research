# nano-megatron: Minimal Pretraining for DeepSeek-V3

> **Design Principles Applied:**
> - Write usage code first (Casey)
> - Frozen dataclasses for config/data (CLASSES_VS_FUNCTIONAL)
> - Classes only for legitimate state (resources, lifecycle)
> - Pure functions for computation
> - Functions orchestrate stateful objects
> - Minimize stateful components (Sean)
> - Boring, well-tested components (Sean)
> - Rely on transformer-engine like nano-inference relies on sgl-kernel

---

## Motivation

Megatron-LM is ~100k LOC. Most of it is flexibility we don't need:
- Multiple model architectures (GPT, BERT, T5, ...)
- Multiple parallelism strategies with fallbacks
- Legacy code paths for older hardware

**nano-megatron**: ~2000 LOC that trains DeepSeek-V3 at 200+ TFLOP/s/GPU by:
1. Using **transformer-engine** for fused FP8 kernels (like nano-inference uses sgl-kernel)
2. Implementing **only DSv3 code paths** (MLA attention, MoE with aux-loss-free routing)
3. Targeting **Hopper+** (no fallbacks)

---

## Usage Code First

What we *want* the API to look like:

```python
# ═══════════════════════════════════════════════════
# USAGE: Pretraining
# ═══════════════════════════════════════════════════

from pretrain import Trainer, TrainConfig, ModelConfig, DataConfig

config = TrainConfig(
    model=ModelConfig(
        hidden_size=7168,
        num_layers=61,
        num_attention_heads=128,
        num_experts=256,
        num_active_experts=8,
        # DSv3 specifics
        mla_q_lora_rank=1536,
        mla_kv_lora_rank=512,
        routed_scaling_factor=2.5,
    ),
    data=DataConfig(
        path="/data/fineweb_edu",
        seq_len=4096,
    ),
    batch_size=1024,
    lr=1e-4,
    warmup_steps=2000,
    max_steps=100_000,
)

# Trainer owns distributed state, model, optimizer
trainer = Trainer(config)

# Training loop
for step, metrics in trainer.train():
    if step % 100 == 0:
        print(f"step={step} loss={metrics.loss:.3f} tflops={metrics.tflops:.1f}")

    if step % 1000 == 0:
        trainer.save_checkpoint(f"/checkpoints/step_{step}")

trainer.shutdown()
```

```python
# ═══════════════════════════════════════════════════
# USAGE: Low-Level Control
# ═══════════════════════════════════════════════════

from pretrain import (
    init_distributed,
    create_model,
    create_optimizer,
    forward_backward,
    optimizer_step,
)

# Initialize distributed (NCCL, process groups)
dist_ctx = init_distributed()

# Create model with TP/EP sharding
model = create_model(model_config, dist_ctx)

# Optimizer with ZeRO-1
optimizer = create_optimizer(model, lr=1e-4)

# Training step (pure function orchestrating stateful objects)
for batch in dataloader:
    loss, grads = forward_backward(model, batch)
    optimizer_step(optimizer, grads)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Trainer (class)                             │
│   Legitimate state: model, optimizer, distributed context           │
│                                                                      │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│   │ DSv3Model        │  │ FusedOptimizer   │  │ DistContext      │  │
│   │ (nn.Module)      │  │ (class)          │  │ (class)          │  │
│   │ - MLA attention  │  │ - ZeRO-1 sharded │  │ - TP/EP groups   │  │
│   │ - MoE layers     │  │ - FP8 master wts │  │ - NCCL handles   │  │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                      │
│   Built on transformer-engine:                                       │
│   - te.TransformerLayer for fused FP8 ops                           │
│   - te.LayerNormMLP for expert FFNs                                 │
│   - te.fp8_autocast for automatic scaling                           │
└─────────────────────────────────────────────────────────────────────┘
```

**Why classes?**
- `Trainer`: Owns model, optimizer, checkpointing lifecycle
- `DSv3Model`: Owns parameters (nn.Module)
- `DistContext`: Owns NCCL process groups

**Why transformer-engine?**
- Fused FP8 GEMMs: 2x memory bandwidth vs BF16
- Grouped GEMMs for MoE: all experts in one kernel
- Per-tensor scaling with amax history
- We skip writing ~10k LOC of CUDA kernels

---

## Data Types (Frozen Dataclasses)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    """DSv3 model configuration. Immutable."""
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_experts: int
    num_active_experts: int

    # MLA (Multi-head Latent Attention)
    mla_q_lora_rank: int = 1536
    mla_kv_lora_rank: int = 512
    mla_qk_nope_dim: int = 128
    mla_qk_rope_dim: int = 64
    mla_v_head_dim: int = 128

    # MoE routing
    routed_scaling_factor: float = 2.5
    n_shared_experts: int = 1


@dataclass(frozen=True)
class DataConfig:
    """Data configuration. Immutable."""
    path: str
    seq_len: int = 4096


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration. Immutable."""
    model: ModelConfig
    data: DataConfig
    batch_size: int
    lr: float
    warmup_steps: int = 2000
    max_steps: int = 100_000

    # Parallelism
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    data_parallel_size: int = 1  # computed from world_size


@dataclass(frozen=True)
class StepMetrics:
    """Metrics from one training step. Immutable."""
    step: int
    loss: float
    grad_norm: float
    tflops: float
    tokens_per_sec: float
```

---

## Key Components

### 1. DSv3 Model (~500 LOC)

```python
class DSv3Model(nn.Module):
    """DeepSeek-V3 architecture.

    Key differences from vanilla transformer:
    - MLA (Multi-head Latent Attention) instead of MHA
    - MoE with aux-loss-free load balancing
    - Shared + routed experts
    """

    def __init__(self, config: ModelConfig, dist_ctx: DistContext):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DSv3Layer(config, dist_ctx, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        self.norm = te.RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)


class DSv3Layer(nn.Module):
    """Single DSv3 layer: MLA + MoE."""

    def __init__(self, config: ModelConfig, dist_ctx: DistContext, layer_idx: int):
        super().__init__()
        self.attn = MLAAttention(config, dist_ctx)
        self.moe = MoELayer(config, dist_ctx)
        self.attn_norm = te.RMSNorm(config.hidden_size)
        self.ffn_norm = te.RMSNorm(config.hidden_size)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.moe(self.ffn_norm(x))
        return x
```

### 2. MLA Attention (~200 LOC)

```python
class MLAAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2/V3).

    Compresses KV cache by projecting K,V through low-rank bottleneck.
    Q uses LoRA-style decomposition.
    """

    def __init__(self, config: ModelConfig, dist_ctx: DistContext):
        super().__init__()
        # Q projection: down -> up with RoPE split
        self.q_down = nn.Linear(config.hidden_size, config.mla_q_lora_rank)
        self.q_up = nn.Linear(config.mla_q_lora_rank, config.num_attention_heads * config.mla_qk_nope_dim)
        self.q_rope = nn.Linear(config.mla_q_lora_rank, config.num_attention_heads * config.mla_qk_rope_dim)

        # KV projection: shared low-rank
        self.kv_down = nn.Linear(config.hidden_size, config.mla_kv_lora_rank)
        self.k_up = nn.Linear(config.mla_kv_lora_rank, config.num_attention_heads * config.mla_qk_nope_dim)
        self.k_rope = nn.Linear(config.mla_kv_lora_rank, config.mla_qk_rope_dim)  # shared across heads
        self.v_up = nn.Linear(config.mla_kv_lora_rank, config.num_attention_heads * config.mla_v_head_dim)

        self.o_proj = nn.Linear(config.num_attention_heads * config.mla_v_head_dim, config.hidden_size)
```

### 3. MoE Layer (~300 LOC)

```python
class MoELayer(nn.Module):
    """Mixture of Experts with aux-loss-free routing.

    Uses transformer-engine grouped GEMMs for efficiency.
    """

    def __init__(self, config: ModelConfig, dist_ctx: DistContext):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_experts)

        # Shared expert (always active)
        self.shared_expert = te.LayerNormMLP(
            config.hidden_size,
            config.hidden_size * 4,
        )

        # Routed experts (top-k selected)
        # Using TE grouped GEMM under the hood
        self.experts = GroupedExperts(
            config.num_experts,
            config.hidden_size,
            config.hidden_size * 4,
            dist_ctx.ep_group,
        )

        self.num_active = config.num_active_experts
        self.scaling = config.routed_scaling_factor

    def forward(self, x):
        # Router logits
        logits = self.router(x)

        # Top-k routing (aux-loss-free: bias correction instead of aux loss)
        scores, indices = topk_routing(logits, self.num_active)

        # Shared expert
        shared_out = self.shared_expert(x)

        # Routed experts (grouped GEMM)
        routed_out = self.experts(x, scores, indices)

        return shared_out + self.scaling * routed_out
```

### 4. Distributed Context (~200 LOC)

```python
class DistContext:
    """Distributed training context.

    Owns NCCL process groups for:
    - TP (tensor parallel)
    - EP (expert parallel)
    - DP (data parallel)
    """

    def __init__(self, config: TrainConfig):
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])

        # Initialize process groups
        dist.init_process_group("nccl")

        self.tp_size = config.tensor_parallel_size
        self.ep_size = config.expert_parallel_size
        self.dp_size = self.world_size // (self.tp_size * self.ep_size)

        self.tp_group = self._create_tp_group()
        self.ep_group = self._create_ep_group()
        self.dp_group = self._create_dp_group()

    def shutdown(self):
        dist.destroy_process_group()
```

---

## What We Skip (vs Full Megatron)

| Feature | Megatron | nano-megatron |
|---------|----------|---------------|
| Model architectures | GPT, BERT, T5, LLaMA, ... | DSv3 only |
| Parallelism | TP, PP, EP, DP, CP, SP, ... | TP, EP, DP only |
| Hardware | A100, H100, B100, ... | Hopper+ only |
| Precision | FP32, FP16, BF16, FP8 | FP8 only (via TE) |
| Fallbacks | Extensive | None |
| LOC | ~100k | ~2k |

---

## Implementation Plan

### Phase 1: Single-GPU DSv3 (~800 LOC)
- [ ] Frozen config dataclasses
- [ ] DSv3Model with MLA + MoE
- [ ] Basic training loop
- [ ] transformer-engine FP8 integration
- [ ] Validate loss curve on small model

### Phase 2: Multi-GPU DP (~300 LOC)
- [ ] DistContext with process groups
- [ ] ZeRO-1 optimizer sharding
- [ ] Gradient all-reduce

### Phase 3: Expert Parallel (~400 LOC)
- [ ] Expert sharding across EP group
- [ ] All-to-all for token dispatch
- [ ] Or: RDEP-style direct dispatch (from nmoe)

### Phase 4: Tensor Parallel (~300 LOC)
- [ ] Column/row parallel linear
- [ ] Attention head sharding
- [ ] Vocab parallel embedding

### Phase 5: Data Pipeline (~200 LOC)
- [ ] Tokenized shard loading
- [ ] Streaming without full materialization
- [ ] Deterministic shuffle with resume

---

## Non-Goals

- Pipeline parallelism (adds complexity, DSv3 fits in memory with EP+TP)
- Context parallelism (sequence parallel)
- Speculative training
- Inference (use nano-inference for that)
- H100/A100 support

---

## Open Questions

1. **RDEP vs all-to-all for MoE?**
   - nmoe uses RDEP (NVSHMEM direct puts)
   - Megatron uses NCCL all-to-all
   - Decision: Start with all-to-all (simpler), benchmark RDEP later

2. **Pipeline parallel?**
   - DSv3-671B needs PP for memory
   - Decision: Skip for now, target smaller configs first

3. **Checkpoint format?**
   - Compatible with HuggingFace?
   - Decision: Simple sharded state_dict, conversion scripts later
