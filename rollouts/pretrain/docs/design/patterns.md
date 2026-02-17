# Patterns Reference

> Code patterns extracted from research, ready to copy-paste.

## 1. Functional Model (from rollouts/tools/functional_extractor/llama_functional.py)

```python
"""Pure functional transformer - no classes, just functions + weight dicts."""

import torch
import torch.nn.functional as F
from torch import Tensor


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return weight * x_normed.to(x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    """Apply RoPE to query and key."""
    cos = cos.unsqueeze(1)  # [batch, 1, seq, head_dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def compute_rope_embeddings(
    positions: Tensor,
    head_dim: int,
    theta: float = 10000.0,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor]:
    """Compute RoPE cos/sin embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    positions_expanded = positions[:, None, :].float()
    inv_freq_expanded = inv_freq[None, :, None]
    freqs = (inv_freq_expanded @ positions_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def attention(
    hidden_states: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    o_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tensor:
    """Self-attention with RoPE and GQA."""
    batch_size, seq_len, _ = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    # Project Q, K, V (no bias in Llama)
    q = F.linear(hidden_states, q_weight)
    k = F.linear(hidden_states, k_weight)
    v = F.linear(hidden_states, v_weight)

    # Reshape to [batch, num_heads, seq_len, head_dim]
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # Repeat KV for GQA
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)

    # Scaled dot-product attention
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, num_heads * head_dim)
    return F.linear(attn_output, o_weight)


def mlp(hidden_states: Tensor, gate_weight: Tensor, up_weight: Tensor, down_weight: Tensor) -> Tensor:
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""
    gate = F.linear(hidden_states, gate_weight)
    up = F.linear(hidden_states, up_weight)
    return F.linear(F.silu(gate) * up, down_weight)


def transformer_layer(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    layer_idx: int,
    cos: Tensor,
    sin: Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tensor:
    """Single transformer layer."""
    prefix = f"layers.{layer_idx}"

    # Pre-attention norm + attention + residual
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights[f"{prefix}.input_layernorm.weight"])
    hidden_states = attention(
        hidden_states,
        weights[f"{prefix}.self_attn.q_proj.weight"],
        weights[f"{prefix}.self_attn.k_proj.weight"],
        weights[f"{prefix}.self_attn.v_proj.weight"],
        weights[f"{prefix}.self_attn.o_proj.weight"],
        cos, sin, num_heads, num_kv_heads, head_dim,
    )
    hidden_states = residual + hidden_states

    # Pre-MLP norm + MLP + residual
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights[f"{prefix}.post_attention_layernorm.weight"])
    hidden_states = mlp(
        hidden_states,
        weights[f"{prefix}.mlp.gate_proj.weight"],
        weights[f"{prefix}.mlp.up_proj.weight"],
        weights[f"{prefix}.mlp.down_proj.weight"],
    )
    hidden_states = residual + hidden_states

    return hidden_states


def forward(
    input_ids: Tensor,
    weights: dict[str, Tensor],
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    vocab_size: int,
) -> Tensor:
    """Full transformer forward pass."""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Embeddings
    hidden_states = F.embedding(input_ids, weights["embed_tokens.weight"])

    # RoPE
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    cos, sin = compute_rope_embeddings(positions, head_dim, dtype=hidden_states.dtype)

    # Layers
    for i in range(num_layers):
        hidden_states = transformer_layer(
            hidden_states, weights, i, cos, sin,
            num_heads, num_kv_heads, head_dim,
        )

    # Output
    hidden_states = rms_norm(hidden_states, weights["norm.weight"])
    logits = F.linear(hidden_states, weights["lm_head.weight"])
    return logits
```

---

## 2. Generator-Based Data Loading (from modded-nanogpt)

```python
"""Generator pattern for dynamic batch size/seq len."""

import numpy as np
import torch
from pathlib import Path
from threading import Thread
from queue import Queue


def load_shard(path: Path) -> torch.Tensor:
    """Load tokenized .npy shard."""
    return torch.from_numpy(np.load(path).astype(np.int64))


def load_shard_async(path: Path, queue: Queue):
    """Load shard in background thread."""
    def _load():
        queue.put(load_shard(path))
    Thread(target=_load, daemon=True).start()


class Shard:
    """Manages a single data shard with offset tracking."""

    def __init__(self, tokens: torch.Tensor, rank: int, world_size: int):
        # Shard data across ranks
        chunk_size = len(tokens) // world_size
        start = rank * chunk_size
        end = start + chunk_size
        self.tokens = tokens[start:end]
        self.offset = 0

    def next_batch(self, num_tokens: int, max_seq_len: int) -> dict:
        """Get next batch of tokens."""
        if self.offset + num_tokens > len(self.tokens):
            return None  # Exhausted

        batch_tokens = self.tokens[self.offset : self.offset + num_tokens]
        self.offset += num_tokens

        # Reshape into sequences
        # Simple: fixed seq_len (modded-nanogpt does BOS alignment)
        num_seqs = num_tokens // max_seq_len
        batch_tokens = batch_tokens[: num_seqs * max_seq_len]
        batch_tokens = batch_tokens.view(num_seqs, max_seq_len)

        return {
            "input_ids": batch_tokens[:, :-1],
            "labels": batch_tokens[:, 1:],
        }

    def exhausted(self) -> bool:
        return self.offset >= len(self.tokens)


def distributed_data_generator(
    shard_paths: list[Path],
    batch_tokens: int,
    max_seq_len: int,
    rank: int,
    world_size: int,
):
    """Generator yielding batches, supports .send() for dynamic params."""
    shard_idx = 0

    # Load first shard
    shard = Shard(load_shard(shard_paths[shard_idx]), rank, world_size)

    # Prefetch next shard
    prefetch_queue = Queue()
    if len(shard_paths) > 1:
        load_shard_async(shard_paths[1], prefetch_queue)

    while True:
        batch = shard.next_batch(batch_tokens, max_seq_len)

        if batch is None:
            # Move to next shard
            shard_idx = (shard_idx + 1) % len(shard_paths)

            # Get prefetched shard
            if not prefetch_queue.empty():
                tokens = prefetch_queue.get()
            else:
                tokens = load_shard(shard_paths[shard_idx])

            shard = Shard(tokens, rank, world_size)

            # Prefetch next
            next_idx = (shard_idx + 1) % len(shard_paths)
            load_shard_async(shard_paths[next_idx], prefetch_queue)

            continue

        # Yield batch, receive new params
        new_params = yield batch

        if new_params is not None:
            batch_tokens, max_seq_len = new_params
```

---

## 3. Distributed Setup (from miniray + modded-nanogpt)

```python
"""NCCL distributed training setup."""

import os
import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int]:
    """Initialize NCCL process group.

    Returns:
        (rank, world_size, local_rank)

    Launch with:
        torchrun --standalone --nproc_per_node=8 train.py
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce with mean."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def all_reduce_grads(weights: dict[str, torch.Tensor]):
    """Sync gradients across ranks."""
    if not dist.is_initialized():
        return

    for param in weights.values():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


def is_main_process() -> bool:
    """Check if this is rank 0."""
    return int(os.environ.get("RANK", 0)) == 0


def print0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)
```

---

## 4. Metrics Writer (from nmoe)

```python
"""Parquet-based metrics logging."""

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class MetricRecord:
    run: str
    step: int
    ts_ms: int
    tag: str
    value: float


class MetricsWriter:
    """Write metrics to parquet, queryable with DuckDB."""

    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self._buffer: list[MetricRecord] = []

        self.run_dir.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, **metrics: float):
        """Buffer metrics for this step."""
        ts_ms = int(time.time() * 1000)
        for tag, value in metrics.items():
            self._buffer.append(MetricRecord(
                run=self.run_id,
                step=step,
                ts_ms=ts_ms,
                tag=tag,
                value=float(value),
            ))

    def flush(self):
        """Write buffer to parquet (atomic)."""
        if not self._buffer:
            return

        df = pd.DataFrame([asdict(r) for r in self._buffer])

        # Atomic write via rename
        step = self._buffer[-1].step
        tmp_path = self.run_dir / f".tmp_step_{step:08d}.parquet"
        final_path = self.run_dir / f"step_{step:08d}.parquet"

        df.to_parquet(tmp_path)
        tmp_path.rename(final_path)

        self._buffer.clear()

    def close(self):
        """Flush remaining buffer."""
        self.flush()


# Query with DuckDB:
# duckdb -c "SELECT * FROM 'metrics/*.parquet' WHERE tag = 'loss' ORDER BY step"
```

---

## 5. Config + Fingerprinting (from nmoe)

```python
"""Configuration with reproducibility."""

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int = 64
    mlp_dim: int | None = None  # Default: 4 * dim
    vocab_size: int = 50304
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.mlp_dim is None:
            object.__setattr__(self, 'mlp_dim', 4 * self.dim)


@dataclass(frozen=True)
class TrainConfig:
    # Model
    model: ModelConfig

    # Data
    data_pattern: str
    max_seq_len: int = 1024

    # Optimization
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Training
    steps: int = 10000
    log_every: int = 10
    checkpoint_every: int = 1000

    # Output
    output_dir: str = "output"
    run_id: str = ""

    def fingerprint(self) -> str:
        """Stable hash for resume checks."""
        d = asdict(self)
        # Remove runtime fields
        d.pop("output_dir", None)
        d.pop("run_id", None)
        s = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def get_git_info() -> tuple[str, bool]:
    """Get git hash and dirty status."""
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        return git_hash, len(status) > 0
    except subprocess.CalledProcessError:
        return "unknown", False


def save_config(config: TrainConfig, path: Path):
    """Save config + git info for reproducibility."""
    git_hash, git_dirty = get_git_info()

    meta = {
        "config": asdict(config),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "fingerprint": config.fingerprint(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
```

---

## 6. Simple Training Loop

```python
"""Minimal training loop tying everything together."""

import torch
import torch.nn.functional as F
from pathlib import Path

from .models.llama import forward as model_forward
from .data import distributed_data_generator
from .distributed import setup_distributed, all_reduce_grads, print0
from .metrics import MetricsWriter
from .config import TrainConfig, save_config


def init_weights(config: TrainConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Initialize model weights."""
    m = config.model
    weights = {}

    # Embedding
    weights["embed_tokens.weight"] = torch.randn(m.vocab_size, m.dim, device=device) * 0.02

    # Layers
    for i in range(m.n_layers):
        prefix = f"layers.{i}"
        # Attention
        weights[f"{prefix}.input_layernorm.weight"] = torch.ones(m.dim, device=device)
        weights[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(m.n_heads * m.head_dim, m.dim, device=device) * 0.02
        weights[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(m.n_kv_heads * m.head_dim, m.dim, device=device) * 0.02
        weights[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(m.n_kv_heads * m.head_dim, m.dim, device=device) * 0.02
        weights[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(m.dim, m.n_heads * m.head_dim, device=device) * 0.02
        # MLP
        weights[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(m.dim, device=device)
        weights[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(m.mlp_dim, m.dim, device=device) * 0.02
        weights[f"{prefix}.mlp.up_proj.weight"] = torch.randn(m.mlp_dim, m.dim, device=device) * 0.02
        weights[f"{prefix}.mlp.down_proj.weight"] = torch.randn(m.dim, m.mlp_dim, device=device) * 0.02

    # Output
    weights["norm.weight"] = torch.ones(m.dim, device=device)
    weights["lm_head.weight"] = torch.randn(m.vocab_size, m.dim, device=device) * 0.02

    # Enable gradients
    for w in weights.values():
        w.requires_grad_(True)

    return weights


def train(config: TrainConfig):
    """Main training function."""
    # Distributed setup
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Output dir
    output_dir = Path(config.output_dir) / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    if rank == 0:
        save_config(config, output_dir / "config.json")

    # Model
    weights = init_weights(config, device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        weights.values(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Data
    shard_paths = sorted(Path(".").glob(config.data_pattern))
    data_gen = distributed_data_generator(
        shard_paths, config.batch_size * config.max_seq_len,
        config.max_seq_len, rank, world_size,
    )

    # Metrics
    metrics = MetricsWriter(output_dir / "metrics", config.run_id)

    # Training loop
    m = config.model
    for step in range(config.steps):
        batch = next(data_gen)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward
        logits = model_forward(
            input_ids, weights,
            m.n_layers, m.n_heads, m.n_kv_heads, m.head_dim, m.vocab_size,
        )

        # Loss
        loss = F.cross_entropy(
            logits.view(-1, m.vocab_size),
            labels.view(-1),
        )

        # Backward
        loss.backward()
        all_reduce_grads(weights)

        # Clip + step
        torch.nn.utils.clip_grad_norm_(weights.values(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # Log
        if step % config.log_every == 0:
            loss_val = loss.item()
            metrics.log(step, loss=loss_val)
            print0(f"step={step} loss={loss_val:.4f}")

        # Checkpoint
        if step % config.checkpoint_every == 0 and step > 0:
            if rank == 0:
                ckpt_path = output_dir / f"step_{step:08d}.pt"
                torch.save({
                    "weights": {k: v.cpu() for k, v in weights.items()},
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }, ckpt_path)
                print0(f"Saved checkpoint to {ckpt_path}")

        # Flush metrics periodically
        if step % 100 == 0:
            metrics.flush()

    metrics.close()
    print0("Training complete!")
```

---

## 7. Modal Runner (from rollouts/modal_runner.py)

```python
"""Run training on Modal."""

import modal
from pathlib import Path

# Define Modal app
app = modal.App("pretrain")

# GPU image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tiktoken", "pandas")
)


@app.function(
    image=image,
    gpu="H100:8",  # 8x H100
    timeout=24 * 60 * 60,  # 24 hours
    secrets=[modal.Secret.from_name("wandb")],  # Optional
)
def train_modal(config_dict: dict):
    """Training function that runs on Modal."""
    import sys
    sys.path.insert(0, "/root/pretrain")

    from pretrain.config import TrainConfig, ModelConfig
    from pretrain.train import train

    # Reconstruct config
    model_config = ModelConfig(**config_dict["model"])
    config = TrainConfig(model=model_config, **{
        k: v for k, v in config_dict.items() if k != "model"
    })

    # Run training
    train(config)


@app.local_entrypoint()
def main(config_path: str = "configs/small.toml"):
    """Local entrypoint - provisions and runs on Modal."""
    import tomllib

    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)

    train_modal.remote(config_dict)
```

---

## Usage

```bash
# Local single GPU
python -m pretrain.train --config configs/tiny.toml

# Local 8x GPU
torchrun --standalone --nproc_per_node=8 -m pretrain.train --config configs/small.toml

# Modal 8x H100
modal run scripts/run_modal.py --config configs/small.toml
```
