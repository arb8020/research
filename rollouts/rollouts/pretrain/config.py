"""Configuration dataclasses with fingerprinting for reproducibility.

Usage:
    from rollouts.pretrain.config import ModelConfig, TrainConfig

    config = TrainConfig(
        model=ModelConfig(dim=256, n_layers=4, n_heads=4),
        steps=100,
        batch_size=4,
    )
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""

    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int | None = None  # None = same as n_heads (MHA)
    head_dim: int | None = None  # None = dim // n_heads
    mlp_dim: int | None = None  # None = 4 * dim
    vocab_size: int = 50304  # Padded for efficiency
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    # Architecture variants
    use_qk_norm: bool = False  # QK Norm (normalize Q and K after RoPE)
    use_relu2: bool = False  # ReLU² instead of SwiGLU
    # TODO: logit_softcap: float | None = None  # Cap logits (e.g., 15.0) to stabilize training
    # TODO: use_value_embeds: bool = False  # ResFormer-style value embeddings (alternating layers)
    # TODO: sliding_window_pattern: str | None = None  # e.g., "SSSL" (3 sliding + 1 global)

    def __post_init__(self) -> None:
        assert self.dim > 0, "dim must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.rope_theta > 0, "rope_theta must be positive"
        assert self.rms_norm_eps > 0, "rms_norm_eps must be positive"

        # Set defaults via object.__setattr__ since frozen
        if self.n_kv_heads is None:
            object.__setattr__(self, "n_kv_heads", self.n_heads)
        if self.head_dim is None:
            object.__setattr__(self, "head_dim", self.dim // self.n_heads)
        if self.mlp_dim is None:
            object.__setattr__(self, "mlp_dim", 4 * self.dim)

        assert self.n_kv_heads is not None
        assert self.head_dim is not None
        assert self.mlp_dim is not None
        assert self.n_kv_heads > 0, "n_kv_heads must be positive"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.head_dim > 0, "head_dim must be positive"
        assert self.head_dim * self.n_heads <= self.dim, "head_dim * n_heads must not exceed dim"
        assert self.mlp_dim > 0, "mlp_dim must be positive"


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""

    # Model
    model: ModelConfig

    # Data
    data_pattern: str = ""  # Empty = random data (for testing)
    max_seq_len: int = 512

    # Optimization
    batch_size: int = 4
    grad_accum_steps: int = 1  # Gradient accumulation steps (effective_batch = batch_size * grad_accum_steps * world_size)
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Muon optimizer (for 2D weight matrices)
    use_muon: bool = True
    lr_muon: float = 0.02  # Muon LR for 2D matrices
    muon_momentum: float = 0.95

    # AdamW optimizer (for embeddings, norms, biases)
    lr_adamw: float = 3e-4
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8

    # Training
    steps: int = 1000
    log_every: int = 10
    checkpoint_every: int = 500
    val_every: int = 50  # Validate every N steps (0 to disable)
    val_batches: int = 10  # Number of batches for validation

    # Performance
    use_compile: bool = True  # torch.compile the forward pass (CUDA only)
    use_fp8: bool = False  # FP8 training (H100+ only)

    # Output
    output_dir: str = "output"
    run_id: str = ""

    # Random seed
    seed: int = 42

    def __post_init__(self) -> None:
        assert self.data_pattern is not None, "data_pattern cannot be None"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.grad_accum_steps > 0, "grad_accum_steps must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"
        assert self.lr_muon > 0, "lr_muon must be positive"
        assert 0 <= self.muon_momentum < 1, "muon_momentum must be in [0, 1)"
        assert self.lr_adamw > 0, "lr_adamw must be positive"
        assert len(self.adam_betas) == 2, "adam_betas must contain exactly two values"
        assert 0 <= self.adam_betas[0] < 1, "adam beta1 must be in [0, 1)"
        assert 0 <= self.adam_betas[1] < 1, "adam beta2 must be in [0, 1)"
        assert self.adam_eps > 0, "adam_eps must be positive"
        assert self.steps > 0, "steps must be positive"
        assert self.log_every > 0, "log_every must be positive"
        assert self.checkpoint_every > 0, "checkpoint_every must be positive"
        assert self.val_every >= 0, "val_every must be non-negative"
        assert self.val_batches > 0, "val_batches must be positive"
        assert self.output_dir.strip(), "output_dir cannot be empty"
        assert self.seed >= 0, "seed must be non-negative"

    def fingerprint(self) -> str:
        """Stable hash for resume checks.

        Excludes runtime fields (output_dir, run_id) so fingerprint
        only changes when training-relevant config changes.
        """
        d = asdict(self)
        # Remove runtime fields
        d.pop("output_dir", None)
        d.pop("run_id", None)
        s = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def get_git_info() -> tuple[str, bool]:
    """Get git hash and dirty status for reproducibility."""
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        return git_hash, len(status) > 0
    except subprocess.CalledProcessError:
        return "unknown", False
