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

    def __post_init__(self) -> None:
        # Set defaults via object.__setattr__ since frozen
        if self.n_kv_heads is None:
            object.__setattr__(self, "n_kv_heads", self.n_heads)
        if self.head_dim is None:
            object.__setattr__(self, "head_dim", self.dim // self.n_heads)
        if self.mlp_dim is None:
            object.__setattr__(self, "mlp_dim", 4 * self.dim)


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
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Training
    steps: int = 1000
    log_every: int = 10
    checkpoint_every: int = 500
    val_every: int = 50  # Validate every N steps (0 to disable)
    val_batches: int = 10  # Number of batches for validation

    # Output
    output_dir: str = "output"
    run_id: str = ""

    # Random seed
    seed: int = 42

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
