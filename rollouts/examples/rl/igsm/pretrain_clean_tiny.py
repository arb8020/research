"""Pretrain on clean iGSM data (tiny config for local testing).

Quick iteration config - runs on CPU or single GPU.

Usage:
    # Local CPU test
    python examples/rl/igsm/pretrain_clean_tiny.py

    # With GPU
    CUDA_VISIBLE_DEVICES=0 python examples/rl/igsm/pretrain_clean_tiny.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from rollouts.pretrain.config import ModelConfig, TrainConfig

# =============================================================================
# Training Configuration (Tiny)
# =============================================================================

model_config = ModelConfig(
    dim=256,
    n_layers=4,
    n_heads=4,
    vocab_size=50257,  # GPT2 vocab
)

config = TrainConfig(
    model=model_config,
    steps=100,
    batch_size=4,
    max_seq_len=256,
    warmup_steps=10,
    lr_adamw=0.002,
    weight_decay=0.05,
    adam_betas=(0.9, 0.98),
    log_every=10,
    checkpoint_every=50,
    val_every=25,
    val_batches=5,
    output_dir="output/igsm_clean_tiny",
    use_muon=False,
    use_compile=False,  # Faster startup for testing
)

igsm_config = {
    "difficulty": "easy",
    "max_op": 10,
    "max_edge": 15,
}

# =============================================================================
# iGSM Setup
# =============================================================================

IGSM_PATH = Path("/tmp/iGSM")


def ensure_igsm_cloned() -> None:
    """Clone iGSM repo if not present."""
    if not IGSM_PATH.exists():
        print(f"Cloning iGSM to {IGSM_PATH}...")
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/iGSM.git", str(IGSM_PATH)],
            check=True,
        )
        print("iGSM cloned successfully")


# =============================================================================
# Entry Point
# =============================================================================


def train(
    config: TrainConfig | None = None,
    igsm_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run training locally."""
    # Use module-level defaults if not provided
    if config is None:
        from examples.rl.igsm.pretrain_clean_tiny import config as default_config

        config = default_config
    if igsm_config is None:
        from examples.rl.igsm.pretrain_clean_tiny import igsm_config as default_igsm

        igsm_config = default_igsm

    # Ensure iGSM is available
    ensure_igsm_cloned()

    resume = kwargs.get("resume", False)

    # Import and run training
    sys.path.insert(0, str(Path(__file__).parent))
    from train_igsm import train as _train

    _train(config, igsm_config, resume=resume)
    return {"status": "completed", "output_dir": config.output_dir}


if __name__ == "__main__":
    train()
