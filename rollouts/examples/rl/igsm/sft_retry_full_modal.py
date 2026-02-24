"""Continued pretrain on retry data (full parameters) - Modal deployment.

This is the paper's "success" condition for learning error correction.
The model learns to recover from mistakes via [BACK] tokens.

From: pretrained clean checkpoint
To: model that can use retry tokens

Paper result: 78% → 94% accuracy on hard problems

Usage:
    python -m rollouts.modal_runner --config examples/rl/igsm/sft_retry_full_modal.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from rollouts.pretrain.config import ModelConfig, TrainConfig
from rollouts.training.configs import DepsConfig, HardwareConfig

# =============================================================================
# Hardware Configuration (Modal)
# =============================================================================

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=1,
    provider="modal",
    deps=DepsConfig(
        pip_packages=(
            "torch>=2.4",
            "transformers>=4.50",
            "numpy",
            "tiktoken",
        ),
        pip_index_url="https://download.pytorch.org/whl/cu124",
    ),
)

# =============================================================================
# Training Configuration
# =============================================================================

# Same architecture as pretrain - we're continuing from checkpoint
model_config = ModelConfig(
    dim=768,
    n_layers=12,
    n_heads=12,
    vocab_size=50257,
)

# Shorter training since we're finetuning, not pretraining from scratch
# Paper uses ~20k steps for continued training
config = TrainConfig(
    model=model_config,
    steps=20_000,
    batch_size=512,
    max_seq_len=768,
    warmup_steps=500,
    lr_adamw=0.0002,  # 10x lower than pretrain
    weight_decay=0.05,
    adam_betas=(0.9, 0.98),
    log_every=100,
    checkpoint_every=5_000,
    val_every=2_500,
    val_batches=20,
    output_dir="output/igsm_sft_retry_full",
    use_muon=False,
    use_compile=True,
)

# iGSM config with retry tokens
igsm_config = {
    "difficulty": "med",
    "max_op": 15,
    "max_edge": 20,
    "retry_rate": 0.2,  # 20% error rate
    "retry_type": "strong",
}

# Path to pretrained checkpoint (must be set before running)
PRETRAIN_CHECKPOINT = "output/igsm_clean/checkpoint_100000"

# =============================================================================
# iGSM Setup
# =============================================================================

IGSM_PATH = Path("/tmp/iGSM")


def ensure_igsm_cloned() -> None:
    """Clone iGSM repo if not present and create missing __init__.py files."""
    if not IGSM_PATH.exists():
        print(f"Cloning iGSM to {IGSM_PATH}...")
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/iGSM.git", str(IGSM_PATH)],
            check=True,
        )
        print("iGSM cloned successfully")

    init_dirs = ["tools", "data_gen", "data_gen/pretrain", "const"]
    for subdir in init_dirs:
        init_file = IGSM_PATH / subdir / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"Created {init_file}")


# =============================================================================
# Entry Point
# =============================================================================


def train(
    config: TrainConfig | None = None,
    igsm_config: dict[str, Any] | None = None,
    checkpoint: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Entry point for Modal execution."""
    if config is None:
        from examples.rl.igsm.sft_retry_full_modal import config as default_config

        config = default_config
    if igsm_config is None:
        from examples.rl.igsm.sft_retry_full_modal import igsm_config as default_igsm

        igsm_config = default_igsm
    if checkpoint is None:
        from examples.rl.igsm.sft_retry_full_modal import PRETRAIN_CHECKPOINT

        checkpoint = PRETRAIN_CHECKPOINT

    ensure_igsm_cloned()

    # Use retry training script with checkpoint loading
    sys.path.insert(0, str(Path(__file__).parent))
    from train_igsm_retry import train as _train

    _train(config, igsm_config, resume=False, load_checkpoint=checkpoint)
    return {"status": "completed", "output_dir": config.output_dir}


if __name__ == "__main__":
    train()
