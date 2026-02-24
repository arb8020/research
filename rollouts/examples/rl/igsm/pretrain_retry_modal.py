"""Pretrain on iGSM data with retry tokens (Modal deployment).

This trains the "retry-capable" model that the paper shows works well.
The model learns to recover from mistakes via [BACK] tokens.

Paper setup (GPT2-12-12 on iGSM-med with 20% retry rate):
- Same architecture as clean pretrain
- retry_rate=0.2 (20% of steps have errors + corrections)
- retry_type="strong" (wrong param must not have appeared yet)

Usage:
    # Run on Modal
    python -m rollouts.modal_runner --config examples/rl/igsm/pretrain_retry_modal.py
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

model_config = ModelConfig(
    dim=768,
    n_layers=12,
    n_heads=12,
    vocab_size=50304,  # GPT2 vocab padded for efficiency
)

config = TrainConfig(
    model=model_config,
    steps=100_000,
    batch_size=512,
    max_seq_len=768,
    warmup_steps=1000,
    lr_adamw=0.002,
    weight_decay=0.05,
    adam_betas=(0.9, 0.98),
    log_every=100,
    checkpoint_every=10_000,
    val_every=5_000,
    val_batches=20,
    output_dir="output/igsm_retry",
    use_muon=False,
    use_compile=True,
)

# iGSM config with retry tokens
igsm_config = {
    "difficulty": "med",
    "max_op": 15,
    "max_edge": 20,
    # Retry-specific settings (paper used 20-50% error rates)
    "retry_rate": 0.2,  # 20% of steps have errors
    "retry_type": "strong",  # Wrong param must not have appeared yet
}

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

    # Create __init__.py files needed for Python imports
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
    **kwargs: Any,
) -> dict[str, Any]:
    """Entry point for Modal execution."""
    if config is None:
        from examples.rl.igsm.pretrain_retry_modal import config as default_config

        config = default_config
    if igsm_config is None:
        from examples.rl.igsm.pretrain_retry_modal import igsm_config as default_igsm

        igsm_config = default_igsm

    ensure_igsm_cloned()

    resume = kwargs.get("resume", False)

    # Use retry training script
    sys.path.insert(0, str(Path(__file__).parent))
    from train_igsm_retry import train as _train

    _train(config, igsm_config, resume=resume)
    return {"status": "completed", "output_dir": config.output_dir}


if __name__ == "__main__":
    train()
