"""Pretrain on clean iGSM data (Modal deployment).

This is the baseline model for RL experiments.

Paper setup (GPT2-12-12 on iGSM-med):
- 12 layers, 12 heads, 768 hidden dim
- LR: 0.002, weight decay: 0.05
- Batch size: 512, context: 768
- 100k steps, cosine decay to 0.01x, 1000 step warmup
- AdamW β=(0.9, 0.98), fp16 mixed precision

Our setup: 8x H100 DDP, ~1.5-2 hours, ~$47 on Modal.

Usage:
    python -m argus run --config examples/rl/igsm/pretrain_clean_modal.py
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
    gpu_type="H100",
    gpu_count=8,
    provider="modal",
    deps=DepsConfig(
        pip_packages=(
            "torch>=2.4",
            "transformers>=4.50",
            "numpy",
            "tiktoken",  # GPT2 tokenizer
        ),
        pip_index_url="https://download.pytorch.org/whl/cu124",
    ),
)

# =============================================================================
# Training Configuration
# =============================================================================

# Paper config: GPT2-12-12 with RoPE
# Paper uses LayerNorm + GELU; we use RMSNorm + SwiGLU (LLaMA-style)
# Both use RoPE. Paper notes LLaMA arch performs similarly.
model_config = ModelConfig(
    dim=768,
    n_layers=12,
    n_heads=12,
    vocab_size=50304,  # GPT2 vocab padded for efficiency
)

config = TrainConfig(
    model=model_config,
    steps=100_000,
    batch_size=64,  # Per-GPU batch (effective = 64 * 8 = 512)
    max_seq_len=768,
    warmup_steps=1000,
    lr_adamw=0.002,
    weight_decay=0.05,
    adam_betas=(0.9, 0.98),
    log_every=100,
    checkpoint_every=10_000,
    val_every=5_000,
    val_batches=20,
    output_dir="output/igsm_clean",
    use_muon=False,  # AdamW only, like paper
    use_compile=True,
)

igsm_config = {
    "difficulty": "med",
    "max_op": 15,
    "max_edge": 20,
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
    # (iGSM repo doesn't include these)
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
    """Entry point for Modal execution.

    Args:
        config: Training config (uses module-level default if None)
        igsm_config: iGSM config (uses module-level default if None)
        **kwargs: Additional args (resume)

    Returns:
        Dict with training results
    """
    # Use module-level defaults if not provided
    if config is None:
        from examples.rl.igsm.pretrain_clean_modal import config as default_config

        config = default_config
    if igsm_config is None:
        from examples.rl.igsm.pretrain_clean_modal import igsm_config as default_igsm

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
