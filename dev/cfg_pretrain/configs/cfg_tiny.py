"""Tiny config for CFG-based pretraining (single GPU, fast iteration).

Usage:
    python train_cfg.py configs/cfg_tiny.py
    python train_cfg.py configs/cfg_tiny.py --resume
    torchrun --standalone --nproc_per_node=8 train_cfg.py configs/cfg_tiny.py
"""

from pathlib import Path

# Import from rollouts
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "rollouts"))

from rollouts.pretrain.config import ModelConfig, TrainConfig

# Path to CFG config
cfg_path = Path(__file__).parent.parent.parent.parent / "PhysicsLM4" / "data-synthetic-pretrain" / "Lano-cfg" / "configs" / "cfg3f.json"

# Training config
config = TrainConfig(
    model=ModelConfig(
        dim=256,
        n_layers=4,
        n_heads=4,
        vocab_size=7,  # Will be overridden by CFG vocab_size + 4
    ),
    steps=100,
    batch_size=4,
    max_seq_len=128,
    warmup_steps=10,
    log_every=10,
    checkpoint_every=50,
    val_every=25,
    val_batches=5,
    output_dir="output/cfg_tiny",
)
