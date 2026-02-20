"""Small config for CFG-based pretraining (~30M params).

Usage:
    python train_cfg.py configs/cfg_small.py
    python train_cfg.py configs/cfg_small.py --resume
    torchrun --standalone --nproc_per_node=8 train_cfg.py configs/cfg_small.py
"""

from pathlib import Path

# Import from rollouts
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "rollouts"))

from rollouts.pretrain.config import ModelConfig, TrainConfig

# Path to CFG config - using cfg3k which is more complex
cfg_path = Path(__file__).parent.parent.parent.parent / "PhysicsLM4" / "data-synthetic-pretrain" / "Lano-cfg" / "configs" / "cfg3k.json"

# Training config
config = TrainConfig(
    model=ModelConfig(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=7,  # Will be overridden by CFG vocab_size + 4
    ),
    steps=1000,
    batch_size=8,
    max_seq_len=512,
    warmup_steps=100,
    log_every=10,
    checkpoint_every=500,
    val_every=100,
    val_batches=20,
    output_dir="output/cfg_small",
)
