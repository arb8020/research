"""Tiny config for iGSM-based pretraining (single GPU, fast iteration).

Usage:
    python train_igsm.py configs/igsm_tiny.py
    python train_igsm.py configs/igsm_tiny.py --resume
    torchrun --standalone --nproc_per_node=8 train_igsm.py configs/igsm_tiny.py
"""

from pathlib import Path
import sys

# Import from rollouts (need to add path carefully to avoid conflicts with iGSM)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "rollouts"))

from rollouts.pretrain.config import ModelConfig, TrainConfig

# Training config
config = TrainConfig(
    model=ModelConfig(
        dim=256,
        n_layers=4,
        n_heads=4,
        vocab_size=50257,  # GPT2 vocab size for iGSM
    ),
    steps=100,
    batch_size=4,
    max_seq_len=512,  # iGSM problems can be long
    warmup_steps=10,
    log_every=10,
    checkpoint_every=50,
    val_every=25,
    val_batches=5,
    output_dir="output/igsm_tiny",
)

# iGSM-specific settings
igsm_config = {
    "difficulty": "med",  # "easy", "med", or "hard"
    "max_op": 15,
    "max_edge": 20,
}
