"""Tiny config for testing (single GPU, fast iteration).

Usage:
    python rollouts/pretrain/configs/tiny.py
    python rollouts/pretrain/configs/tiny.py --resume
    torchrun --standalone --nproc_per_node=8 rollouts/pretrain/configs/tiny.py
"""

from rollouts.pretrain.config import ModelConfig, TrainConfig

config = TrainConfig(
    model=ModelConfig(
        dim=256,
        n_layers=4,
        n_heads=4,
    ),
    steps=100,
    batch_size=4,
    max_seq_len=128,
    warmup_steps=10,
    log_every=10,
    # Muon + AdamW optimizer settings (defaults are fine for tiny)
)

if __name__ == "__main__":
    import argparse

    from rollouts.pretrain.train import train

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--real-data", action="store_true", help="Use fineweb data")
    args = parser.parse_args()

    train(config, use_real_data=args.real_data, resume=args.resume)
