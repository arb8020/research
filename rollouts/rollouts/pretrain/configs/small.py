"""Small LLaMA-style model (~30M params).

Usage:
    python rollouts/pretrain/configs/small.py --real-data
    python rollouts/pretrain/configs/small.py --real-data --resume
    torchrun --standalone --nproc_per_node=8 rollouts/pretrain/configs/small.py --real-data
"""

from rollouts.pretrain.config import ModelConfig, TrainConfig

config = TrainConfig(
    model=ModelConfig(
        dim=512,
        n_layers=8,
        n_heads=8,
    ),
    steps=1000,
    batch_size=8,
    max_seq_len=512,
    lr=3e-4,
    warmup_steps=100,
    log_every=10,
    checkpoint_every=500,
    val_every=100,
    val_batches=20,
)

if __name__ == "__main__":
    import argparse

    from rollouts.pretrain.train import train

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--real-data", action="store_true", help="Use fineweb data")
    args = parser.parse_args()

    train(config, use_real_data=args.real_data, resume=args.resume)
