"""Smoke config for nano-pretrain (few steps, fast iteration).

Run:
  python rollouts/pretrain/configs/smoke.py
  python rollouts/pretrain/configs/smoke.py --resume
"""

from rollouts.pretrain.config import ModelConfig, TrainConfig

config = TrainConfig(
    model=ModelConfig(
        dim=256,
        n_layers=4,
        n_heads=4,
    ),
    steps=20,
    batch_size=2,
    max_seq_len=64,
    warmup_steps=5,
    log_every=1,
    checkpoint_every=10,
    val_every=0,  # keep smoke runs minimal
    output_dir="output_smoke",
    run_id="nano_pretrain_smoke",
)


if __name__ == "__main__":
    import argparse

    from rollouts.pretrain.train import train

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--real-data", action="store_true", help="Use fineweb data")
    args = parser.parse_args()

    train(config, use_real_data=args.real_data, resume=args.resume)
