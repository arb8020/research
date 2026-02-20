"""Pretrain on clean iGSM data (no retry tokens).

This is the baseline model for RL experiments.

Paper setup (GPT2-12-12 on iGSM-med):
- 12 layers, 12 heads, 768 hidden dim
- LR: 0.002, weight decay: 0.05
- Batch size: 512, context: 768
- 100k steps, cosine decay to 0.01x, 1000 step warmup
- AdamW β=(0.9, 0.98), fp16 mixed precision

Usage:
    # Quick test (CPU, tiny model)
    python examples/rl/igsm/pretrain_clean.py --tiny

    # Full training (requires GPU)
    python examples/rl/igsm/pretrain_clean.py

    # Remote execution
    python examples/rl/igsm/pretrain_clean.py --provision --provider runpod

    # Resume from checkpoint
    python examples/rl/igsm/pretrain_clean.py --resume
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rollouts.pretrain.config import ModelConfig, TrainConfig

# iGSM setup
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


def get_config(tiny: bool = False) -> tuple[TrainConfig, dict]:
    """Get training config.

    Args:
        tiny: If True, use tiny model for testing

    Returns:
        (TrainConfig, igsm_config dict)
    """
    if tiny:
        # Tiny config for quick iteration / CPU testing
        model = ModelConfig(
            dim=256,
            n_layers=4,
            n_heads=4,
            vocab_size=50257,
        )
        train = TrainConfig(
            model=model,
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
            use_muon=False,  # AdamW only, like paper
        )
        igsm = {"difficulty": "easy", "max_op": 10, "max_edge": 15}
    else:
        # Paper config: GPT2-12-12
        model = ModelConfig(
            dim=768,
            n_layers=12,
            n_heads=12,
            vocab_size=50257,
        )
        train = TrainConfig(
            model=model,
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
            output_dir="output/igsm_clean",
            use_muon=False,  # AdamW only, like paper
            use_compile=True,
        )
        igsm = {"difficulty": "med", "max_op": 15, "max_edge": 20}

    return train, igsm


def train(config: TrainConfig | None = None, igsm_config: dict | None = None, **kwargs) -> dict:
    """Entry point for remote execution.

    Args:
        config: Training config (uses default if None)
        igsm_config: iGSM config (uses default if None)
        **kwargs: Additional args (tiny, resume)

    Returns:
        Dict with training results
    """
    # Ensure iGSM is available
    ensure_igsm_cloned()

    tiny = kwargs.get("tiny", False)
    resume = kwargs.get("resume", False)

    if config is None:
        config, igsm_config = get_config(tiny=tiny)

    # Import training function
    sys.path.insert(0, str(Path(__file__).parent))
    from train_igsm import train as _train

    _train(config, igsm_config, resume=resume)
    return {"status": "completed", "output_dir": config.output_dir}


# Export config for rollouts.run compatibility
config, igsm_config = get_config(tiny=False)


def main():
    parser = argparse.ArgumentParser(description="Pretrain on clean iGSM data")
    parser.add_argument("--tiny", action="store_true", help="Use tiny model for testing")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--provision", action="store_true", help="Run on remote GPU")
    parser.add_argument("--provider", type=str, default="runpod", help="GPU provider")
    parser.add_argument("--keep-alive", action="store_true", help="Keep instance after completion")
    parser.add_argument("--node-id", type=str, default=None, help="Reuse existing instance")
    args = parser.parse_args()

    if args.provision or args.node_id:
        # Remote execution
        import functools

        import trio

        from examples.rl.base_config import run_remote

        trio.run(
            functools.partial(
                run_remote,
                script_path=__file__,
                keep_alive=args.keep_alive,
                node_id=args.node_id,
            )
        )
    else:
        # Local execution
        train(tiny=args.tiny, resume=args.resume)


if __name__ == "__main__":
    main()
