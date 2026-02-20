"""Pretrain on iGSM data WITH retry tokens.

This is the paper's winning condition - models pretrained with retry data
learn self-correction without increasing error rates.

Paper findings:
- retry_rate=0.1 works well
- Loss masking on error tokens is unnecessary for reasonable retry_rate
- Models rarely use retry at inference despite training data containing it
- "strong" retry type: wrong param must not have appeared yet
- "weak" retry type: any future param works (simpler, nearly as good)

Usage:
    # Full training
    python pretrain_retry.py

    # Different retry rates
    python pretrain_retry.py --retry-rate 0.1
    python pretrain_retry.py --retry-rate 0.3

    # Weak retry (simpler, paper showed nearly as good)
    python pretrain_retry.py --retry-type weak

    # Quick test
    python pretrain_retry.py --tiny
"""

from __future__ import annotations

import argparse
from typing import Literal

from rollouts.pretrain.config import ModelConfig, TrainConfig


def get_config(
    tiny: bool = False,
    retry_rate: float = 0.1,
    retry_type: Literal["strong", "weak"] = "strong",
) -> tuple[TrainConfig, dict]:
    """Get training config with retry data.

    Args:
        tiny: If True, use tiny model for testing
        retry_rate: Probability of inserting retry at each step (0.0 to 1.0)
        retry_type: "strong" (stricter) or "weak" (simpler, nearly as good)

    Returns:
        (TrainConfig, igsm_config dict)
    """
    if tiny:
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
            output_dir=f"output/igsm_retry_{retry_type}_{retry_rate}_tiny",
            use_muon=False,
        )
        igsm = {
            "difficulty": "easy",
            "max_op": 10,
            "max_edge": 15,
            "retry_rate": retry_rate,
            "retry_type": retry_type,
        }
    else:
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
            max_seq_len=768,  # May need longer for retry sequences
            warmup_steps=1000,
            lr_adamw=0.002,
            weight_decay=0.05,
            adam_betas=(0.9, 0.98),
            log_every=100,
            checkpoint_every=10_000,
            val_every=5_000,
            val_batches=20,
            output_dir=f"output/igsm_retry_{retry_type}_{retry_rate}",
            use_muon=False,
            use_compile=True,
        )
        igsm = {
            "difficulty": "med",
            "max_op": 15,
            "max_edge": 20,
            "retry_rate": retry_rate,
            "retry_type": retry_type,
        }

    return train, igsm


def main():
    parser = argparse.ArgumentParser(description="Pretrain on iGSM with retry data")
    parser.add_argument("--tiny", action="store_true", help="Use tiny model for testing")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--retry-rate", type=float, default=0.1, help="Retry probability (default: 0.1)"
    )
    parser.add_argument(
        "--retry-type",
        choices=["strong", "weak"],
        default="strong",
        help="Retry type (default: strong)",
    )
    args = parser.parse_args()

    config, igsm_config = get_config(
        tiny=args.tiny,
        retry_rate=args.retry_rate,
        retry_type=args.retry_type,
    )

    # Import training function - use direct import to support running as script
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from train_igsm_retry import train

    train(config, igsm_config, resume=args.resume)


if __name__ == "__main__":
    main()
