"""Training loop for pretraining.

Simple, explicit training loop with logging. No hidden magic.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import TINY_CONFIG, ModelConfig, TrainConfig, get_git_info
from .models.llama import count_parameters, forward, init_weights

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for training."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_random_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random batch for testing (no real data yet)."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Labels are input_ids shifted by 1 (next token prediction)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids, labels


def train_step(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
) -> torch.Tensor:
    """Single training step: forward, loss, backward.

    Returns loss (still attached to graph - caller handles optimizer step).
    """
    logits = forward(input_ids, weights, config)
    loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        labels.view(-1),
    )
    loss.backward()
    return loss


def save_config(config: TrainConfig, output_dir: Path) -> None:
    """Save config + git info for reproducibility."""
    git_hash, git_dirty = get_git_info()

    meta = {
        "config": asdict(config),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "fingerprint": config.fingerprint(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=2)


def save_checkpoint(
    weights: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainConfig,
    output_dir: Path,
) -> None:
    """Save training checkpoint."""
    ckpt_path = output_dir / f"step_{step:08d}.pt"
    torch.save(
        {
            "weights": {k: v.cpu() for k, v in weights.items()},
            "optimizer": optimizer.state_dict(),
            "step": step,
            "fingerprint": config.fingerprint(),
        },
        ckpt_path,
    )
    logger.info(f"saved checkpoint: {ckpt_path}")


def train(config: TrainConfig) -> None:
    """Main training function."""
    # Setup
    setup_logging()
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    logger.info(f"device: {device}, dtype: {dtype}")

    # Output directory
    output_dir = Path(config.output_dir)
    if config.run_id:
        output_dir = output_dir / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_config(config, output_dir)

    # Seed
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)

    # Model
    logger.info("initializing model...")
    weights = init_weights(config.model, device, dtype)
    n_params = count_parameters(weights)
    logger.info(f"parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        weights.values(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Training loop
    logger.info(f"starting training for {config.steps} steps...")
    start_time = time.time()

    for step in range(config.steps):
        step_start = time.time()

        # Generate batch (random data for now)
        input_ids, labels = generate_random_batch(
            config.batch_size,
            config.max_seq_len,
            config.model.vocab_size,
            device,
        )

        # Forward, backward
        loss = train_step(input_ids, labels, weights, config.model)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(weights.values(), config.max_grad_norm)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        if step % config.log_every == 0:
            step_time = time.time() - step_start
            tokens_per_sec = (config.batch_size * config.max_seq_len) / step_time
            elapsed = time.time() - start_time

            logger.info(
                f"step={step:5d} | loss={loss.item():.4f} | "
                f"grad_norm={grad_norm:.4f} | "
                f"tok/s={tokens_per_sec:.0f} | "
                f"elapsed={elapsed:.1f}s"
            )

        # Checkpoint
        if config.checkpoint_every > 0 and step > 0 and step % config.checkpoint_every == 0:
            save_checkpoint(weights, optimizer, step, config, output_dir)

    # Final checkpoint
    save_checkpoint(weights, optimizer, config.steps, config, output_dir)

    total_time = time.time() - start_time
    logger.info(f"training complete in {total_time:.1f}s")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Pretraining")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--log-every", type=int, default=None, help="Log every N steps")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # Start with tiny config, override with CLI args
    config_dict = asdict(TINY_CONFIG)
    model_dict = config_dict.pop("model")

    # Apply CLI overrides
    for key in ["steps", "batch_size", "lr", "log_every", "output_dir", "run_id", "seed"]:
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            config_dict[key] = value

    # Reconstruct config
    model_config = ModelConfig(**model_dict)
    config = TrainConfig(model=model_config, **config_dict)

    train(config)


if __name__ == "__main__":
    main()
