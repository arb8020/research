"""Training loop for pretraining.

Simple, explicit training loop with logging. No hidden magic.

Usage:
    python rollouts/pretrain/configs/tiny.py
    python rollouts/pretrain/configs/small.py --real-data --resume
    torchrun --standalone --nproc_per_node=8 rollouts/pretrain/configs/small.py
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from ..training.datasets import BufferState, get_token_batch, load_fineweb_tokens
from . import runtime
from .config import ModelConfig, TrainConfig, get_git_info
from .models.llama import count_parameters, forward, init_weights
from .schedule import get_lr

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for training."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
    autocast_ctx: torch.amp.autocast | None = None,
) -> torch.Tensor:
    """Single training step: forward, loss, backward.

    Returns loss (still attached to graph - caller handles optimizer step).
    """
    if autocast_ctx is not None:
        with autocast_ctx:
            logits = forward(input_ids, weights, config)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                labels.view(-1),
            )
    else:
        logits = forward(input_ids, weights, config)
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
        )
    loss.backward()
    return loss


@torch.no_grad()
def eval_loss(
    tokens: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    num_batches: int,
    device: torch.device,
) -> float:
    """Compute average loss over validation data.

    Args:
        tokens: Validation token tensor
        weights: Model weights
        config: Model config
        batch_size: Batch size
        seq_len: Sequence length
        num_batches: Number of batches to evaluate
        device: Device to run on

    Returns:
        Average cross-entropy loss
    """
    state = BufferState(seed=0)
    total_loss = 0.0

    for _ in range(num_batches):
        (input_ids, labels), state = get_token_batch(tokens, state, batch_size, seq_len)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = forward(input_ids, weights, config)
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
        )
        total_loss += loss.item()

    return total_loss / num_batches


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


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the most recent checkpoint in output_dir."""
    checkpoints = sorted(output_dir.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(
    ckpt_path: Path,
    weights: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
) -> int:
    """Load checkpoint into weights and optimizer.

    Args:
        ckpt_path: Path to checkpoint file
        weights: Model weights dict (modified in place)
        optimizer: Optimizer (state loaded in place)
        config: Config for fingerprint verification
        device: Device to load weights to

    Returns:
        Step number to resume from

    Raises:
        ValueError: If checkpoint fingerprint doesn't match config
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Verify fingerprint matches
    if ckpt["fingerprint"] != config.fingerprint():
        raise ValueError(
            f"Checkpoint fingerprint mismatch: {ckpt['fingerprint']} != {config.fingerprint()}. "
            "Config has changed since checkpoint was saved."
        )

    # Load weights
    for key, value in ckpt["weights"].items():
        weights[key].copy_(value.to(device))

    # Load optimizer state
    optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt["step"]


def train(config: TrainConfig, use_real_data: bool = False, resume: bool = False) -> None:
    """Main training function.

    Supports distributed training via torchrun:
        torchrun --standalone --nproc_per_node=8 -m rollouts.pretrain.train

    Args:
        config: Training configuration
        use_real_data: If True, use fineweb data. If False, use random data.
        resume: If True, resume from latest checkpoint in output_dir.
    """
    # Setup distributed runtime (handles device, seeds, NCCL init)
    rank, world, device = runtime.init(seed=config.seed)
    setup_logging()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Mixed precision autocast (bf16 on CUDA, disabled on CPU)
    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = None

    if runtime.is_main():
        logger.info(f"device: {device}, dtype: {dtype}, world: {world}")
        logger.info(f"use_real_data: {use_real_data}")

    # Output directory (only rank 0 creates)
    output_dir = Path(config.output_dir)
    if config.run_id:
        output_dir = output_dir / config.run_id
    if runtime.is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir)

    # Load data (each rank uses different seed for data parallelism)
    train_tokens = None
    val_tokens = None
    train_state = BufferState(seed=config.seed + rank)

    if use_real_data:
        if runtime.is_main():
            logger.info("loading fineweb tokens...")
        # Each rank loads different shards for distributed training
        train_tokens = load_fineweb_tokens(
            split="train", num_chunks=world, rank=rank, world_size=world
        )
        val_tokens = load_fineweb_tokens(split="val")
        if runtime.is_main():
            logger.info(f"train tokens per rank: {len(train_tokens):,}")
            logger.info(f"val tokens: {len(val_tokens):,}")

    # Model
    if runtime.is_main():
        logger.info("initializing model...")
    weights = init_weights(config.model, device, dtype)
    n_params = count_parameters(weights)
    if runtime.is_main():
        logger.info(f"parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        weights.values(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Resume from checkpoint
    start_step = 0
    if resume:
        ckpt_path = find_latest_checkpoint(output_dir)
        if ckpt_path:
            start_step = load_checkpoint(ckpt_path, weights, optimizer, config, device)
            if runtime.is_main():
                logger.info(f"resumed from {ckpt_path} at step {start_step}")
        elif runtime.is_main():
            logger.info("no checkpoint found, starting from scratch")

    # Schedule config (dict for get_lr)
    schedule_config = {
        "steps": config.steps,
        "lr": config.lr,
        "warmup_steps": config.warmup_steps,
    }

    # Training loop
    remaining_steps = config.steps - start_step
    grad_accum_steps = config.grad_accum_steps
    effective_batch = config.batch_size * grad_accum_steps * world

    if runtime.is_main():
        logger.info(
            f"starting training for {remaining_steps} steps (step {start_step} to {config.steps})..."
        )
        logger.info(
            f"effective batch size: {effective_batch} (batch={config.batch_size} x accum={grad_accum_steps} x world={world})"
        )
    start_time = time.time()

    val_every = config.val_every
    val_batches = config.val_batches

    for step in range(start_step, config.steps):
        step_start = time.time()

        # Update learning rate
        lr = get_lr(step, schedule_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation loop
        accum_loss = 0.0
        for accum_step in range(grad_accum_steps):
            # Get batch
            if use_real_data and train_tokens is not None:
                (input_ids, labels), train_state = get_token_batch(
                    train_tokens, train_state, config.batch_size, config.max_seq_len
                )
                input_ids = input_ids.to(device)
                labels = labels.to(device)
            else:
                input_ids, labels = generate_random_batch(
                    config.batch_size,
                    config.max_seq_len,
                    config.model.vocab_size,
                    device,
                )

            # Forward, backward (scale loss for accumulation)
            loss = train_step(input_ids, labels, weights, config.model, autocast_ctx)
            # Scale gradients by 1/grad_accum_steps (loss.backward already computed grads)
            if grad_accum_steps > 1:
                for param in weights.values():
                    if param.grad is not None:
                        param.grad.div_(grad_accum_steps)
            accum_loss += loss.item() / grad_accum_steps

        # All-reduce gradients across ranks (no-op if single GPU)
        runtime.all_reduce_grads(weights)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(weights.values(), config.max_grad_norm)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Logging (rank 0 only)
        if runtime.is_main() and step % config.log_every == 0:
            step_time = time.time() - step_start
            # Tokens per second accounts for all ranks and accumulation
            tokens_per_sec = (effective_batch * config.max_seq_len) / step_time
            elapsed = time.time() - start_time

            logger.info(
                f"step={step:5d} | loss={accum_loss:.4f} | "
                f"lr={lr:.2e} | "
                f"grad_norm={grad_norm:.4f} | "
                f"tok/s={tokens_per_sec:.0f} | "
                f"elapsed={elapsed:.1f}s"
            )

        # Validation (rank 0 only)
        if (
            runtime.is_main()
            and use_real_data
            and val_tokens is not None
            and val_every > 0
            and step > 0
            and step % val_every == 0
        ):
            val_loss = eval_loss(
                val_tokens,
                weights,
                config.model,
                config.batch_size,
                config.max_seq_len,
                val_batches,
                device,
            )
            logger.info(f"step={step:5d} | val_loss={val_loss:.4f}")

        # Checkpoint (rank 0 only)
        if (
            runtime.is_main()
            and config.checkpoint_every > 0
            and step > 0
            and step % config.checkpoint_every == 0
        ):
            save_checkpoint(weights, optimizer, step, config, output_dir)

    # Final validation and checkpoint (rank 0 only)
    if runtime.is_main():
        if use_real_data and val_tokens is not None:
            val_loss = eval_loss(
                val_tokens,
                weights,
                config.model,
                config.batch_size,
                config.max_seq_len,
                val_batches,
                device,
            )
            logger.info(f"final val_loss={val_loss:.4f}")

        save_checkpoint(weights, optimizer, config.steps, config, output_dir)

        total_time = time.time() - start_time
        logger.info(f"training complete in {total_time:.1f}s")

    # Cleanup distributed state
    runtime.finalize()
