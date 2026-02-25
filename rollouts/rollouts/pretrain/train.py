"""Training loop for pretraining.

Simple, explicit training loop with logging. No hidden magic.

Usage:
    python rollouts/pretrain/configs/tiny.py
    python rollouts/pretrain/configs/small.py --real-data --resume
    torchrun --standalone --nproc_per_node=8 rollouts/pretrain/configs/small.py
"""

from __future__ import annotations

import gc
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from . import runtime
from .config import ModelConfig, TrainConfig, get_git_info
from .dataloader import DeterministicLoader, build_loader
from .models.llama import count_parameters, forward, init_weights
from .optim import Muon, build_optimizers
from .schedule import get_lr

logger = logging.getLogger(__name__)


def _make_forward_and_loss(
    config: ModelConfig,
    use_compile: bool = True,
    use_fp8: bool = False,
) -> Callable[[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], torch.Tensor]:
    """Create a forward+loss function, optionally compiled and with FP8.

    Fusing forward and loss into one function gives torch.compile more room
    to optimize (e.g., fuse softmax into cross-entropy).
    """
    # Select linear function (FP8 or standard)
    if use_fp8:
        from .fp8 import fp8_linear, is_fp8_available

        if not is_fp8_available():
            logger.warning("FP8 requested but not available (requires H100+), falling back to bf16")
            linear_fn = F.linear
        else:
            linear_fn = fp8_linear
    else:
        linear_fn = F.linear

    def forward_and_loss(
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        weights: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = forward(input_ids, weights, config, linear_fn)
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
        )
        return loss

    if use_compile and torch.cuda.is_available():
        # mode="reduce-overhead" uses CUDA graphs for lower kernel launch overhead
        # dynamic=False because shapes are fixed (batch_size, seq_len)
        return torch.compile(forward_and_loss, mode="reduce-overhead", dynamic=False)
    return forward_and_loss


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
    forward_and_loss_fn: Callable[
        [torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], torch.Tensor
    ],
    autocast_ctx: torch.amp.autocast | None = None,
) -> torch.Tensor:
    """Single training step: forward, loss, backward.

    Returns loss (still attached to graph - caller handles optimizer step).
    """
    if autocast_ctx is not None:
        with autocast_ctx:
            loss = forward_and_loss_fn(input_ids, labels, weights)
    else:
        loss = forward_and_loss_fn(input_ids, labels, weights)
    loss.backward()
    return loss


@torch.no_grad()
def eval_loss(
    loader: DeterministicLoader,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    num_batches: int,
) -> float:
    """Compute average loss over validation data.

    Args:
        loader: Validation data loader
        weights: Model weights
        config: Model config
        num_batches: Number of batches to evaluate

    Returns:
        Average cross-entropy loss
    """
    total_loss = 0.0

    for _ in range(num_batches):
        input_ids, labels = loader.next()

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
    muon_optimizer: Muon | None,
    adamw_optimizer: torch.optim.AdamW,
    step: int,
    config: TrainConfig,
    output_dir: Path,
    train_loader: DeterministicLoader | None = None,
) -> None:
    """Save training checkpoint."""
    ckpt_path = output_dir / f"step_{step:08d}.pt"
    torch.save(
        {
            "weights": {k: v.cpu() for k, v in weights.items()},
            "muon_optimizer": muon_optimizer.state_dict() if muon_optimizer else None,
            "adamw_optimizer": adamw_optimizer.state_dict(),
            "step": step,
            "fingerprint": config.fingerprint(),
            "loader_state": train_loader.state_dict() if train_loader else None,
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
    muon_optimizer: Muon | None,
    adamw_optimizer: torch.optim.AdamW,
    config: TrainConfig,
    device: torch.device,
    train_loader: DeterministicLoader | None = None,
) -> int:
    """Load checkpoint into weights and optimizers.

    Args:
        ckpt_path: Path to checkpoint file
        weights: Model weights dict (modified in place)
        muon_optimizer: Muon optimizer (state loaded in place)
        adamw_optimizer: AdamW optimizer (state loaded in place)
        config: Config for fingerprint verification
        device: Device to load weights to
        train_loader: Data loader (state loaded in place)

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

    # Load optimizer states
    if muon_optimizer is not None and ckpt.get("muon_optimizer") is not None:
        muon_optimizer.load_state_dict(ckpt["muon_optimizer"])
    adamw_optimizer.load_state_dict(ckpt["adamw_optimizer"])

    # Load loader state
    if train_loader is not None and ckpt.get("loader_state") is not None:
        train_loader.load_state_dict(ckpt["loader_state"])

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

    # Load data
    train_loader: DeterministicLoader | None = None
    val_loader: DeterministicLoader | None = None

    if use_real_data:
        if not config.data_pattern:
            raise ValueError("use_real_data=True but config.data_pattern is empty")
        if runtime.is_main():
            logger.info(f"loading data from {config.data_pattern}...")

        train_loader = build_loader(
            sources=config.data_pattern,
            seq_len=config.max_seq_len,
            batch_size=config.batch_size,
            rank=rank,
            world_size=world,
            device=device,
        )
        # Val loader uses same source but no distributed slicing
        val_loader = build_loader(
            sources=config.data_pattern,
            seq_len=config.max_seq_len,
            batch_size=config.batch_size,
            rank=0,
            world_size=1,
            device=device,
        )

    # Model
    if runtime.is_main():
        logger.info("initializing model...")
    weights = init_weights(config.model, device, dtype)
    n_params = count_parameters(weights)
    if runtime.is_main():
        logger.info(f"parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Compiled forward+loss (torch.compile on CUDA, plain on CPU)
    use_compile = config.use_compile and device.type == "cuda"
    use_fp8 = config.use_fp8 and device.type == "cuda"
    forward_and_loss_fn = _make_forward_and_loss(
        config.model, use_compile=use_compile, use_fp8=use_fp8
    )
    if runtime.is_main():
        logger.info(f"torch.compile: {use_compile}, fp8: {use_fp8}")

    # Optimizers (Muon for 2D matrices, AdamW for embeddings/norms)
    muon_optimizer, adamw_optimizer = build_optimizers(
        weights,
        lr_muon=config.lr_muon,
        lr_adamw=config.lr_adamw,
        momentum=config.muon_momentum,
        weight_decay=config.weight_decay,
        betas=config.adam_betas,
        eps=config.adam_eps,
        use_muon=config.use_muon,
    )
    if runtime.is_main():
        if muon_optimizer is not None:
            n_muon = sum(p.numel() for g in muon_optimizer.param_groups for p in g["params"])
            logger.info(f"muon params: {n_muon:,}")
        n_adamw = sum(p.numel() for g in adamw_optimizer.param_groups for p in g["params"])
        logger.info(f"adamw params: {n_adamw:,}")

    # Resume from checkpoint
    start_step = 0
    if resume:
        ckpt_path = find_latest_checkpoint(output_dir)
        if ckpt_path:
            start_step = load_checkpoint(
                ckpt_path, weights, muon_optimizer, adamw_optimizer, config, device, train_loader
            )
            if runtime.is_main():
                logger.info(f"resumed from {ckpt_path} at step {start_step}")
        elif runtime.is_main():
            logger.info("no checkpoint found, starting from scratch")

    # Schedule config (dict for get_lr)
    schedule_config = {
        "steps": config.steps,
        "lr": 1.0,  # We'll scale Muon and AdamW LRs separately
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

        # Disable GC after first step to avoid ~500ms pauses during training
        # (setup objects are frozen and excluded from future scans)
        if step == start_step + 1:
            gc.collect()
            gc.freeze()
            gc.disable()

        # Update learning rate (scale factor from schedule, applied to base LRs)
        lr_scale = get_lr(step, schedule_config)
        if muon_optimizer is not None:
            for param_group in muon_optimizer.param_groups:
                param_group["lr"] = config.lr_muon * lr_scale
        for param_group in adamw_optimizer.param_groups:
            param_group["lr"] = config.lr_adamw * lr_scale

        # Gradient accumulation loop
        accum_loss = 0.0
        for accum_step in range(grad_accum_steps):
            # Get batch
            if train_loader is not None:
                input_ids, labels = train_loader.next()
            else:
                input_ids, labels = generate_random_batch(
                    config.batch_size,
                    config.max_seq_len,
                    config.model.vocab_size,
                    device,
                )

            # Forward, backward (scale loss for accumulation)
            loss = train_step(input_ids, labels, weights, forward_and_loss_fn, autocast_ctx)
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

        # Optimizer steps
        if muon_optimizer is not None:
            muon_optimizer.step()
            muon_optimizer.zero_grad()
        adamw_optimizer.step()
        adamw_optimizer.zero_grad()

        # Logging (rank 0 only)
        if runtime.is_main() and step % config.log_every == 0:
            step_time = time.time() - step_start
            # Tokens per second accounts for all ranks and accumulation
            tokens_per_sec = (effective_batch * config.max_seq_len) / step_time
            elapsed = time.time() - start_time

            logger.info(
                f"step={step:5d} | loss={accum_loss:.4f} | "
                f"lr_scale={lr_scale:.2e} | "
                f"grad_norm={grad_norm:.4f} | "
                f"tok/s={tokens_per_sec:.0f} | "
                f"elapsed={elapsed:.1f}s"
            )

        # Validation (rank 0 only)
        if (
            runtime.is_main()
            and val_loader is not None
            and val_every > 0
            and step > 0
            and step % val_every == 0
        ):
            val_loss = eval_loss(val_loader, weights, config.model, val_batches)
            # TODO: log BPB (bits per byte) alongside loss
            #   bpb = val_loss / ln(2) * (tokens / bytes)
            #   Requires token_bytes mapping from tokenizer (see nmoe/token_bytes.py)
            logger.info(f"step={step:5d} | val_loss={val_loss:.4f}")

        # Checkpoint (rank 0 only)
        if (
            runtime.is_main()
            and config.checkpoint_every > 0
            and step > 0
            and step % config.checkpoint_every == 0
        ):
            save_checkpoint(
                weights, muon_optimizer, adamw_optimizer, step, config, output_dir, train_loader
            )

    # Final validation and checkpoint (rank 0 only)
    if runtime.is_main():
        if val_loader is not None:
            val_loss = eval_loss(val_loader, weights, config.model, val_batches)
            logger.info(f"final val_loss={val_loss:.4f}")

        save_checkpoint(
            weights, muon_optimizer, adamw_optimizer, config.steps, config, output_dir, train_loader
        )

        total_time = time.time() - start_time
        logger.info(f"training complete in {total_time:.1f}s")

    # Cleanup distributed state
    runtime.finalize()
