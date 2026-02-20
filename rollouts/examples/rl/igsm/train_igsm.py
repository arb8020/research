"""Training script for iGSM pretraining (clean data).

Adapted from the pretrain infrastructure to use iGSM synthetic data.

Usage:
    python -m examples.rl.igsm.pretrain_clean --tiny
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

from rollouts.pretrain.config import ModelConfig, TrainConfig, get_git_info
from rollouts.pretrain.models.llama import count_parameters, forward, init_weights
from rollouts.pretrain.optim import Muon, build_optimizers
from rollouts.pretrain.runtime import all_reduce_grads, finalize, init, is_main
from rollouts.pretrain.schedule import get_lr
from rollouts.synthetic import build_igsm_loader

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _make_forward_and_loss(
    config: ModelConfig,
    use_compile: bool = True,
) -> Callable[[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], torch.Tensor]:
    def forward_and_loss(
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        weights: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = forward(input_ids, weights, config)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            labels.reshape(-1),
        )
        return loss

    if use_compile and torch.cuda.is_available():
        return torch.compile(forward_and_loss, mode="reduce-overhead", dynamic=False)
    return forward_and_loss


def train_step(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    weights: dict[str, torch.Tensor],
    forward_and_loss_fn: Callable,
    autocast_ctx: torch.amp.autocast | None = None,
) -> torch.Tensor:
    if autocast_ctx is not None:
        with autocast_ctx:
            loss = forward_and_loss_fn(input_ids, labels, weights)
    else:
        loss = forward_and_loss_fn(input_ids, labels, weights)
    loss.backward()
    return loss


@torch.no_grad()
def eval_loss(
    loader,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    num_batches: int,
) -> float:
    total_loss = 0.0
    for _ in range(num_batches):
        input_ids, labels = loader.next()
        logits = forward(input_ids, weights, config)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            labels.reshape(-1),
        )
        total_loss += loss.item()
    return total_loss / num_batches


def save_config(config: TrainConfig, output_dir: Path, igsm_config: dict | None = None) -> None:
    git_hash, git_dirty = get_git_info()

    meta = {
        "config": asdict(config),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "fingerprint": config.fingerprint(),
    }

    if igsm_config is not None:
        meta["igsm"] = igsm_config

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
    train_loader=None,
) -> None:
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
    checkpoints = sorted(output_dir.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(
    ckpt_path: Path,
    weights: dict[str, torch.Tensor],
    muon_optimizer: Muon | None,
    adamw_optimizer: torch.optim.AdamW,
    config: TrainConfig,
    device: torch.device,
    train_loader=None,
) -> int:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if ckpt["fingerprint"] != config.fingerprint():
        raise ValueError(
            f"Checkpoint fingerprint mismatch: {ckpt['fingerprint']} != {config.fingerprint()}"
        )

    for key, value in ckpt["weights"].items():
        weights[key].copy_(value.to(device))

    if muon_optimizer is not None and ckpt.get("muon_optimizer") is not None:
        muon_optimizer.load_state_dict(ckpt["muon_optimizer"])
    adamw_optimizer.load_state_dict(ckpt["adamw_optimizer"])

    if train_loader is not None and ckpt.get("loader_state") is not None:
        train_loader.load_state_dict(ckpt["loader_state"])

    return ckpt["step"]


def train(
    config: TrainConfig,
    igsm_config: dict,
    resume: bool = False,
) -> None:
    """Main training function for iGSM pretraining.

    Args:
        config: Training configuration
        igsm_config: iGSM-specific configuration (difficulty, max_op, max_edge)
        resume: If True, resume from latest checkpoint
    """
    rank, world, device = init(seed=config.seed)
    setup_logging()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = None

    if is_main():
        logger.info(f"device: {device}, dtype: {dtype}, world: {world}")
        logger.info(f"iGSM config: {igsm_config}")

    output_dir = Path(config.output_dir)
    if config.run_id:
        output_dir = output_dir / config.run_id
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir, igsm_config)

    # Create iGSM data loaders
    train_loader = build_igsm_loader(
        difficulty=igsm_config.get("difficulty", "med"),
        seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        rank=rank,
        world_size=world,
        device=device,
        seed=config.seed,
        mode="train",
        max_op=igsm_config.get("max_op", 15),
        max_edge=igsm_config.get("max_edge", 20),
    )

    val_loader = build_igsm_loader(
        difficulty=igsm_config.get("difficulty", "med"),
        seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        rank=0,
        world_size=1,
        device=device,
        seed=config.seed + 1000000,
        mode="val",
        max_op=igsm_config.get("max_op", 15),
        max_edge=igsm_config.get("max_edge", 20),
    )

    # Model
    if is_main():
        logger.info("initializing model...")
    weights = init_weights(config.model, device, dtype)
    n_params = count_parameters(weights)
    if is_main():
        logger.info(f"parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    use_compile = config.use_compile and device.type == "cuda"
    forward_and_loss_fn = _make_forward_and_loss(config.model, use_compile=use_compile)
    if is_main():
        logger.info(f"torch.compile: {use_compile}")

    # Optimizers
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

    # Resume
    start_step = 0
    if resume:
        ckpt_path = find_latest_checkpoint(output_dir)
        if ckpt_path:
            start_step = load_checkpoint(
                ckpt_path, weights, muon_optimizer, adamw_optimizer, config, device, train_loader
            )
            if is_main():
                logger.info(f"resumed from {ckpt_path} at step {start_step}")

    schedule_config = {
        "steps": config.steps,
        "lr": 1.0,
        "warmup_steps": config.warmup_steps,
    }

    # Training loop
    grad_accum_steps = config.grad_accum_steps
    effective_batch = config.batch_size * grad_accum_steps * world

    if is_main():
        logger.info(f"starting training for {config.steps - start_step} steps...")
        logger.info(f"effective batch size: {effective_batch}")

    start_time = time.time()

    for step in range(start_step, config.steps):
        step_start = time.time()

        if step == start_step + 1:
            gc.collect()
            gc.freeze()
            gc.disable()

        lr_scale = get_lr(step, schedule_config)
        if muon_optimizer is not None:
            for param_group in muon_optimizer.param_groups:
                param_group["lr"] = config.lr_muon * lr_scale
        for param_group in adamw_optimizer.param_groups:
            param_group["lr"] = config.lr_adamw * lr_scale

        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            input_ids, labels = train_loader.next()
            loss = train_step(input_ids, labels, weights, forward_and_loss_fn, autocast_ctx)
            if grad_accum_steps > 1:
                for param in weights.values():
                    if param.grad is not None:
                        param.grad.div_(grad_accum_steps)
            accum_loss += loss.item() / grad_accum_steps

        all_reduce_grads(weights)
        grad_norm = torch.nn.utils.clip_grad_norm_(weights.values(), config.max_grad_norm)

        if muon_optimizer is not None:
            muon_optimizer.step()
            muon_optimizer.zero_grad()
        adamw_optimizer.step()
        adamw_optimizer.zero_grad()

        if is_main() and step % config.log_every == 0:
            step_time = time.time() - step_start
            tokens_per_sec = (effective_batch * config.max_seq_len) / step_time
            elapsed = time.time() - start_time
            logger.info(
                f"step={step:5d} | loss={accum_loss:.4f} | "
                f"lr_scale={lr_scale:.2e} | grad_norm={grad_norm:.4f} | "
                f"tok/s={tokens_per_sec:.0f} | elapsed={elapsed:.1f}s"
            )

        if is_main() and config.val_every > 0 and step > 0 and step % config.val_every == 0:
            val_loss = eval_loss(val_loader, weights, config.model, config.val_batches)
            logger.info(f"step={step:5d} | val_loss={val_loss:.4f}")

        if (
            is_main()
            and config.checkpoint_every > 0
            and step > 0
            and step % config.checkpoint_every == 0
        ):
            save_checkpoint(
                weights, muon_optimizer, adamw_optimizer, step, config, output_dir, train_loader
            )

    if is_main():
        val_loss = eval_loss(val_loader, weights, config.model, config.val_batches)
        logger.info(f"final val_loss={val_loss:.4f}")
        save_checkpoint(
            weights, muon_optimizer, adamw_optimizer, config.steps, config, output_dir, train_loader
        )
        logger.info(f"training complete in {time.time() - start_time:.1f}s")

    finalize()
