"""Training script for CFG-based synthetic pretraining.

Usage:
    python train_cfg.py configs/cfg_tiny.py
    python train_cfg.py configs/cfg_small.py
    torchrun --standalone --nproc_per_node=8 train_cfg.py configs/cfg_small.py
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

# Import from rollouts pretrain module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rollouts"))

from rollouts.pretrain.config import ModelConfig, TrainConfig, get_git_info
from rollouts.pretrain.models.llama import count_parameters, forward, init_weights
from rollouts.pretrain.optim import Muon, build_optimizers
from rollouts.pretrain.runtime import all_reduce_grads, finalize, init, is_main
from rollouts.pretrain.schedule import get_lr

# Import our CFG dataloader
from cfg_generator import build_cfg_loader, CFGConfig

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for training."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _make_forward_and_loss(
    config: ModelConfig,
    use_compile: bool = True,
) -> Callable[[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], torch.Tensor]:
    """Create a forward+loss function, optionally compiled."""
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
    """Single training step: forward, loss, backward."""
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
    """Compute average loss over validation data."""
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


def save_config(config: TrainConfig, output_dir: Path, cfg_config: CFGConfig | None = None) -> None:
    """Save config + git info for reproducibility."""
    git_hash, git_dirty = get_git_info()
    
    meta = {
        "config": asdict(config),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "fingerprint": config.fingerprint(),
    }
    
    if cfg_config is not None:
        meta["cfg"] = {
            "depth": cfg_config.depth,
            "num_sym": cfg_config.num_sym,
            "vocab_size": cfg_config.vocab_size,
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
    train_loader=None,
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
    train_loader=None,
) -> int:
    """Load checkpoint into weights and optimizers."""
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


def train(
    config: TrainConfig,
    cfg_path: str | Path,
    resume: bool = False,
) -> None:
    """Main training function for CFG-based pretraining.
    
    Args:
        config: Training configuration
        cfg_path: Path to CFG JSON file
        resume: If True, resume from latest checkpoint
    """
    # Setup distributed runtime
    rank, world, device = init(seed=config.seed)
    setup_logging()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    # Mixed precision autocast
    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = None
    
    if is_main():
        logger.info(f"device: {device}, dtype: {dtype}, world: {world}")
        logger.info(f"cfg_path: {cfg_path}")
    
    # Output directory
    output_dir = Path(config.output_dir)
    if config.run_id:
        output_dir = output_dir / config.run_id
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CFG config
    cfg_config = CFGConfig.from_graph(cfg_path)
    
    # Update model vocab_size to match CFG
    model_config = ModelConfig(
        dim=config.model.dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        n_kv_heads=config.model.n_kv_heads,
        head_dim=config.model.head_dim,
        mlp_dim=config.model.mlp_dim,
        vocab_size=cfg_config.vocab_size + 4,  # +4 for special tokens
        rope_theta=config.model.rope_theta,
        rms_norm_eps=config.model.rms_norm_eps,
    )
    
    # Create a new config with updated model
    train_config = TrainConfig(
        model=model_config,
        data_pattern="",  # Not used for CFG
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        grad_accum_steps=config.grad_accum_steps,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        use_muon=config.use_muon,
        lr_muon=config.lr_muon,
        muon_momentum=config.muon_momentum,
        lr_adamw=config.lr_adamw,
        adam_betas=config.adam_betas,
        adam_eps=config.adam_eps,
        steps=config.steps,
        log_every=config.log_every,
        checkpoint_every=config.checkpoint_every,
        val_every=config.val_every,
        val_batches=config.val_batches,
        use_compile=config.use_compile,
        use_fp8=config.use_fp8,
        output_dir=config.output_dir,
        run_id=config.run_id,
        seed=config.seed,
    )
    
    if is_main():
        save_config(train_config, output_dir, cfg_config)
    
    # Create data loaders
    train_loader = build_cfg_loader(
        cfg_path=cfg_path,
        seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        rank=rank,
        world_size=world,
        device=device,
        seed=config.seed,
    )
    
    # Val loader uses same source but no distributed slicing
    val_loader = build_cfg_loader(
        cfg_path=cfg_path,
        seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        rank=0,
        world_size=1,
        device=device,
        seed=config.seed + 1000000,  # Different seed for val
    )
    
    # Model
    if is_main():
        logger.info("initializing model...")
    weights = init_weights(model_config, device, dtype)
    n_params = count_parameters(weights)
    if is_main():
        logger.info(f"parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    
    # Compiled forward+loss
    use_compile = config.use_compile and device.type == "cuda"
    forward_and_loss_fn = _make_forward_and_loss(model_config, use_compile=use_compile)
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
    if is_main():
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
                ckpt_path, weights, muon_optimizer, adamw_optimizer, train_config, device, train_loader
            )
            if is_main():
                logger.info(f"resumed from {ckpt_path} at step {start_step}")
        elif is_main():
            logger.info("no checkpoint found, starting from scratch")
    
    # Schedule config
    schedule_config = {
        "steps": config.steps,
        "lr": 1.0,
        "warmup_steps": config.warmup_steps,
    }
    
    # Training loop
    remaining_steps = config.steps - start_step
    grad_accum_steps = config.grad_accum_steps
    effective_batch = config.batch_size * grad_accum_steps * world
    
    if is_main():
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
        
        # Disable GC after first step
        if step == start_step + 1:
            gc.collect()
            gc.freeze()
            gc.disable()
        
        # Update learning rate
        lr_scale = get_lr(step, schedule_config)
        if muon_optimizer is not None:
            for param_group in muon_optimizer.param_groups:
                param_group["lr"] = config.lr_muon * lr_scale
        for param_group in adamw_optimizer.param_groups:
            param_group["lr"] = config.lr_adamw * lr_scale
        
        # Gradient accumulation loop
        accum_loss = 0.0
        for accum_step in range(grad_accum_steps):
            input_ids, labels = train_loader.next()
            
            loss = train_step(input_ids, labels, weights, forward_and_loss_fn, autocast_ctx)
            if grad_accum_steps > 1:
                for param in weights.values():
                    if param.grad is not None:
                        param.grad.div_(grad_accum_steps)
            accum_loss += loss.item() / grad_accum_steps
        
        # All-reduce gradients
        all_reduce_grads(weights)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(weights.values(), config.max_grad_norm)
        
        # Optimizer steps
        if muon_optimizer is not None:
            muon_optimizer.step()
            muon_optimizer.zero_grad()
        adamw_optimizer.step()
        adamw_optimizer.zero_grad()
        
        # Logging
        if is_main() and step % config.log_every == 0:
            step_time = time.time() - step_start
            tokens_per_sec = (effective_batch * config.max_seq_len) / step_time
            elapsed = time.time() - start_time
            
            logger.info(
                f"step={step:5d} | loss={accum_loss:.4f} | "
                f"lr_scale={lr_scale:.2e} | "
                f"grad_norm={grad_norm:.4f} | "
                f"tok/s={tokens_per_sec:.0f} | "
                f"elapsed={elapsed:.1f}s"
            )
        
        # Validation
        if is_main() and val_every > 0 and step > 0 and step % val_every == 0:
            val_loss = eval_loss(val_loader, weights, model_config, val_batches)
            logger.info(f"step={step:5d} | val_loss={val_loss:.4f}")
        
        # Checkpoint
        if is_main() and config.checkpoint_every > 0 and step > 0 and step % config.checkpoint_every == 0:
            save_checkpoint(weights, muon_optimizer, adamw_optimizer, step, train_config, output_dir, train_loader)
    
    # Final validation and checkpoint
    if is_main():
        if val_loader is not None:
            val_loss = eval_loss(val_loader, weights, model_config, val_batches)
            logger.info(f"final val_loss={val_loss:.4f}")
        
        save_checkpoint(weights, muon_optimizer, adamw_optimizer, config.steps, train_config, output_dir, train_loader)
        
        total_time = time.time() - start_time
        logger.info(f"training complete in {total_time:.1f}s")
    
    # Cleanup
    finalize()


if __name__ == "__main__":
    import argparse
    import importlib.util
    
    parser = argparse.ArgumentParser(description="CFG-based pretraining")
    parser.add_argument("config", help="Path to config .py file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config from file
    spec = importlib.util.spec_from_file_location("train_config", args.config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    config = module.config
    cfg_path = module.cfg_path
    
    train(config, cfg_path, resume=args.resume)
