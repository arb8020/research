#!/usr/bin/env python3
"""Post-training orchestration using rollouts/ module.

This script is like dev/outlier-features/deploy.py but for local training.
It loads a config, sets up training, and runs SFT or RL using rollouts/.

Usage:
    # SFT training
    python deploy.py configs/01_debug_sft.py --mode sft

    # RL training
    python deploy.py configs/03_debug_rl_04.py --mode rl --source-checkpoint outputs/exp_001/checkpoint_500.pt

Following:
- Tiger Style: Assertions, explicit control flow
- Casey Muratori: Pure functions, explicit dependencies
- nanochat: Proven training approach

Split into functions <70 lines (Tiger Style).
"""

import importlib.util
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import trio

# Import local config (from same directory - no sys.path needed!)
from base_config import Config

# Import from rollouts module (installed via workspace)
from rollouts.training import (
    JSONLLogger,
    PyTorchTrainingBackend,
    SFTTrainingConfig,
    load_sft_dataset,
    run_sft_training,
)

# Import shared logging
from shared.logging_config import setup_logging

logger = logging.getLogger(__name__)


def load_config_from_file(config_path: str) -> Config:
    """Load config from Python file.

    Args:
        config_path: Path to config .py file

    Returns:
        Config object

    Tiger Style: Assert preconditions.
    """
    assert config_path.endswith('.py'), f"Config must be .py file, got {config_path}"
    assert Path(config_path).exists(), f"Config file not found: {config_path}"

    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    assert spec is not None, f"Failed to load spec from {config_path}"
    assert spec.loader is not None, f"Spec loader is None for {config_path}"

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'config'), "Config file must define 'config' variable"
    config: Config = getattr(module, 'config')
    assert isinstance(config, Config), f"Expected Config object, got {type(config)}"

    return config


def setup_logging_from_config(config: Config):
    """Setup logging configuration from Config object.

    Args:
        config: Configuration object

    Uses shared logging infrastructure with:
    - Rich console output (colored, formatted)
    - Async-safe QueueHandler
    - File rotation support
    """
    # Use shared logging setup
    setup_logging(
        level=config.output.log_level.upper(),
        use_rich=True,  # Nice colored output
        rich_tracebacks=True,  # Better error messages
        use_queue_handler=True,  # Async-safe
    )


def load_tokenizer(model_name: str):
    """Load HuggingFace tokenizer.

    Args:
        model_name: Model name on HuggingFace

    Returns:
        Tokenizer instance

    Tiger Style: Explicit imports, no hidden dependencies.
    """
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure we have required special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(model_name: str, device_type: str, dtype: str, gpu_ranks: list[int]):
    """Load HuggingFace model.

    Args:
        model_name: Model name on HuggingFace
        device_type: Device type (cuda|cpu|mps)
        dtype: Data type (bfloat16|float32)
        gpu_ranks: GPU indices to use

    Returns:
        Model instance

    Tiger Style: Explicit control flow, assertions.
    """
    import torch
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading model: {model_name}")
    logger.info(f"  Device: {device_type}")
    logger.info(f"  Dtype: {dtype}")
    logger.info(f"  GPU ranks: {gpu_ranks}")

    # Tiger Style: Assert valid inputs
    assert device_type in ["cuda", "cpu", "mps"], f"Invalid device: {device_type}"
    assert dtype in ["bfloat16", "float32"], f"Invalid dtype: {dtype}"
    assert len(gpu_ranks) > 0, "gpu_ranks cannot be empty"

    # Tiger Style: Warn about multi-GPU limitations
    if len(gpu_ranks) > 1:
        logger.warning(f"  gpu_ranks has {len(gpu_ranks)} GPUs but only gpu_ranks[0]={gpu_ranks[0]} will be used")
        logger.warning(f"  Multi-GPU training requires DataParallel/DDP (not yet implemented)")

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    # Note: We DON'T set CUDA_VISIBLE_DEVICES here
    # Instead, we use explicit device placement (cuda:4 means physical GPU 4)
    # This is clearer and avoids remapping confusion on shared nodes
    # The backend will place model/data on the exact GPU index we specify

    # Explicit device placement: use first GPU in gpu_ranks
    # device_map="auto" would use CUDA_VISIBLE_DEVICES (confusing on shared nodes)
    # Instead, explicitly place on the specified physical GPU
    #
    # Note: Current code is single-GPU only. If gpu_ranks=[4,5,6,7], only GPU 4 is used.
    # For multi-GPU training, would need DataParallel/DDP or tensor parallelism.
    # Using explicit GPU index (not CUDA_VISIBLE_DEVICES) works with non-contiguous
    # allocations like gpu_ranks=[0,2,4,6] - each refers to physical GPU index.
    if device_type == "cuda":
        device_map = {"": gpu_ranks[0]}  # Place entire model on first allocated GPU
    else:
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    if device_type != "cuda":
        model = model.to(device_type)

    return model


def create_optimizer(model, config: Config, mode: Literal["sft", "rl"]):
    """Create optimizer with nanochat's tiered learning rates.

    Args:
        model: PyTorch model
        config: Configuration object
        mode: Training mode (sft|rl)

    Returns:
        Optimizer instance

    nanochat uses tiered LRs: different rates for embedding, unembedding, and matrices.
    For simplicity, we use a single AdamW optimizer here.
    """
    import torch

    # Get config for mode
    train_config = config.sft if mode == "sft" else config.rl

    # Tiger Style: Use explicit LR (nanochat's matrix_lr as default)
    lr = train_config.matrix_lr

    # Tiger Style: All optimizer parameters explicit from config
    logger.info(f"Creating AdamW optimizer with LR={lr}, betas=({train_config.adam_beta1}, {train_config.adam_beta2})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        eps=train_config.adam_eps,
        weight_decay=train_config.weight_decay,
    )

    return optimizer


def create_loss_fn():
    """Create loss function for training.

    Returns:
        Loss function

    Casey Muratori: Pure function, explicit.
    """
    import torch.nn.functional as F

    def cross_entropy_loss(logits, batch):
        """Compute cross-entropy loss with optional masking.

        Following SLIME's pattern: pass entire batch dict, extract fields inside.

        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            batch: Training batch dict containing:
                - labels: Target labels [batch, seq_len]
                - loss_mask: Loss mask [batch, seq_len] (optional)
                - advantages: For RL training (optional, unused for SFT)

        Returns:
            Scalar loss
        """
        # Extract fields from batch (SLIME pattern)
        labels = batch["labels"]
        loss_mask = batch.get("loss_mask")
        # advantages = batch.get("advantages")  # Available but unused for SFT

        # Reshape for cross_entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Compute loss
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            reduction='none'
        )
        loss = loss.view(batch_size, seq_len)

        # Apply mask if provided
        if loss_mask is not None:
            loss = loss * loss_mask
            # Average over valid tokens
            num_valid = loss_mask.sum().clamp(min=1.0)
            return loss.sum() / num_valid
        else:
            return loss.mean()

    return cross_entropy_loss


async def create_fsdp_backend(config: Config, output_dir: Path):
    """Create FSDP backend for multi-GPU training.

    Args:
        config: Configuration object
        output_dir: Output directory

    Returns:
        FSDPTrainingBackend instance

    Tiger Style: Explicit initialization, clear error messages.

    Note: This function assumes torch.distributed is already initialized by torchrun.
    For FSDP training, launch with:
        torchrun --nproc_per_node=4 train.py configs/02_debug_sft_fsdp.py
    """
    import torch
    import torch.distributed as dist
    from rollouts.training.backends.fsdp import FSDPConfig, FSDPTrainingBackend

    # Tiger Style: Assert torch.distributed is initialized
    if not dist.is_available():
        raise RuntimeError(
            "torch.distributed is not available. "
            "Please install PyTorch with distributed support."
        )

    if not dist.is_initialized():
        # Initialize torch.distributed for FSDP
        # Set environment variables for distributed training
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        logger.info(f"Initializing torch.distributed: rank={rank}/{world_size}, local_rank={local_rank}")

        # Initialize process group
        dist.init_process_group(backend="nccl")

        # Set CUDA device for this rank
        torch.cuda.set_device(local_rank)

    # Get distributed info
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    logger.info(f"[Rank {rank}/{world_size}] Creating FSDP backend on GPU {local_rank}")

    # Load model (will be wrapped by FSDP)
    model = load_model(
        config.model.name,
        config.target.device_type,
        config.model.dtype,
        [local_rank],  # Each rank uses its local GPU
    )

    # Create optimizer (after model is on device, before FSDP wrapping)
    optimizer = create_optimizer(model, config, mode="sft")

    # Create loss function
    loss_fn = create_loss_fn()

    # Create FSDP config (Tiger Style: explicit parameters from user config)
    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        mixed_precision=True,
        cpu_offload=False,
        auto_wrap_min_params=1_000_000,
        gradient_checkpointing=False,
        clip_grad=config.sft.clip_grad,
    )

    # Create FSDP backend
    backend = FSDPTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        checkpoint_dir=output_dir / "checkpoints",
        config=fsdp_config,
    )

    # Create learning rate scheduler with warmup (SLIME pattern)
    # Warmup prevents initial instability from large learning rate
    # Tiger Style: All parameters from config
    total_steps = config.sft.num_iterations
    warmup_ratio = config.sft.warmup_ratio
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    decay_steps = total_steps - warmup_steps

    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    # Warmup: ramp from 10% to 100% of base LR
    # Note: verbose param added in PyTorch 2.5+, omit for compatibility
    warmup = LinearLR(
        optimizer=optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
        last_epoch=-1,
    )

    # Decay: uses lr_decay_style from config
    if config.sft.lr_decay_style == "cosine":
        decay = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=decay_steps,
            eta_min=config.sft.matrix_lr * 0.1,
            last_epoch=-1,
        )
    elif config.sft.lr_decay_style == "linear":
        decay = LinearLR(
            optimizer=optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=decay_steps,
            last_epoch=-1,
        )
    elif config.sft.lr_decay_style == "constant":
        # No decay - just use a dummy scheduler
        from torch.optim.lr_scheduler import LambdaLR
        decay = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: 1.0,
            last_epoch=-1,
        )
    else:
        raise ValueError(f"Unknown lr_decay_style: {config.sft.lr_decay_style}")

    # Combine: warmup then decay
    backend.scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup, decay],
        milestones=[warmup_steps],
        last_epoch=-1,
    )

    logger.info(f"[Rank {rank}/{world_size}] LR scheduler: {warmup_steps} warmup steps + {config.sft.lr_decay_style} decay")
    logger.info(f"[Rank {rank}/{world_size}] FSDP backend created")

    return backend


async def run_sft(config: Config, output_dir: Path):
    """Run SFT training.

    Args:
        config: Configuration object
        output_dir: Output directory for results

    Tiger Style: <70 lines, clear control flow.
    """
    logger.info("=" * 60)
    logger.info("Starting SFT Training")
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = load_tokenizer(config.model.name)

    # Check if we should use FSDP (multi-GPU training)
    use_fsdp = (config.target.train_backend == "fsdp" and
                len(config.target.gpu_ranks) > 1)

    if use_fsdp:
        # FSDP multi-GPU training
        backend = await create_fsdp_backend(config, output_dir)
    else:
        # Single-GPU PyTorch backend
        model = load_model(
            config.model.name,
            config.target.device_type,
            config.model.dtype,
            config.target.gpu_ranks,
        )
        optimizer = create_optimizer(model, config, mode="sft")
        loss_fn = create_loss_fn()

        # Create backend
        import torch
        device = torch.device(f"{config.target.device_type}:{config.target.gpu_ranks[0]}"
                             if config.target.device_type == "cuda"
                             else config.target.device_type)

        backend = PyTorchTrainingBackend(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            checkpoint_dir=output_dir / "checkpoints",
            device=device,
        )

    # Load SFT data mixture (TaskMixture-style)
    logger.info("Loading SFT data mixture:")
    samples = []
    for spec in config.data.sft_mixture:
        logger.info(f"  Loading {spec.name} (split={spec.split}, subset={spec.subset}, max={spec.max_samples})")

        # Load dataset based on spec
        dataset_samples = load_sft_dataset(
            spec.name,
            tokenizer=tokenizer,
            split=spec.split,
            subset=spec.subset,
            max_samples=spec.max_samples,
            max_length=config.data.max_length,
        )

        # Handle repeat (nanochat does identity_conversations twice)
        for _ in range(spec.repeat):
            samples.extend(dataset_samples)

        total_samples = len(dataset_samples) * spec.repeat
        logger.info(f"    Loaded {len(dataset_samples)} samples (x{spec.repeat} = {total_samples})")

    logger.info(f"Total training samples: {len(samples)}")
    assert len(samples) > 0, "No samples loaded from mixture!"

    # Create training config
    sft_config = SFTTrainingConfig(
        num_steps=config.sft.num_iterations if config.sft.num_iterations > 0
                 else (len(samples) // config.sft.target_examples_per_step) * config.sft.num_epochs,
        batch_size=config.sft.batch_size,
        log_every=config.sft.log_every,
        checkpoint_every=config.sft.checkpoint_every,
    )

    # Create metrics logger
    metrics_logger = JSONLLogger(output_dir / "metrics") if config.output.use_wandb else None

    # Run SFT training loop (from rollouts/training/sft_loop.py)
    logger.info("Running SFT training loop...")
    metrics = await run_sft_training(
        backend=backend,
        samples=samples,
        config=sft_config,
        metrics_logger=metrics_logger,
    )

    logger.info("=" * 60)
    logger.info("SFT Training Complete!")
    logger.info(f"  Total steps: {len(metrics)}")
    logger.info(f"  Final loss: {metrics[-1]['loss']:.4f}")
    logger.info("=" * 60)

    return metrics


async def run_rl(config: Config, output_dir: Path, source_checkpoint: str | None):
    """Run RL training.

    Args:
        config: Configuration object
        output_dir: Output directory for results
        source_checkpoint: Path to SFT checkpoint (optional)

    Tiger Style: <70 lines, clear control flow.
    """
    logger.info("=" * 60)
    logger.info("Starting RL Training")
    logger.info("=" * 60)

    # Load model and tokenizer
    load_tokenizer(config.model.name)

    if source_checkpoint:
        logger.info(f"Loading from checkpoint: {source_checkpoint}")
        # TODO: Implement checkpoint loading
        raise NotImplementedError("Checkpoint loading not yet implemented")
    else:
        model = load_model(
            config.model.name,
            config.target.device_type,
            config.model.dtype,
            config.target.gpu_ranks,
        )

    optimizer = create_optimizer(model, config, mode="rl")
    loss_fn = create_loss_fn()

    # Create backend
    # Tiger Style: Explicit device handling
    import torch
    device = torch.device(f"{config.target.device_type}:{config.target.gpu_ranks[0]}"
                         if config.target.device_type == "cuda"
                         else config.target.device_type)

    PyTorchTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        checkpoint_dir=output_dir / "checkpoints",
        device=device,
    )

    # Load RL dataset (GSM8K)
    logger.info(f"Loading RL dataset: {config.data.rl_dataset}")
    # TODO: Implement RL dataset loading and setup
    # This requires implementing a reward function and rollout generation
    raise NotImplementedError("RL training not yet fully implemented - needs reward function and rollout generation")


def main():
    """Main orchestrator.

    Usage:
        python train.py configs/01_debug_sft.py
        python train.py configs/02_debug_sft_fsdp.py

    Mode (sft|rl) is auto-detected from config.output.mode.
    FSDP is auto-detected and re-launches with torchrun if needed.

    Tiger Style: Clear control flow, explicit steps.
    """
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(
        description="Post-training with rollouts/",
        epilog="Example: python train.py configs/01_debug_sft.py"
    )
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    # Tiger Style: Assert valid inputs
    config_path = Path(args.config)
    assert config_path.exists(), f"Config file not found: {args.config}"

    # Load config
    config = load_config_from_file(str(config_path))

    # Check if FSDP multi-GPU and not already launched by torchrun
    use_fsdp = (config.target.train_backend == "fsdp" and
                len(config.target.gpu_ranks) > 1)
    already_distributed = os.environ.get("RANK") is not None

    if use_fsdp and not already_distributed:
        # Auto-relaunch with torchrun
        # Note: GPU preflight check is done by deploy.py before calling train.py
        logger.info("=" * 60)
        logger.info("FSDP backend detected - relaunching with torchrun")
        logger.info(f"GPUs: {config.target.gpu_ranks}")
        logger.info("=" * 60)

        nproc = len(config.target.gpu_ranks)
        gpu_list = ",".join(str(r) for r in config.target.gpu_ranks)

        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            "--nnodes=1",
            "--node_rank=0",
            f"--master_addr={config.target.master_addr}",
            f"--master_port={config.target.master_port}",
            sys.argv[0],  # train.py
            str(config_path),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list

        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)

    setup_logging_from_config(config)

    # Extract mode from config
    mode = config.output.mode
    assert mode in ["sft", "rl", "sft+rl"], f"Invalid mode in config: {mode}. Must be 'sft', 'rl', or 'sft+rl'"

    # Validate checkpoint path for RL
    if mode == "rl" and config.output.source_checkpoint:
        checkpoint_path = Path(config.output.source_checkpoint)
        assert checkpoint_path.exists(), f"Checkpoint not found: {config.output.source_checkpoint}"

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.output.experiment_name or f"{mode}_{timestamp}"
    output_dir = config.output.save_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Save config to output directory
    config.save(output_dir / "config.json")
    logger.info(f"Saved config to {output_dir / 'config.json'}")

    # Run training (mode auto-detected from config)
    if mode == "sft":
        trio.run(run_sft, config, output_dir)
    elif mode == "rl":
        trio.run(run_rl, config, output_dir, config.output.source_checkpoint)
    elif mode == "sft+rl":
        # Run SFT first, then RL using the SFT checkpoint
        logger.info("Running SFT+RL pipeline")

        # Step 1: SFT
        logger.info("Step 1/2: Running SFT training")
        trio.run(run_sft, config, output_dir)

        # Find the latest checkpoint from SFT
        sft_checkpoint_dir = output_dir / "checkpoints"
        if not sft_checkpoint_dir.exists():
            raise RuntimeError(f"SFT checkpoint directory not found: {sft_checkpoint_dir}")

        # Get latest checkpoint directory (PyTorchTrainingBackend saves as step_NNNN/ dirs)
        # Each checkpoint dir contains: pytorch_model.bin, optimizer.bin, metadata.json
        checkpoint_dirs = sorted(sft_checkpoint_dir.glob("step_*"))
        if not checkpoint_dirs:
            raise RuntimeError(f"No checkpoints found in {sft_checkpoint_dir}")

        latest_checkpoint = str(checkpoint_dirs[-1])
        logger.info(f"Using SFT checkpoint for RL: {latest_checkpoint}")

        # Step 2: RL
        logger.info("Step 2/2: Running RL training")
        trio.run(run_rl, config, output_dir, latest_checkpoint)

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
