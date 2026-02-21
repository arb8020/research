"""FSDP worker entry point for multi-node training.

This module is launched on each trainer GPU in a multi-node setup.
It initializes torch.distributed, creates the FSDP backend, and
participates in distributed training.

Usage:
    # Launched by multi_node.py with proper env vars
    CUDA_VISIBLE_DEVICES=2 \
    MASTER_ADDR=192.168.1.10 \
    MASTER_PORT=29500 \
    WORLD_SIZE=12 \
    RANK=0 \
    LOCAL_RANK=0 \
    python -m rollouts.training.fsdp_worker \
        --config /path/to/config.json \
        --is-rank-0 1

Rank 0 responsibilities:
    - Participates in NCCL weight sync (broadcasts to inference engines)
    - Saves checkpoints to disk
    - Logs training metrics

All ranks:
    - Participate in FSDP forward/backward
    - Participate in get_weights() collective operation

Tiger Style: Explicit rank-aware behavior, clear state transitions.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def init_distributed() -> tuple[int, int, int]:
    """Initialize torch.distributed from environment variables.

    Expects env vars set by multi_node.py:
        MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK

    Returns:
        Tuple of (rank, world_size, local_rank)

    Side effects:
        - Initializes NCCL process group
        - Sets CUDA device
    """
    import torch
    import torch.distributed as dist

    # Read from environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    logger.info(
        f"Initializing distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}"
    )

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    # Set CUDA device
    torch.cuda.set_device(local_rank)

    logger.info(f"Rank {rank}: torch.distributed initialized, using GPU {local_rank}")

    return rank, world_size, local_rank


def create_fsdp_backend(
    config: Any,  # GRPOConfig
    checkpoint_dir: Path,
    rank: int,
) -> Any:  # FSDPTrainingBackend
    """Create FSDP training backend.

    Args:
        config: Training configuration
        checkpoint_dir: Directory for checkpoints
        rank: Global FSDP rank

    Returns:
        FSDPTrainingBackend instance

    Side effects:
        - Loads model
        - Wraps with FSDP
        - Creates optimizer
    """
    import torch
    from transformers import AutoModelForCausalLM

    from rollouts.training.backends.fsdp import FSDPConfig, FSDPTrainingBackend
    from rollouts.training.losses import compute_grpo_loss

    logger.info(f"Rank {rank}: Loading model {config.model.name}...")

    # Load model (will be wrapped with FSDP)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Optimizer factory (called AFTER FSDP wrapping)
    def make_optimizer(fsdp_model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            fsdp_model.parameters(),
            lr=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
        )

    # Loss function
    def loss_fn(logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        return compute_grpo_loss(
            logits=logits,
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
            advantages=batch["advantages"],
        )

    # Create FSDP backend
    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        mixed_precision=True,
        clip_grad=config.trainer.max_grad_norm,
    )

    backend = FSDPTrainingBackend(
        model=model,
        optimizer_fn=make_optimizer,
        loss_fn=loss_fn,
        checkpoint_dir=checkpoint_dir,
        config=fsdp_config,
    )

    logger.info(f"Rank {rank}: FSDP backend created")

    return backend


async def init_weight_sync_nccl(
    backend: Any,
    inference_endpoints: list[str],
    nccl_port: int,
    rank: int,
) -> Any | None:  # PipelineWeightSyncManager | None
    """Initialize NCCL weight sync group (rank 0 only).

    Only rank 0 participates in weight sync to inference engines.
    Other ranks return None.

    Args:
        backend: FSDP training backend
        inference_endpoints: List of inference engine URLs
        nccl_port: Port for weight sync NCCL group
        rank: Global FSDP rank

    Returns:
        PipelineWeightSyncManager for rank 0, None for others
    """
    if rank != 0:
        logger.info(f"Rank {rank}: Skipping weight sync init (not rank 0)")
        return None

    from rollouts.training.weight_sync import PipelineWeightSyncManager

    logger.info(f"Rank 0: Initializing NCCL weight sync with {len(inference_endpoints)} engines...")

    manager = PipelineWeightSyncManager(
        inference_endpoints=inference_endpoints,
        max_lag=2,
        nccl_master_port=nccl_port,
    )

    await manager.init_nccl_group()

    logger.info("Rank 0: NCCL weight sync initialized")

    return manager


async def create_rollout_generator(
    config: Any,
    inference_endpoints: list[str],
    rank: int,
    world_size: int,
) -> Any:
    """Create rollout generator for FSDP training.

    Rank 0 generates rollouts from inference engines and broadcasts to other ranks.
    Non-rank-0 workers receive batches via torch.distributed.

    Args:
        config: Training configuration
        inference_endpoints: List of inference engine URLs
        rank: Global FSDP rank
        world_size: Total FSDP world size

    Returns:
        Async generator yielding training batches
    """
    if rank != 0:
        # Non-rank-0: will receive batches via broadcast
        return None

    if not inference_endpoints:
        logger.warning("No inference endpoints provided, using dummy data")
        return None

    # Rank 0: Create rollout manager
    from rollouts.dtypes import Endpoint
    from rollouts.training.datasets.data_buffer import DataBuffer
    from rollouts.training.rollout_gen.async_rollout_manager import AsyncRolloutManager
    from rollouts.training.types import RolloutConfig

    # Create endpoint using first inference engine
    # TODO: Load balance across multiple engines
    endpoint = Endpoint(base_url=inference_endpoints[0])

    # Load prompts from config or use dummy
    prompts = [{"prompt": "Reverse this text: hello world"}] * config.rollout.batch_size

    data_buffer = DataBuffer(prompts=prompts)

    rollout_config = RolloutConfig(
        batch_size=config.rollout.batch_size,
        n_samples_per_prompt=config.rollout.n_samples_per_prompt,
        over_sampling_factor=1.0,
        generate_fn=None,  # Will be set up by manager
        score_fn=lambda x: 1.0,  # Dummy score
    )

    return AsyncRolloutManager(
        data_buffer=data_buffer,
        config=rollout_config,
    )


async def fsdp_train_loop(
    backend: Any,
    weight_sync: Any | None,
    rollout_generator: Any | None,
    config: Any,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    """Main FSDP training loop.

    All ranks participate in:
    - forward_backward()
    - optim_step()
    - get_weights() during checkpointing

    Only rank 0:
    - Generates rollouts from inference engines
    - Broadcasts batches to other ranks
    - Broadcasts weights to inference
    - Saves checkpoints to disk
    - Logs metrics

    Args:
        backend: FSDP training backend
        weight_sync: PipelineWeightSyncManager (rank 0 only)
        rollout_generator: AsyncRolloutManager (rank 0 only)
        config: Training configuration
        rank: Global FSDP rank
        world_size: Total FSDP world size

    Returns:
        Training metrics history
    """
    import torch
    import trio

    metrics_history = []

    async with trio.open_nursery() as nursery:
        for step in range(config.checkpoint.num_steps):
            if rank == 0:
                logger.info(f"\n--- Step {step + 1}/{config.checkpoint.num_steps} ---")

            # Get batch
            if rank == 0 and rollout_generator is not None:
                # Rank 0: Generate rollouts
                batch = await rollout_generator.generate_batch()
                # TODO: Broadcast batch tensors to other ranks
            else:
                # For now, create dummy batch for testing
                batch_size = config.rollout.batch_size * config.rollout.n_samples_per_prompt
                seq_len = 64
                batch = {
                    "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                    "labels": torch.randint(0, 1000, (batch_size, seq_len)),
                    "loss_mask": torch.ones(batch_size, seq_len),
                    "advantages": torch.randn(batch_size),
                }

            # Forward + backward (FSDP collective operation)
            fb_result = await backend.forward_backward(batch).result()

            # Optimizer step
            opt_result = await backend.optim_step().result()

            # Metrics (rank 0 only)
            if rank == 0:
                step_metrics = {**fb_result, **opt_result}
                metrics_history.append({"step": step + 1, **step_metrics})
                logger.info(f"Step {step + 1}: loss={fb_result['loss']:.4f}")

            # Weight sync (rank 0 broadcasts to inference, non-blocking)
            should_sync = (step + 1) % config.checkpoint.sync_weights_every == 0
            if should_sync and weight_sync is not None:
                await weight_sync.broadcast_weights_async(backend._fsdp_model, nursery)

            # Checkpoint (ALL ranks participate in get_weights collective)
            should_checkpoint = (step + 1) % config.checkpoint.checkpoint_every == 0
            if should_checkpoint:
                await backend.save_checkpoint(step + 1, fb_result)

    return {"metrics_history": metrics_history}


def fsdp_train_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Worker entry point launched on each trainer GPU.

    Args:
        args: Command-line arguments
            --config: Path to training config
            --is-rank-0: 1 if this is rank 0 (participates in weight sync)

    Returns:
        Training results
    """
    import json

    import trio

    # Initialize distributed
    rank, world_size, local_rank = init_distributed()

    # Load config
    with open(args.config) as f:
        config_dict = json.load(f)

    # Import here to avoid circular dependencies
    from rollouts.training.grpo import GRPOConfig

    config = GRPOConfig.from_dict(config_dict)

    # Setup output directory
    output_dir = Path(config.output.output_dir) / config.output.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create FSDP backend
    backend = create_fsdp_backend(config, output_dir, rank)

    async def _train() -> dict[str, Any]:
        # Get inference endpoints from environment (set by multi_node.py)
        inference_endpoints_raw = os.environ.get("INFERENCE_ENDPOINTS", "")
        inference_endpoints = [e for e in inference_endpoints_raw.split(",") if e]

        # Initialize weight sync (rank 0 only)
        weight_sync = None
        if args.is_rank_0 and inference_endpoints:
            weight_sync = await init_weight_sync_nccl(
                backend,
                inference_endpoints,
                config.checkpoint.nccl_master_port + 1,  # Different port from FSDP
                rank,
            )

        # Create rollout generator (rank 0 only generates, then broadcasts)
        rollout_gen = await create_rollout_generator(
            config=config,
            inference_endpoints=inference_endpoints,
            rank=rank,
            world_size=world_size,
        )

        # Training loop
        results = await fsdp_train_loop(
            backend=backend,
            weight_sync=weight_sync,
            rollout_generator=rollout_gen,
            config=config,
            rank=rank,
            world_size=world_size,
        )

        # Cleanup
        if weight_sync is not None:
            await weight_sync.cleanup()

        return results

    return trio.run(_train)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="FSDP worker for multi-node training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--is-rank-0", type=int, default=0, help="1 if rank 0 (weight sync)")

    args = parser.parse_args()

    results = fsdp_train_worker(args)

    if results.get("metrics_history"):
        final = results["metrics_history"][-1]
        logger.info(f"Training complete. Final loss: {final.get('loss', 'N/A')}")


if __name__ == "__main__":
    main()
