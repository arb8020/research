"""Megatron training worker for miniray.

This is the work function that runs on each GPU. It initializes Megatron,
creates the model, and processes training commands from the coordinator.

Usage:
    # Launched by miniray Cluster
    cluster = Cluster(nodes=[NodeConfig("node1", num_workers=8)])
    workers = cluster.start(work_fn="rollouts.training.megatron_worker.train")
"""

from __future__ import annotations

import logging
import os
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miniray import Worker

logger = logging.getLogger(__name__)


class Command(IntEnum):
    """Command IDs for rank synchronization.

    Using explicit IDs instead of hash() for deterministic behavior.
    """

    SHUTDOWN = 0
    TRAIN_STEP = 1
    SYNC_WEIGHTS = 2
    SAVE_CHECKPOINT = 3


def train(handle: Worker) -> None:
    """Miniray work function for Megatron distributed training.

    Protocol:
    1. Receive "init" message with rank, world_size, config
    2. Initialize Megatron process groups
    3. Create model/optimizer
    4. Loop: receive commands, execute, respond

    Commands:
    - {"cmd": "init", "rank": int, "world_size": int, "config": dict}
    - {"cmd": "train_step", "batch": dict}
    - {"cmd": "sync_weights"}
    - {"cmd": "save_checkpoint", "path": str}
    - {"cmd": "shutdown"}

    Only rank 0 sends responses back to coordinator.
    """
    import sys
    import traceback

    # Phase 1: Wait for init message
    init_msg = handle.recv(max_size=10 * 1024 * 1024)  # 10MB for config
    assert init_msg["cmd"] == "init", f"Expected init, got {init_msg['cmd']}"

    rank = init_msg["rank"]
    world_size = init_msg["world_size"]
    config = init_msg["config"]

    # Set up logging with rank
    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank {rank}] %(levelname)s %(name)s: %(message)s",
        force=True,  # Override any existing config
    )
    logger.info("Worker starting: rank=%d/%d", rank, world_size)

    # Set CUDA device BEFORE importing torch
    local_rank = rank % 8  # Assume max 8 GPUs per node
    cuda_device_ids = config.get("cuda_device_ids")
    if cuda_device_ids and rank < len(cuda_device_ids):
        cuda_device = cuda_device_ids[rank]
    else:
        # Default: use local_rank directly (rank 0 -> GPU 0, rank 1 -> GPU 1, etc.)
        cuda_device = local_rank
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    logger.info("Worker rank %d using CUDA_VISIBLE_DEVICES=%s", rank, cuda_device)

    try:
        # Phase 2: Initialize Megatron
        logger.info("Importing Megatron modules...")
        from rollouts.training.backends.megatron import init_megatron, setup_megatron_model
        from rollouts.training.backends.megatron.initialize import MegatronParallelismConfig
        from rollouts.training.backends.megatron.model import MegatronModelConfig
        from rollouts.training.backends.megatron_backend import (
            MegatronConfig,
            MegatronTrainingBackend,
        )

        logger.info("Megatron imports successful")

        parallelism_config = MegatronParallelismConfig(
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            pipeline_parallel_size=config.get("pipeline_parallel_size", 1),
            expert_parallel_size=config.get("expert_parallel_size", 1),
        )

        logger.info("Calling init_megatron...")
        init_megatron(
            rank=rank,
            world_size=world_size,
            config=parallelism_config,
            master_addr=config.get("master_addr"),
            master_port=config.get("master_port"),
        )
        logger.info("init_megatron complete")

        # Phase 3: Create model
        logger.info("Creating model config...")
        model_config = MegatronModelConfig(
            model_name=config["model_name"],
            lr=config.get("lr", 1e-6),
            bf16=config.get("bf16", True),
            micro_batch_size=config.get("micro_batch_size", 1),
            global_batch_size=config.get("global_batch_size", 8),
            seq_length=config.get("seq_length", 4096),
        )

        logger.info("Setting up Megatron model...")
        model, optimizer, scheduler = setup_megatron_model(model_config)
        logger.info("Model setup complete")

        # Create backend
        backend_config = MegatronConfig(
            tensor_model_parallel_size=parallelism_config.tensor_parallel_size,
            pipeline_model_parallel_size=parallelism_config.pipeline_parallel_size,
            expert_model_parallel_size=parallelism_config.expert_parallel_size,
            micro_batch_size=model_config.micro_batch_size,
            global_batch_size=model_config.global_batch_size,
            seq_length=model_config.seq_length,
            clip_grad=config.get("clip_grad", 1.0),
            bf16=model_config.bf16,
        )

        backend = MegatronTrainingBackend(
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            config=backend_config,
        )

        logger.info("Model initialized, entering training loop")

        # Rank 0 confirms init complete
        if rank == 0:
            handle.send({"status": "initialized"})

        # Phase 4: Training loop
        _training_loop(handle, backend, rank, config)

    except Exception as e:
        # Capture full traceback and send to coordinator
        tb = traceback.format_exc()
        error_msg = f"Worker rank {rank} failed: {e}\n{tb}"
        logger.exception(error_msg)
        print(error_msg, file=sys.stderr, flush=True)

        # Try to send error to coordinator (rank 0 only)
        if rank == 0:
            try:
                handle.send({"status": "error", "error": str(e), "traceback": tb})
            except Exception:
                pass  # Socket might be closed

        raise  # Re-raise to trigger worker exit


def _training_loop(
    handle: Worker,
    backend: Any,
    rank: int,
    config: dict[str, Any],
) -> None:
    """Main training loop - receive commands, execute, respond.

    Args:
        handle: Miniray worker handle for IPC
        backend: MegatronTrainingBackend instance
        rank: This worker's rank
        config: Training config dict
    """
    import torch
    import torch.distributed as dist

    # Map command strings to enum values
    CMD_MAP = {
        "shutdown": Command.SHUTDOWN,
        "train_step": Command.TRAIN_STEP,
        "sync_weights": Command.SYNC_WEIGHTS,
        "save_checkpoint": Command.SAVE_CHECKPOINT,
    }

    while True:
        # All ranks wait for command from coordinator
        # Rank 0 receives directly, others wait for broadcast
        if rank == 0:
            msg = handle.recv(max_size=100 * 1024 * 1024)  # 100MB for batches
            cmd_str = msg["cmd"]
            cmd_id = CMD_MAP.get(cmd_str, -1)
            if cmd_id == -1:
                error_msg = f"Unknown command: {cmd_str}"
                logger.error(error_msg)
                handle.send({"status": "error", "error": error_msg})
                raise ValueError(error_msg)

            # Broadcast command ID to other ranks
            cmd_tensor = torch.tensor([cmd_id], dtype=torch.long, device="cuda")
            dist.broadcast(cmd_tensor, src=0)
        else:
            # Receive broadcast command ID
            cmd_tensor = torch.zeros(1, dtype=torch.long, device="cuda")
            dist.broadcast(cmd_tensor, src=0)
            cmd_id = int(cmd_tensor.item())
            msg = {}

            if cmd_id == -1:
                raise ValueError(f"Unknown command ID: {cmd_id}")

        # Handle shutdown
        if cmd_id == Command.SHUTDOWN:
            if rank == 0:
                logger.info("Shutdown requested")
            break

        # Handle train_step
        if cmd_id == Command.TRAIN_STEP:
            batch = msg.get("batch") if rank == 0 else None
            metrics = _do_train_step(backend, batch, rank)
            if rank == 0:
                handle.send({"status": "ok", "metrics": metrics})

        # Handle sync_weights
        elif cmd_id == Command.SYNC_WEIGHTS:
            _do_sync_weights(backend, config.get("inference_endpoints", []) if rank == 0 else [])
            if rank == 0:
                handle.send({"status": "synced"})

        # Handle save_checkpoint
        elif cmd_id == Command.SAVE_CHECKPOINT:
            if rank == 0:
                path = msg.get("path", "./checkpoints")
                step = msg.get("step", 0)
                backend.save_checkpoint(step)
                handle.send({"status": "saved", "path": path})

    logger.info("Worker exiting")


def _do_train_step(
    backend: Any,
    batch: dict[str, Any] | None,
    rank: int,
) -> dict[str, float]:
    """Execute one training step.

    Rank 0 has the batch, broadcasts to other ranks via NCCL.
    All ranks call forward_backward together.

    Args:
        backend: MegatronTrainingBackend
        batch: Training batch (only rank 0 has this, as Python lists from JSON)
        rank: This worker's rank

    Returns:
        Training metrics (only meaningful on rank 0)
    """
    import torch
    import torch.distributed as dist

    # Broadcast batch from rank 0 to all ranks
    if rank == 0:
        assert batch is not None, "Rank 0 must have batch"

        # Convert lists (from JSON) back to tensors
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device="cuda")
        labels = torch.tensor(batch["labels"], dtype=torch.long, device="cuda")
        loss_mask = (
            torch.tensor(batch["loss_mask"], device="cuda")
            if batch.get("loss_mask") is not None
            else None
        )
        advantages = (
            torch.tensor(batch["advantages"], device="cuda")
            if batch.get("advantages") is not None
            else None
        )

        # Broadcast shapes first
        shapes = torch.tensor(
            [
                input_ids.shape[0],
                input_ids.shape[1],
                1 if loss_mask is not None else 0,
                1 if advantages is not None else 0,
            ],
            dtype=torch.long,
            device="cuda",
        )
        dist.broadcast(shapes, src=0)

        # Broadcast tensors
        dist.broadcast(input_ids.cuda(), src=0)
        dist.broadcast(labels.cuda(), src=0)
        if loss_mask is not None:
            dist.broadcast(loss_mask.cuda(), src=0)
        if advantages is not None:
            dist.broadcast(advantages.cuda(), src=0)
    else:
        # Receive shapes
        shapes = torch.zeros(4, dtype=torch.long, device="cuda")
        dist.broadcast(shapes, src=0)
        batch_size, seq_len, has_loss_mask, has_advantages = shapes.tolist()

        # Receive tensors
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device="cuda")
        labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device="cuda")
        dist.broadcast(input_ids, src=0)
        dist.broadcast(labels, src=0)

        loss_mask = None
        if has_loss_mask:
            loss_mask = torch.zeros(batch_size, seq_len, device="cuda")
            dist.broadcast(loss_mask, src=0)

        advantages = None
        if has_advantages:
            advantages = torch.zeros(batch_size, device="cuda")
            dist.broadcast(advantages, src=0)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "advantages": advantages,
        }

    # All ranks call forward_backward
    metrics_future = backend.forward_backward(batch)
    metrics = metrics_future.result()

    # Optimizer step
    step_future = backend.optim_step()
    step_metrics = step_future.result()
    metrics.update(step_metrics)

    return metrics


def _do_sync_weights(backend: Any, inference_endpoints: list[str]) -> None:
    """Sync weights to inference engines.

    Gathers weights from all TP/PP ranks and pushes to SGLang.

    Args:
        backend: MegatronTrainingBackend
        inference_endpoints: List of SGLang endpoint URLs
    """
    if not inference_endpoints:
        return

    # Get weights (handles TP/PP gathering internally)
    weights_future = backend.get_weights()
    weights = weights_future.result()

    if not weights:
        return  # Not rank 0, nothing to send

    # TODO: Convert to HF format and push to SGLang
    # For now, just log
    logger.info(
        "Weight sync: %d parameters to %d endpoints", len(weights), len(inference_endpoints)
    )
