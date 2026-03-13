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
    INIT_NCCL_WEIGHT_SYNC = 4
    SYNC_WEIGHTS_NCCL = 5
    CLEANUP_NCCL_WEIGHT_SYNC = 6


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
        from rollouts.training.lowering import MegatronLowering, ParallelIntent, RealizationPlan

        logger.info("Megatron imports successful")

        lowering_payload = config["lowering"]
        lowering = MegatronLowering.from_realization(
            parallel=ParallelIntent(**lowering_payload["parallel"]),
            realization=RealizationPlan(**lowering_payload["realization"]),
        )
        parallel = lowering.parallel

        parallelism_config = MegatronParallelismConfig(
            tensor_parallel_size=parallel.tp,
            pipeline_parallel_size=parallel.pp,
            expert_parallel_size=parallel.ep,
            sequence_parallel=config.get("sequence_parallel", False),
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
            sequence_parallel=parallelism_config.sequence_parallel,
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
            lowering=lowering,
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

        # Keep the control channel small and structured; full tracebacks already
        # go to stderr/logs and can exceed the miniray init message size.
        if rank == 0:
            try:
                handle.send({"status": "error", "error": str(e)})
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
        "init_nccl_weight_sync": Command.INIT_NCCL_WEIGHT_SYNC,
        "sync_weights_nccl": Command.SYNC_WEIGHTS_NCCL,
        "cleanup_nccl_weight_sync": Command.CLEANUP_NCCL_WEIGHT_SYNC,
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
        elif cmd_id == Command.INIT_NCCL_WEIGHT_SYNC:
            if rank == 0:
                _init_nccl_weight_sync(
                    backend,
                    inference_endpoints=config.get("inference_endpoints", []),
                    model_name=config.get("model_name", ""),
                    master_addr=config.get("master_addr"),
                    master_port=config.get("master_port", 29500),
                )
                handle.send({"status": "nccl_initialized"})
            else:
                _init_nccl_weight_sync(backend, inference_endpoints=[], model_name="")

        elif cmd_id == Command.SYNC_WEIGHTS_NCCL:
            _do_sync_weights_nccl(
                backend,
                model_name=config.get("model_name", ""),
                inference_endpoints=config.get("inference_endpoints", []) if rank == 0 else [],
            )
            if rank == 0:
                handle.send({"status": "nccl_synced"})

        elif cmd_id == Command.CLEANUP_NCCL_WEIGHT_SYNC:
            if rank == 0:
                _cleanup_nccl_weight_sync(
                    backend,
                    config.get("inference_endpoints", []),
                )
                handle.send({"status": "nccl_cleanup"})
            else:
                _cleanup_nccl_weight_sync(backend, [])

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


def _init_nccl_weight_sync(
    backend: Any,
    inference_endpoints: list[str],
    model_name: str,
    master_addr: str | None = None,
    master_port: int = 29500,
) -> None:
    """Initialize NCCL sender and connect inference engines."""
    if hasattr(backend, "_nccl_weight_sender") and backend._nccl_weight_sender is not None:
        return

    backend._nccl_weight_sender = None
    backend._nccl_model_name = model_name

    if not inference_endpoints:
        logger.info("NCCL weight sync skipped (no inference endpoints)")
        return

    import socket

    if master_addr is None:
        master_addr = "127.0.0.1"

    # Find an available port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", master_port))
        master_port = sock.getsockname()[1]

    group_name = "weight_sync"
    world_size = 1 + len(inference_endpoints)

    from rollouts.inference.weight_sync import WeightSyncSender

    sender_holder: dict[str, Any] = {}
    errors: list[tuple[str, str]] = []

    async def register_inference_endpoint(
        endpoint: str,
        rank: int,
    ) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(
                    f"{endpoint}/init_weights_update_group",
                    json={
                        "master_address": master_addr,
                        "master_port": master_port,
                        "rank_offset": rank,
                        "world_size": world_size,
                        "group_name": group_name,
                        "backend": "nccl",
                    },
                )
                response.raise_for_status()
            except Exception as e:
                errors.append((endpoint, str(e)))

    async def trainer_join() -> None:
        def _join() -> None:
            import os

            os.environ.setdefault("NCCL_SHM_DISABLE", "1")
            os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
            sender = WeightSyncSender(
                master_addr=master_addr,
                master_port=master_port,
                inference_world_size=len(inference_endpoints),
                group_name=group_name,
            )
            sender.init_group()
            sender_holder["sender"] = sender

        import trio

        await trio.to_thread.run_sync(_join)

    import trio

    async def _setup() -> None:
        async with trio.open_nursery() as nursery:
            for i, endpoint in enumerate(inference_endpoints):
                inference_rank = i + 1
                nursery.start_soon(register_inference_endpoint, endpoint, inference_rank)
            nursery.start_soon(trainer_join)

    trio.run(_setup)

    if errors:
        raise RuntimeError(f"Failed to register inference endpoints for NCCL: {errors}")

    sender = sender_holder.get("sender")
    if sender is None:
        raise RuntimeError("Failed to initialize NCCL weight sync sender")

    backend._nccl_weight_sender = sender
    backend._nccl_master_addr = master_addr
    backend._nccl_master_port = master_port
    logger.info("Initialized NCCL weight sync: world_size=%d", world_size)


def _do_sync_weights_nccl(
    backend: Any,
    model_name: str,
    inference_endpoints: list[str],
) -> None:
    """NCCL sync path for inference updates."""
    if not inference_endpoints:
        return

    sender = getattr(backend, "_nccl_weight_sender", None)
    if sender is None:
        raise RuntimeError("NCCL sender not initialized. Call init_nccl_weight_sync first.")

    import concurrent.futures

    import requests
    import torch

    # Gather weights (rank 0 only has full state).
    weights_future = backend.get_weights()
    weights = weights_future.result()
    if not weights:
        return

    state_dict = _convert_megatron_state_dict(model_name, weights)
    if not state_dict:
        raise RuntimeError("No weights produced for NCCL sync")

    # Inform inference engines and broadcast in the same order.
    param_info = [
        {
            "name": name,
            "shape": list(p.shape),
            "dtype": str(p.dtype).replace("torch.", ""),
        }
        for name, p in state_dict.items()
    ]

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(inference_endpoints)))
    futures = []
    for endpoint in inference_endpoints:
        futures.append(
            executor.submit(
                requests.post,
                f"{endpoint}/update_weights_from_distributed",
                json={
                    "names": [item["name"] for item in param_info],
                    "shapes": [item["shape"] for item in param_info],
                    "dtypes": [item["dtype"] for item in param_info],
                    "group_name": "weight_sync",
                    "weight_version": str(sender.weight_version + 1),
                },
                timeout=300.0,
            )
        )

    handles = sender.broadcast_weights(state_dict, async_op=True)

    for handle in handles:
        handle.wait()

    for future in futures:
        response = future.result()
        response.raise_for_status()

    executor.shutdown(wait=False)
    torch.cuda.empty_cache()


def _cleanup_nccl_weight_sync(
    backend: Any,
    inference_endpoints: list[str],
) -> None:
    """Best-effort NCCL cleanup for Megatron worker."""
    sender = getattr(backend, "_nccl_weight_sender", None)
    if sender is not None:
        try:
            sender.cleanup()
        except Exception:
            pass

    if inference_endpoints:
        import concurrent.futures

        import requests

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(inference_endpoints))
        )
        futures = []
        for endpoint in inference_endpoints:
            futures.append(
                executor.submit(
                    requests.post,
                    f"{endpoint}/destroy_weights_update_group",
                    json={"group_name": "weight_sync"},
                    timeout=10.0,
                )
            )
        for future in futures:
            try:
                future.result()
            except Exception:
                pass
        executor.shutdown(wait=False)

    backend._nccl_weight_sender = None


def _convert_megatron_state_dict(model_name: str, state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert Megatron shard names to HuggingFace names (best effort)."""
    import logging

    from transformers import AutoConfig

    from rollouts.training.backends.megatron.weight_conversion import (
        convert_megatron_to_hf,
        remove_padding,
    )

    logger = logging.getLogger(__name__)

    if not model_name:
        return state_dict

    try:
        hf_config = AutoConfig.from_pretrained(model_name)
        num_layers = getattr(hf_config, "num_hidden_layers", 0) or getattr(hf_config, "n_layers", 0)
        if not num_layers:
            raise ValueError("Unable to infer num_layers")
        vocab_size = int(getattr(hf_config, "vocab_size", 0))
        num_attention_heads = int(getattr(hf_config, "num_attention_heads", 0))
        hidden_size = int(getattr(hf_config, "hidden_size", 0))
        num_query_groups = getattr(hf_config, "num_query_groups", num_attention_heads)
        kv_channels = getattr(hf_config, "kv_channels", None)
        q_lora_rank = getattr(hf_config, "q_lora_rank", None)
    except Exception as exc:
        logger.warning("Failed to load HF config for Megatron->HF conversion: %s", exc)
        return {_strip_chunk_prefix(name): value for name, value in state_dict.items()}

    output: dict[str, Any] = {}
    conversion_attempted = False

    for name, param in state_dict.items():
        clean_name = _strip_chunk_prefix(name)
        try:
            for hf_name, hf_param in convert_megatron_to_hf(
                model_name=model_name,
                name=clean_name,
                param=param,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                num_query_groups=num_query_groups,
                kv_channels=kv_channels,
                q_lora_rank=q_lora_rank,
            ):
                output[_strip_chunk_prefix(hf_name)] = remove_padding(hf_name, hf_param, vocab_size)
            conversion_attempted = True
        except Exception:
            # Keep raw names if conversion fails for this param.
            output[clean_name] = param

    if not conversion_attempted:
        logger.warning("Megatron->HF conversion not applied for any tensors; using raw keys.")

    return output


def _strip_chunk_prefix(name: str) -> str:
    """Drop `chunk_<N>.` prefix from chunked Megatron parameter names."""
    prefix, separator, remainder = name.partition(".")
    if prefix.startswith("chunk_") and separator and prefix[6:].isdigit():
        return remainder
    return name
