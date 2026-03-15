"""Megatron training worker for miniray.

This is the work function that runs on each GPU. It initializes Megatron,
creates the model, and processes training commands from the coordinator.

Usage:
    # Launched by miniray Cluster
    cluster = Cluster(nodes=[NodeConfig("node1", num_workers=8)])
    workers = cluster.start(work_fn="rollouts.training.megatron_worker.train")
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import sys
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miniray import Worker

logger = logging.getLogger(__name__)
_ARGUS_DIAG_EVENT_SENTINEL = "__ARGUS_DIAG__"


def _emit_argus_diag(event: str, **data: object) -> None:
    """Best-effort structured diagnostics for remote sandbox runs."""
    try:
        sys.stderr.write(
            f"{_ARGUS_DIAG_EVENT_SENTINEL}{json.dumps({'event': event, **data}, sort_keys=True)}\n"
        )
        sys.stderr.flush()
    except Exception:
        return


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
    VALIDATE_INFERENCE_EXPORT = 7


def _read_proc_status_snapshot() -> str:
    """Return a compact /proc/self/status snapshot for import-stage debugging."""
    wanted_keys = {"VmRSS", "VmHWM", "VmSize", "Threads"}
    snapshot: dict[str, str] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                key, _, value = line.partition(":")
                if key in wanted_keys:
                    snapshot[key] = value.strip()
    except OSError as exc:
        return f"proc_status=unavailable error={exc}"

    parts = [f"pid={os.getpid()}"]
    for key in ("VmRSS", "VmHWM", "VmSize", "Threads"):
        value = snapshot.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _log_import_stage(stage: str) -> None:
    logger.info("import_stage=%s %s", stage, _read_proc_status_snapshot())


def _resolve_train_future(future: Any) -> Any:
    """Resolve the local synchronous future shape used by Megatron workers."""
    from rollouts.training.types import ImmediateTrainFuture

    if not isinstance(future, ImmediateTrainFuture):
        raise TypeError(
            "Megatron worker expected ImmediateTrainFuture from local backend; "
            f"got {type(future).__name__}"
        )
    return future._result


def _select_native_loss_fn(config: dict[str, Any]) -> Any:
    """Select the backend-native loss function fixed for this worker."""
    from rollouts.training.backends.megatron_backend import (
        megatron_grpo_loss,
        megatron_grpo_loss_clipped,
        megatron_grpo_loss_masked,
        megatron_opd_loss,
    )

    loss_type = config.get("loss_type", "vanilla")
    if loss_type == "vanilla":
        return megatron_grpo_loss
    if loss_type == "clipped":
        return megatron_grpo_loss_clipped
    if loss_type == "masked":
        ratio_low = float(config.get("mask_ratio_low", 0.125))
        ratio_high = float(config.get("mask_ratio_high", 8.0))

        def _masked(logits: Any, batch: dict[str, Any]) -> Any:
            return megatron_grpo_loss_masked(
                logits,
                batch,
                ratio_low=ratio_low,
                ratio_high=ratio_high,
            )

        return _masked
    if loss_type == "opd":
        return megatron_opd_loss
    raise ValueError(
        f"Unknown Megatron worker loss_type={loss_type!r}. "
        "Use 'vanilla', 'clipped', 'masked', or 'opd'."
    )


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
        _log_import_stage("start")
        _log_import_stage("before_initialize_import")
        from rollouts.training.backends.megatron.initialize import (
            MegatronParallelismConfig,
            init_megatron,
        )

        _log_import_stage("after_initialize_import")
        _log_import_stage("before_model_import")
        from rollouts.training.backends.megatron.model import (
            MegatronModelConfig,
            setup_megatron_model,
        )

        _log_import_stage("after_model_import")
        _log_import_stage("before_backend_import")
        from rollouts.training.backends.megatron_backend import (
            MegatronConfig,
            MegatronTrainingBackend,
        )

        _log_import_stage("after_backend_import")
        _log_import_stage("before_lowering_import")
        from rollouts.training.lowering import (
            MegatronLowering,
            MegatronProvisioning,
            RealizationPlan,
        )

        _log_import_stage("after_lowering_import")

        logger.info("Megatron imports successful")

        lowering_payload = config["lowering"]
        lowering = MegatronLowering.from_realization(
            provisioning=MegatronProvisioning(**lowering_payload["provisioning"]),
            realization=RealizationPlan(**lowering_payload["realization"]),
        )
        provisioning = lowering.provisioning

        parallelism_config = MegatronParallelismConfig(
            tensor_parallel_size=provisioning.tp,
            pipeline_parallel_size=provisioning.pp,
            expert_parallel_size=provisioning.ep,
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
            sequence_parallel=config.get("sequence_parallel", False),
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
            loss_fn=_select_native_loss_fn(config),
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
        "validate_inference_export": Command.VALIDATE_INFERENCE_EXPORT,
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
                    inference_endpoints=msg.get(
                        "inference_endpoints",
                        config.get("inference_endpoints", []),
                    ),
                    model_name=msg.get("model_name", config.get("model_name", "")),
                    master_addr=msg.get("master_addr", config.get("master_addr")),
                    master_port=msg.get("master_port", config.get("master_port", 29500)),
                )
                handle.send({"status": "nccl_initialized"})
            else:
                _init_nccl_weight_sync(
                    backend,
                    inference_endpoints=[],
                    model_name=msg.get("model_name", config.get("model_name", "")),
                    master_addr=msg.get("master_addr", config.get("master_addr")),
                    master_port=msg.get("master_port", config.get("master_port", 29500)),
                )

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

        elif cmd_id == Command.VALIDATE_INFERENCE_EXPORT:
            details = _do_validate_inference_export(
                backend,
                model_name=config.get("model_name", ""),
            )
            if rank == 0:
                handle.send({"status": "validated", "details": details})

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

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "advantages": advantages,
        }
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
    metrics = _resolve_train_future(metrics_future)

    # Optimizer step
    step_future = backend.optim_step()
    step_metrics = _resolve_train_future(step_future)
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
    weights = _resolve_train_future(weights_future)

    if not weights:
        return  # Not rank 0, nothing to send

    # TODO: Convert to HF format and push to SGLang
    # For now, just log
    logger.info(
        "Weight sync: %d parameters to %d endpoints", len(weights), len(inference_endpoints)
    )


def _do_validate_inference_export(
    backend: Any,
    model_name: str,
) -> dict[str, Any]:
    """Validate Megatron runtime export before inference startup.

    All trainer ranks participate in the runtime export collectives. Rank 0
    returns a small summary for the control channel.
    """
    from megatron.core import mpu

    from rollouts.training.backends.megatron.inference_export import (
        build_megatron_inference_export_from_runtime,
    )

    export = build_megatron_inference_export_from_runtime(model_name, backend.model)
    details = {
        "tensor_count": len(export.tensors),
        "dropped_unconverted_keys": len(export.dropped_unconverted_keys),
        "tp_world_size": int(mpu.get_tensor_model_parallel_world_size()),
        "pp_world_size": int(mpu.get_pipeline_model_parallel_world_size()),
        "ep_world_size": int(mpu.get_expert_model_parallel_world_size()),
    }
    logger.info(
        "Validated Megatron runtime inference export: tensors=%s tp=%s pp=%s ep=%s",
        details["tensor_count"],
        details["tp_world_size"],
        details["pp_world_size"],
        details["ep_world_size"],
    )
    return details


def _init_nccl_weight_sync(
    backend: Any,
    inference_endpoints: list[str],
    model_name: str,
    master_addr: str | None = None,
    master_port: int = 29500,
) -> None:
    """Initialize NCCL sender and connect inference engines."""
    if hasattr(backend, "_nccl_weight_sender") and backend._nccl_weight_sender is not None:
        logger.info("weight_sync_megatron_reuse_existing_sender")
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
    logger.info(
        "weight_sync_megatron_init_start master=%s:%s world_size=%s endpoints=%s group=%s",
        master_addr,
        master_port,
        world_size,
        inference_endpoints,
        group_name,
    )
    _emit_argus_diag(
        "weight_sync_megatron_init_start",
        master_addr=master_addr,
        master_port=master_port,
        world_size=world_size,
        endpoints=inference_endpoints,
        group=group_name,
    )

    def register_inference_endpoint(endpoint: str, rank: int) -> None:
        import requests

        try:
            logger.info(
                "weight_sync_megatron_register_endpoint_start endpoint=%s rank=%s master=%s:%s world_size=%s group=%s",
                endpoint,
                rank,
                master_addr,
                master_port,
                world_size,
                group_name,
            )
            _emit_argus_diag(
                "weight_sync_megatron_register_endpoint_start",
                endpoint=endpoint,
                rank=rank,
                master_addr=master_addr,
                master_port=master_port,
                world_size=world_size,
                group=group_name,
            )
            response = requests.post(
                f"{endpoint}/init_weights_update_group",
                json={
                    "master_address": master_addr,
                    "master_port": master_port,
                    "rank_offset": rank,
                    "world_size": world_size,
                    "group_name": group_name,
                    "backend": "nccl",
                },
                timeout=300.0,
            )
            response.raise_for_status()
            logger.info(
                "weight_sync_megatron_register_endpoint_ok endpoint=%s rank=%s status=%s",
                endpoint,
                rank,
                response.status_code,
            )
            _emit_argus_diag(
                "weight_sync_megatron_register_endpoint_ok",
                endpoint=endpoint,
                rank=rank,
                status_code=response.status_code,
            )
        except Exception as e:
            logger.exception(
                "weight_sync_megatron_register_endpoint_failed endpoint=%s rank=%s",
                endpoint,
                rank,
            )
            _emit_argus_diag(
                "weight_sync_megatron_register_endpoint_failed",
                endpoint=endpoint,
                rank=rank,
                error=f"{type(e).__name__}: {e}",
            )
            errors.append((endpoint, str(e)))

    def trainer_join() -> None:
        import os

        os.environ.setdefault("NCCL_SHM_DISABLE", "1")
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        logger.info(
            "weight_sync_megatron_trainer_join_start master=%s:%s world_size=%s group=%s",
            master_addr,
            master_port,
            world_size,
            group_name,
        )
        _emit_argus_diag(
            "weight_sync_megatron_trainer_join_start",
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            group=group_name,
        )
        sender = WeightSyncSender(
            master_addr=master_addr,
            master_port=master_port,
            inference_world_size=len(inference_endpoints),
            group_name=group_name,
        )
        sender.init_group()
        sender_holder["sender"] = sender
        logger.info(
            "weight_sync_megatron_trainer_join_ok master=%s:%s world_size=%s group=%s",
            master_addr,
            master_port,
            world_size,
            group_name,
        )
        _emit_argus_diag(
            "weight_sync_megatron_trainer_join_ok",
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            group=group_name,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(inference_endpoints) + 1)
    ) as executor:
        futures = [executor.submit(trainer_join)]
        for i, endpoint in enumerate(inference_endpoints):
            inference_rank = i + 1
            futures.append(executor.submit(register_inference_endpoint, endpoint, inference_rank))
        for index, future in enumerate(futures):
            logger.info(
                "weight_sync_megatron_future_wait_start index=%s total=%s", index, len(futures)
            )
            _emit_argus_diag(
                "weight_sync_megatron_future_wait_start",
                index=index,
                total=len(futures),
            )
            future.result()
            logger.info(
                "weight_sync_megatron_future_wait_ok index=%s total=%s", index, len(futures)
            )
            _emit_argus_diag(
                "weight_sync_megatron_future_wait_ok",
                index=index,
                total=len(futures),
            )

    if errors:
        raise RuntimeError(f"Failed to register inference endpoints for NCCL: {errors}")

    sender = sender_holder.get("sender")
    if sender is None:
        raise RuntimeError("Failed to initialize NCCL weight sync sender")

    backend._nccl_weight_sender = sender
    backend._nccl_master_addr = master_addr
    backend._nccl_master_port = master_port
    logger.info(
        "weight_sync_megatron_init_ok master=%s:%s world_size=%s endpoints=%s group=%s",
        master_addr,
        master_port,
        world_size,
        inference_endpoints,
        group_name,
    )
    _emit_argus_diag(
        "weight_sync_megatron_init_ok",
        master_addr=master_addr,
        master_port=master_port,
        world_size=world_size,
        endpoints=inference_endpoints,
        group=group_name,
    )


def _do_sync_weights_nccl(
    backend: Any,
    model_name: str,
    inference_endpoints: list[str],
) -> None:
    """NCCL sync path for inference updates."""
    import concurrent.futures

    import requests
    import torch

    from rollouts.training.backends.megatron.inference_export import (
        build_megatron_inference_export_from_runtime,
    )

    # All trainer ranks must participate in runtime export collectives. Only
    # rank 0 publishes the resulting tensor product type to inference.
    export = build_megatron_inference_export_from_runtime(model_name, backend.model)
    state_dict = export.tensors
    if not state_dict:
        raise RuntimeError("No weights produced for NCCL sync")

    if not inference_endpoints:
        return

    sender = getattr(backend, "_nccl_weight_sender", None)
    if sender is None:
        raise RuntimeError("NCCL sender not initialized. Call init_nccl_weight_sync first.")

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
