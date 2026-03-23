"""Remote Megatron backend using miniray for multi-process orchestration.

This wraps miniray workers running megatron_worker.py and provides the same
interface as other TrainingBackend implementations. The coordinator (grpo.py)
talks to this backend, which forwards commands to distributed workers.

Architecture:
    grpo.py (coordinator)
        |
        v
    MegatronRemoteBackend
        |
        v (miniray send/recv)
    megatron_worker.py (rank 0)
        |
        v (NCCL broadcast)
    megatron_worker.py (ranks 1..N)
"""

from __future__ import annotations

import json
import logging
import os
import select
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...configs import MegatronOverrides
from ...contracts import TrainingDatum
from ...lowering import MegatronLowering
from ...types import ImmediateTrainFuture, TrainFuture

if TYPE_CHECKING:
    from miniray import Worker

logger = logging.getLogger(__name__)

_CONTROL_MESSAGE_MAX_BYTES = 64 * 1024
_WITNESS_RESPONSE_TIMEOUT_SEC = 20.0
_RESPONSE_POLL_INTERVAL_SEC = 0.25
_RESPONSE_PROGRESS_LOG_INTERVAL_SEC = 5.0
_INITIALIZE_TIMEOUT_SEC = 600.0
_PREFLIGHT_TRAIN_STEP_TIMEOUT_SEC = 600.0


@dataclass
class MegatronRemoteConfig:
    """Configuration for remote Megatron backend.

    `lowering` is the semantic source for partition intent. This remote config
    still executes Megatron's native runtime semantics; it does not ship an
    executable collective program to the workers.
    """

    # Model
    model_name: str
    dtype: str = "bfloat16"
    checkpoint_path: str | None = None

    # Lowered Megatron provisioning derived from RealizationPlan
    lowering: MegatronLowering = field(default_factory=MegatronLowering)

    # Backend-native Megatron runtime settings not modeled in RealizationPlan
    sequence_parallel: bool = False
    megatron_overrides: MegatronOverrides | None = None

    # Training
    lr: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    loss_type: str = "vanilla"
    mask_ratio_low: float = 0.125
    mask_ratio_high: float = 8.0
    micro_batch_size: int = 1
    global_batch_size: int = 8
    num_microbatches: int = 1
    seq_length: int = 4096
    save_optimizer_state: bool = True

    # NCCL
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    # Inference endpoints for weight sync
    inference_endpoints: list[str] = field(default_factory=list)

    # GPU assignment (which physical GPUs to use for each rank)
    # If None, workers use rank as GPU index (rank 0 -> GPU 0, etc.)
    cuda_device_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        overrides = self.megatron_overrides
        if overrides is None:
            return
        if overrides.output_materialization != "default":
            raise ValueError(
                "Megatron override output_materialization is not lowered yet. "
                "Only 'default' is supported today."
            )
        if overrides.lm_head_token_chunk_size is not None:
            raise ValueError("Megatron override lm_head_token_chunk_size is not lowered yet.")
        if overrides.max_tokens_per_microbatch is not None:
            raise ValueError("Megatron override max_tokens_per_microbatch is not lowered yet.")


@dataclass
class MegatronRemoteBackend:
    """Training backend that coordinates Megatron workers via miniray.

    This provides the same interface as other TrainingBackend implementations
    but forwards all operations to distributed workers.

    Example:
        >>> workers = spawn_megatron_workers(num_gpus=4, config=config)
        >>> backend = MegatronRemoteBackend(workers=workers, config=config)
        >>> metrics = backend.forward_backward(batch)
        >>> backend.optim_step()
    """

    workers: list[Worker]
    config: MegatronRemoteConfig
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    phase_callback: Callable[..., None] | None = None
    _nccl_inference_endpoints: list[str] = field(default_factory=list, init=False)
    _nccl_initialized: bool = field(default=False, init=False)
    weight_version: int = 0

    _step: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)

    def _restore_local_step_from_checkpoint(self) -> None:
        checkpoint_path = self.config.checkpoint_path
        if not checkpoint_path:
            return

        tracker_path = Path(checkpoint_path) / "latest_checkpointed_iteration.txt"
        if not tracker_path.exists():
            return

        try:
            self._step = int(tracker_path.read_text(encoding="utf-8").strip())
        except Exception as exc:
            logger.warning(
                "Failed to restore Megatron checkpoint iteration from %s: %s: %s",
                tracker_path,
                type(exc).__name__,
                exc,
            )

    def _worker_snapshot(self) -> list[dict[str, Any]]:
        snapshot: list[dict[str, Any]] = []
        for worker in self.workers:
            pid = getattr(worker, "pid", None)
            try:
                alive = bool(worker.is_alive())
            except Exception as exc:
                snapshot.append({
                    "pid": pid,
                    "alive": False,
                    "state_error": f"{type(exc).__name__}: {exc}",
                })
                continue
            snapshot.append({"pid": pid, "alive": alive})
        return snapshot

    def _emit_phase(self, event: str, **data: Any) -> None:
        if callable(self.phase_callback):
            self.phase_callback(event, **data)

    def _abort_workers(self, *, reason: str, context: str) -> None:
        before = self._worker_snapshot()
        logger.warning(
            "megatron_remote_abort_start",
            extra={
                "event": "megatron_remote_abort_start",
                "context": context,
                "reason": reason,
                "workers_before": before,
            },
        )
        try:
            self.shutdown()
        except Exception as exc:
            logger.warning(
                "megatron_remote_abort_shutdown_failed",
                extra={
                    "event": "megatron_remote_abort_shutdown_failed",
                    "context": context,
                    "reason": reason,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for worker in self.workers:
                pid = getattr(worker, "pid", None)
                if pid is None:
                    continue
                try:
                    if worker.is_alive():
                        os.kill(pid, sig)
                except Exception:
                    pass
            if sig == signal.SIGTERM:
                time.sleep(0.5)
        for worker in self.workers:
            try:
                worker.close()
            except Exception:
                pass
        after = self._worker_snapshot()
        logger.warning(
            "megatron_remote_abort_complete",
            extra={
                "event": "megatron_remote_abort_complete",
                "context": context,
                "reason": reason,
                "workers_before": before,
                "workers_after": after,
            },
        )

    def _recv_response(self, worker: Worker, *, context: str, max_size: int) -> dict[str, Any]:
        response = worker.recv(max_size=max_size)
        if response.get("status") == "error":
            error = response.get("error")
            traceback_tail = response.get("traceback_tail")
            if traceback_tail:
                raise RuntimeError(
                    f"Megatron worker failed during {context}: {error}\n{traceback_tail}"
                )
            raise RuntimeError(f"Megatron worker failed during {context}: {error}")
        return response

    def _recv_response_polling(
        self,
        worker: Worker,
        *,
        context: str,
        max_size: int,
        timeout_sec: float,
    ) -> dict[str, Any]:
        start = time.monotonic()
        next_progress_log = start + _RESPONSE_PROGRESS_LOG_INTERVAL_SEC
        worker_snapshot = self._worker_snapshot()
        self._emit_phase(
            "megatron_remote_response_wait_start",
            context=context,
            timeout_sec=timeout_sec,
            workers=worker_snapshot,
        )
        logger.info(
            "megatron_remote_response_wait_start",
            extra={
                "event": "megatron_remote_response_wait_start",
                "context": context,
                "timeout_sec": timeout_sec,
                "workers": worker_snapshot,
            },
        )
        while True:
            ready, _, _ = select.select([worker], [], [], _RESPONSE_POLL_INTERVAL_SEC)
            if ready:
                elapsed = time.monotonic() - start
                self._emit_phase(
                    "megatron_remote_response_wait_ready",
                    context=context,
                    elapsed_sec=round(elapsed, 3),
                )
                logger.info(
                    "megatron_remote_response_wait_ready",
                    extra={
                        "event": "megatron_remote_response_wait_ready",
                        "context": context,
                        "elapsed_sec": round(elapsed, 3),
                    },
                )
                return self._recv_response(worker, context=context, max_size=max_size)

            now = time.monotonic()
            elapsed = now - start
            if now >= next_progress_log:
                worker_snapshot = self._worker_snapshot()
                self._emit_phase(
                    "megatron_remote_response_wait_progress",
                    context=context,
                    elapsed_sec=round(elapsed, 3),
                    timeout_sec=timeout_sec,
                    workers=worker_snapshot,
                )
                logger.warning(
                    "megatron_remote_response_wait_progress",
                    extra={
                        "event": "megatron_remote_response_wait_progress",
                        "context": context,
                        "elapsed_sec": round(elapsed, 3),
                        "timeout_sec": timeout_sec,
                        "workers": worker_snapshot,
                    },
                )
                next_progress_log = now + _RESPONSE_PROGRESS_LOG_INTERVAL_SEC

            workers = self._worker_snapshot()
            dead_workers = [row["pid"] for row in workers if not row.get("alive", False)]
            if dead_workers:
                self._abort_workers(reason="peer_worker_exited_while_waiting", context=context)
                raise EOFError(
                    f"Megatron worker peers {dead_workers} exited while waiting for {context}"
                )

            if not worker.is_alive():
                self._abort_workers(reason="worker_exited_while_waiting", context=context)
                raise EOFError(f"Megatron worker {worker.pid} exited while waiting for {context}")

            if elapsed >= timeout_sec:
                self._abort_workers(reason="response_timeout", context=context)
                raise TimeoutError(
                    f"Megatron worker response timed out during {context} after {elapsed:.3f}s"
                )

    def initialize(self) -> None:
        """Initialize all workers with config.

        Must be called before any training operations.
        """
        if self._initialized:
            return

        def _log_init_event(event: str, **data: Any) -> None:
            self._emit_phase(event, **data)
            logger.info(
                event,
                extra={
                    "event": event,
                    "num_workers": len(self.workers),
                    **data,
                },
            )

        _log_init_event("megatron_remote_initialize_start")

        # Send init command to all workers
        _log_init_event("megatron_remote_initialize_send_start")
        for rank, worker in enumerate(self.workers):
            _log_init_event(
                "megatron_remote_initialize_send_rank",
                rank=rank,
                worker_pid=getattr(worker, "pid", None),
            )
            worker.send({
                "cmd": "init",
                "rank": rank,
                "world_size": len(self.workers),
                "config": {
                    "model_name": self.config.model_name,
                    "checkpoint_path": self.config.checkpoint_path,
                    "checkpoint_dir": str(self.checkpoint_dir),
                    "lowering": {
                        "provisioning": {
                            "tp": self.config.lowering.provisioning.tp,
                            "pp": self.config.lowering.provisioning.pp,
                            "ep": self.config.lowering.provisioning.ep,
                            "enable_loss_parallel": self.config.lowering.provisioning.enable_loss_parallel,
                            "packed_sequences": self.config.lowering.provisioning.packed_sequences,
                        },
                        "realization": {
                            "local_layouts": self.config.lowering.realization.local_layouts,
                            "collective_transitions": self.config.lowering.realization.collective_transitions,
                            "packed_sequences": self.config.lowering.realization.packed_sequences,
                        },
                    },
                    "sequence_parallel": self.config.sequence_parallel,
                    "megatron_overrides": (
                        None
                        if self.config.megatron_overrides is None
                        else {
                            "sequence_parallel": self.config.megatron_overrides.sequence_parallel,
                            "allocator_expandable_segments": (
                                self.config.megatron_overrides.allocator_expandable_segments
                            ),
                            "output_materialization": (
                                self.config.megatron_overrides.output_materialization
                            ),
                            "lm_head_token_chunk_size": (
                                self.config.megatron_overrides.lm_head_token_chunk_size
                            ),
                            "max_tokens_per_microbatch": (
                                self.config.megatron_overrides.max_tokens_per_microbatch
                            ),
                        }
                    ),
                    "lr": self.config.lr,
                    "bf16": self.config.dtype == "bfloat16",
                    "loss_type": self.config.loss_type,
                    "mask_ratio_low": self.config.mask_ratio_low,
                    "mask_ratio_high": self.config.mask_ratio_high,
                    "micro_batch_size": self.config.micro_batch_size,
                    "global_batch_size": self.config.global_batch_size,
                    "num_microbatches": self.config.num_microbatches,
                    "seq_length": self.config.seq_length,
                    "save_optimizer_state": self.config.save_optimizer_state,
                    "clip_grad": self.config.max_grad_norm,
                    "master_addr": self.config.master_addr,
                    "master_port": self.config.master_port,
                    "inference_endpoints": self.config.inference_endpoints,
                    "cuda_device_ids": self.config.cuda_device_ids,
                },
            })
        _log_init_event("megatron_remote_initialize_send_complete")

        # Wait for rank 0 to confirm initialization
        _log_init_event("megatron_remote_initialize_wait_start")
        response = self._recv_response_polling(
            self.workers[0],
            context="initialize",
            max_size=_CONTROL_MESSAGE_MAX_BYTES,
            timeout_sec=_INITIALIZE_TIMEOUT_SEC,
        )
        _log_init_event("megatron_remote_initialize_wait_ready", status=response.get("status"))
        assert response["status"] == "initialized", f"Init failed: {response}"

        initialized_step = response.get("step")
        if initialized_step is not None:
            self._step = int(initialized_step)
        else:
            self._restore_local_step_from_checkpoint()
        _log_init_event("megatron_remote_initialize_step_restored", step=self._step)
        self._initialized = True
        _log_init_event("megatron_remote_initialize_ok", step=self._step)

    def forward_backward(
        self,
        batch: TrainingDatum | dict[str, Any],
        *,
        loss_fn: Any | None = None,
        loss_fn_config: dict[str, float] | None = None,
    ) -> TrainFuture[dict[str, float]]:
        """Compute loss and gradients on batch.

        Args:
            batch: Training batch with input_ids, labels, etc.

        Returns:
            Future resolving to metrics dict with loss, etc.
        """
        assert self._initialized, "Call initialize() first"
        if loss_fn is not None or loss_fn_config is not None:
            raise ValueError(
                "MegatronRemoteBackend.forward_backward does not support per-call loss overrides yet. "
                "Megatron loss selection is fixed at worker initialization from trainer.loss_type."
            )

        # Send batch to rank 0 (it broadcasts to other ranks)
        self.workers[0].send({
            "cmd": "train_step",
            "batch": self._serialize_batch(self._normalize_training_batch(batch)),
        })

        # Wait for metrics from rank 0
        response = self._recv_response(
            self.workers[0],
            context="train_step",
            max_size=10 * 1024 * 1024,
        )
        assert response["status"] == "ok", f"Train step failed: {response}"

        self._step += 1
        return ImmediateTrainFuture(response["metrics"], operation="forward_backward")

    def preflight_step(
        self, batch: TrainingDatum | dict[str, Any]
    ) -> TrainFuture[dict[str, float]]:
        """Run one backend-native synthetic step for health checking.

        Megatron executes the optimizer step inside `forward_backward`, so this
        surface exists to give GRPO an honest preflight hook without pretending
        the remote backend supports the contract-native per-call loss API yet.
        """
        assert self._initialized, "Call initialize() first"
        serialized_batch = self._serialize_batch(self._normalize_training_batch(batch))
        self.workers[0].send({
            "cmd": "train_step",
            "batch": serialized_batch,
        })
        response = self._recv_response_polling(
            self.workers[0],
            context="train_step_preflight",
            max_size=10 * 1024 * 1024,
            timeout_sec=_PREFLIGHT_TRAIN_STEP_TIMEOUT_SEC,
        )
        assert response["status"] == "ok", f"Preflight train step failed: {response}"
        self._step += 1
        return ImmediateTrainFuture(response["metrics"], operation="preflight_step")

    def validate_inference_export(self) -> TrainFuture[dict[str, Any]]:
        """Validate the Megatron runtime -> inference export boundary.

        This is an explicit preflight for the backend-local export/update slice.
        It should fail before inference startup if the current Megatron runtime
        state cannot be lowered into the HF/SGLang tensor contract honestly.
        """
        assert self._initialized, "Call initialize() first"
        self.workers[0].send({"cmd": "validate_inference_export"})
        response = self._recv_response(
            self.workers[0],
            context="validate_inference_export",
            max_size=_CONTROL_MESSAGE_MAX_BYTES,
        )
        assert response["status"] == "validated", f"Inference export validation failed: {response}"
        return ImmediateTrainFuture(response["details"], operation="validate_inference_export")

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        """Apply gradients (already done in forward_backward for Megatron)."""
        # Megatron does optimizer step inside forward_backward
        return ImmediateTrainFuture({"step": self._step}, operation="optim_step")

    def sync_weights(self) -> None:
        """Sync weights to inference engines."""
        assert self._initialized, "Call initialize() first"

        self.workers[0].send({"cmd": "sync_weights"})
        response = self._recv_response(
            self.workers[0],
            context="sync_weights",
            max_size=_CONTROL_MESSAGE_MAX_BYTES,
        )
        assert response["status"] == "synced", f"Weight sync failed: {response}"
        self.weight_version += 1

    def save_checkpoint(self, step: int, metrics: dict[str, Any]) -> TrainFuture[Path]:
        """Save checkpoint."""
        assert self._initialized, "Call initialize() first"

        self.workers[0].send({
            "cmd": "save_checkpoint",
            "step": step,
            "path": str(self.checkpoint_dir),
        })
        response = self._recv_response(
            self.workers[0],
            context="save_checkpoint",
            max_size=_CONTROL_MESSAGE_MAX_BYTES,
        )
        assert response["status"] == "saved", f"Checkpoint save failed: {response}"
        return ImmediateTrainFuture(Path(response["path"]), operation="save_checkpoint")

    def save_weights_for_sampler(self, path: Path) -> TrainFuture[None]:
        """Sync weights for sampler inference update.

        Remote Megatron workers sync weights directly through the existing
        worker sync command, so this is a no-op beyond triggering sync_weights.
        """
        del path
        self.sync_weights()
        return ImmediateTrainFuture(None, operation="save_weights_for_sampler")

    async def init_nccl_weight_sync(
        self,
        inference_endpoints: list[str],
        master_addr: str | None = None,
        master_port: int = 29500,
    ) -> None:
        """Initialize NCCL weight sync inside rank-0 worker."""
        self.workers[0].send({
            "cmd": "init_nccl_weight_sync",
            "inference_endpoints": inference_endpoints,
            "master_addr": master_addr,
            "master_port": master_port,
        })
        response = self._recv_response(
            self.workers[0],
            context="init_nccl_weight_sync",
            max_size=_CONTROL_MESSAGE_MAX_BYTES,
        )
        assert response["status"] == "nccl_initialized", f"NCCL init failed: {response}"
        self._nccl_inference_endpoints = list(inference_endpoints)
        self._nccl_initialized = True
        return None

    async def sync_weights_nccl(self) -> None:
        """NCCL sync to inference engines."""
        assert self._initialized, "Call initialize() first"
        self.workers[0].send({"cmd": "sync_weights_nccl"})
        response = self._recv_response(
            self.workers[0],
            context="sync_weights_nccl",
            max_size=_CONTROL_MESSAGE_MAX_BYTES,
        )
        assert response["status"] == "nccl_synced", f"NCCL weight sync failed: {response}"
        self.weight_version += 1

    async def sync_weights_nccl_witness(self, tensor_limit: int = 1) -> None:
        """Run a tiny NCCL witness update without advancing weight version."""
        assert self._initialized, "Call initialize() first"
        if tensor_limit <= 0:
            raise ValueError(f"tensor_limit must be positive, got {tensor_limit}")
        self.workers[0].send({
            "cmd": "sync_weights_nccl",
            "tensor_limit": tensor_limit,
            "witness": True,
        })
        response = self._recv_response_polling(
            self.workers[0],
            context="sync_weights_nccl_witness",
            max_size=_CONTROL_MESSAGE_MAX_BYTES,
            timeout_sec=_WITNESS_RESPONSE_TIMEOUT_SEC,
        )
        assert response["status"] == "nccl_synced", f"NCCL witness sync failed: {response}"

    async def cleanup_nccl_weight_sync(self) -> None:
        """Best effort cleanup for remote NCCL resources."""
        if not self._nccl_initialized:
            return
        try:
            self.workers[0].send({
                "cmd": "cleanup_nccl_weight_sync",
                "inference_endpoints": self._nccl_inference_endpoints,
            })
            _ = self._recv_response(
                self.workers[0],
                context="cleanup_nccl_weight_sync",
                max_size=_CONTROL_MESSAGE_MAX_BYTES,
            )
        except Exception:
            pass
        self._nccl_initialized = False
        self._nccl_inference_endpoints = []

    def load_checkpoint(self, checkpoint_path: Path) -> TrainFuture[None]:
        """Megatron restore happens during worker initialization, not from the host."""
        raise NotImplementedError(
            "Megatron remote checkpoint restore is initialization-owned. "
            "Set model.checkpoint_path before backend.initialize(), not backend.load_checkpoint(). "
            f"Received direct host restore request for {checkpoint_path}."
        )

    def shutdown(self) -> None:
        """Shutdown all workers."""
        for worker in self.workers:
            try:
                worker.send({"cmd": "shutdown"})
            except Exception as e:
                logger.warning("Failed to shutdown worker: %s", e)

    def _normalize_training_batch(self, batch: TrainingDatum | dict[str, Any]) -> dict[str, Any]:
        """Lower shared training datum into Megatron's backend-native batch shape.

        This is boundary normalization only. Megatron still executes its own
        native runtime/loss semantics after this lowering.
        """
        if isinstance(batch, dict):
            return batch

        normalized: dict[str, Any] = {"input_ids": batch.model_input.tokens}

        if batch.model_input.positions is not None:
            normalized["position_ids"] = batch.model_input.positions
        if batch.model_input.attention_mask is not None:
            normalized["attention_mask"] = batch.model_input.attention_mask

        supported_objective_keys = {
            "labels",
            "loss_mask",
            "advantages",
            "old_logprobs",
            "teacher_logprobs",
            "group_ids",
            "returns",
        }
        unsupported_keys = set(batch.objective_inputs) - supported_objective_keys
        if unsupported_keys:
            raise ValueError(
                "MegatronRemoteBackend does not support objective_inputs keys "
                f"{sorted(unsupported_keys)!r} in TrainingDatum yet."
            )

        normalized.update(batch.objective_inputs)
        return normalized

    def _serialize_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Serialize batch tensors for IPC.

        Converts torch tensors to lists for JSON serialization.
        """
        import torch

        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu().tolist()
            else:
                result[key] = value
        return result


def spawn_megatron_workers(
    num_gpus: int,
    config: MegatronRemoteConfig,
) -> list[Worker]:
    """Spawn local Megatron workers using miniray.

    Uses fork-based workers for single-node multi-GPU training.

    Args:
        num_gpus: Number of GPUs (workers) to spawn
        config: Megatron configuration

    Returns:
        List of Worker handles
    """
    from miniray import Worker

    # Import work function path
    work_fn_module = "rollouts.training.megatron_worker"

    def _emit_worker_bootstrap_diag(event: str, **data: Any) -> None:
        try:
            sys.stderr.write(
                "__ARGUS_DIAG__" + json.dumps({"event": event, **data}, sort_keys=True) + "\n"
            )
            sys.stderr.flush()
        except Exception:
            return

    def _work_fn(handle: Worker) -> None:
        """Wrapper that imports and calls the actual work function."""
        import importlib
        import traceback

        try:
            _emit_worker_bootstrap_diag(
                "megatron_worker_wrapper_import_start",
                module=work_fn_module,
                pid=os.getpid(),
            )
            module = importlib.import_module(work_fn_module)
            _emit_worker_bootstrap_diag(
                "megatron_worker_wrapper_import_ok",
                module=work_fn_module,
                pid=os.getpid(),
            )
            _emit_worker_bootstrap_diag(
                "megatron_worker_wrapper_train_start",
                module=work_fn_module,
                pid=os.getpid(),
            )
            module.train(handle)
        except Exception as e:
            # Print to stderr so we can see the error
            tb = traceback.format_exc()
            print(
                f"[WORKER ERROR] Failed to import/run {work_fn_module}: {e}",
                file=sys.stderr,
                flush=True,
            )
            print(tb, file=sys.stderr, flush=True)
            raise

    workers = [Worker(_work_fn) for _ in range(num_gpus)]
    logger.info("Spawned %d Megatron workers", num_gpus)

    return workers
