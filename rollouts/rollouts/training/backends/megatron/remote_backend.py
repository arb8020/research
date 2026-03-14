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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...contracts import TrainingDatum
from ...lowering import MegatronLowering
from ...types import ImmediateTrainFuture, TrainFuture

if TYPE_CHECKING:
    from miniray import Worker

logger = logging.getLogger(__name__)

_CONTROL_MESSAGE_MAX_BYTES = 64 * 1024


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

    # Lowered partition intent derived from RealizationPlan
    lowering: MegatronLowering = field(default_factory=MegatronLowering)

    # Backend-native Megatron runtime settings not modeled in RealizationPlan
    sequence_parallel: bool = False

    # Training
    lr: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    loss_type: str = "vanilla"
    mask_ratio_low: float = 0.125
    mask_ratio_high: float = 8.0
    micro_batch_size: int = 1
    global_batch_size: int = 8
    seq_length: int = 4096

    # NCCL
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    # Inference endpoints for weight sync
    inference_endpoints: list[str] = field(default_factory=list)

    # GPU assignment (which physical GPUs to use for each rank)
    # If None, workers use rank as GPU index (rank 0 -> GPU 0, etc.)
    cuda_device_ids: tuple[int, ...] | None = None


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
    _nccl_inference_endpoints: list[str] = field(default_factory=list, init=False)
    _nccl_initialized: bool = field(default=False, init=False)
    weight_version: int = 0

    _step: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)

    def _recv_response(self, worker: Worker, *, context: str, max_size: int) -> dict[str, Any]:
        response = worker.recv(max_size=max_size)
        if response.get("status") == "error":
            raise RuntimeError(f"Megatron worker failed during {context}: {response.get('error')}")
        return response

    def initialize(self) -> None:
        """Initialize all workers with config.

        Must be called before any training operations.
        """
        if self._initialized:
            return

        logger.info("Initializing %d Megatron workers...", len(self.workers))

        # Send init command to all workers
        for rank, worker in enumerate(self.workers):
            worker.send({
                "cmd": "init",
                "rank": rank,
                "world_size": len(self.workers),
                "config": {
                    "model_name": self.config.model_name,
                    "lowering": {
                        "parallel": {
                            "dp": self.config.lowering.parallel.dp,
                            "tp": self.config.lowering.parallel.tp,
                            "cp": self.config.lowering.parallel.cp,
                            "pp": self.config.lowering.parallel.pp,
                            "ep": self.config.lowering.parallel.ep,
                            "enable_loss_parallel": self.config.lowering.parallel.enable_loss_parallel,
                            "packed_sequences": self.config.lowering.parallel.packed_sequences,
                        },
                        "realization": {
                            "local_layouts": self.config.lowering.realization.local_layouts,
                            "collective_transitions": self.config.lowering.realization.collective_transitions,
                            "packed_sequences": self.config.lowering.realization.packed_sequences,
                        },
                    },
                    "sequence_parallel": self.config.sequence_parallel,
                    "lr": self.config.lr,
                    "bf16": self.config.dtype == "bfloat16",
                    "loss_type": self.config.loss_type,
                    "mask_ratio_low": self.config.mask_ratio_low,
                    "mask_ratio_high": self.config.mask_ratio_high,
                    "micro_batch_size": self.config.micro_batch_size,
                    "global_batch_size": self.config.global_batch_size,
                    "seq_length": self.config.seq_length,
                    "clip_grad": self.config.max_grad_norm,
                    "master_addr": self.config.master_addr,
                    "master_port": self.config.master_port,
                    "inference_endpoints": self.config.inference_endpoints,
                    "cuda_device_ids": self.config.cuda_device_ids,
                },
            })

        # Wait for rank 0 to confirm initialization
        response = self._recv_response(
            self.workers[0],
            context="initialize",
            max_size=_CONTROL_MESSAGE_MAX_BYTES,
        )
        assert response["status"] == "initialized", f"Init failed: {response}"

        self._initialized = True
        logger.info("All workers initialized")

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
        return self.forward_backward(batch)

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
        """Checkpoint restore is handled inside remote workers; stub for interface."""
        del checkpoint_path
        logger.info("Remote Megatron backend does not support direct checkpoint restore from host.")
        return ImmediateTrainFuture(None, operation="load_checkpoint")

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

    def _work_fn(handle: Worker) -> None:
        """Wrapper that imports and calls the actual work function."""
        import importlib
        import sys
        import traceback

        try:
            module = importlib.import_module(work_fn_module)
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
