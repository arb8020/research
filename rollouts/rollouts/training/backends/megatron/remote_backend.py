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

from ...types import ImmediateTrainFuture, TrainFuture

if TYPE_CHECKING:
    from miniray import Worker

logger = logging.getLogger(__name__)


@dataclass
class MegatronRemoteConfig:
    """Configuration for remote Megatron backend."""

    # Model
    model_name: str
    dtype: str = "bfloat16"

    # Parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1

    # Training
    lr: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    micro_batch_size: int = 1
    global_batch_size: int = 8
    seq_length: int = 4096

    # NCCL
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    # Inference endpoints for weight sync
    inference_endpoints: list[str] = field(default_factory=list)


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
    weight_version: int = 0

    _step: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)

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
                    "tensor_parallel_size": self.config.tensor_parallel_size,
                    "pipeline_parallel_size": self.config.pipeline_parallel_size,
                    "expert_parallel_size": self.config.expert_parallel_size,
                    "lr": self.config.lr,
                    "bf16": self.config.dtype == "bfloat16",
                    "micro_batch_size": self.config.micro_batch_size,
                    "global_batch_size": self.config.global_batch_size,
                    "seq_length": self.config.seq_length,
                    "clip_grad": self.config.max_grad_norm,
                    "master_addr": self.config.master_addr,
                    "master_port": self.config.master_port,
                    "inference_endpoints": self.config.inference_endpoints,
                },
            })

        # Wait for rank 0 to confirm initialization
        response = self.workers[0].recv(max_size=1024)
        assert response["status"] == "initialized", f"Init failed: {response}"

        self._initialized = True
        logger.info("All workers initialized")

    def forward_backward(self, batch: dict[str, Any]) -> TrainFuture[dict[str, float]]:
        """Compute loss and gradients on batch.

        Args:
            batch: Training batch with input_ids, labels, etc.

        Returns:
            Future resolving to metrics dict with loss, etc.
        """
        assert self._initialized, "Call initialize() first"

        # Send batch to rank 0 (it broadcasts to other ranks)
        self.workers[0].send({
            "cmd": "train_step",
            "batch": self._serialize_batch(batch),
        })

        # Wait for metrics from rank 0
        response = self.workers[0].recv(max_size=10 * 1024 * 1024)
        assert response["status"] == "ok", f"Train step failed: {response}"

        self._step += 1
        return ImmediateTrainFuture(response["metrics"], operation="forward_backward")

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        """Apply gradients (already done in forward_backward for Megatron)."""
        # Megatron does optimizer step inside forward_backward
        return ImmediateTrainFuture({"step": self._step}, operation="optim_step")

    def sync_weights(self) -> None:
        """Sync weights to inference engines."""
        assert self._initialized, "Call initialize() first"

        self.workers[0].send({"cmd": "sync_weights"})
        response = self.workers[0].recv(max_size=1024)
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
        response = self.workers[0].recv(max_size=1024)
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
        """No-op stub for now (remote setup uses disk sync path)."""
        logger.debug(
            "Skipping NCCL init for remote Megatron backend (using worker-based weight sync path)."
        )
        del inference_endpoints
        del master_addr
        del master_port
        return None

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
