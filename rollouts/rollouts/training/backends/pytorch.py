"""PyTorch training backend (D6v1).

Standard PyTorch training with OOP, stateful model/optimizer.

Features:
- Async training via Trio futures
- Weight version tracking (SLIME-inspired)
- Simple checkpoint format (nanochat-inspired)
- Minimal surface area (Tinker-inspired)

Tiger Style: Explicit state, assert preconditions.
Casey Muratori: Minimal coupling, futures for pipelining.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import trio

from rollouts.training.types import TrainFuture

# FSDP checkpoint support (SLIME pattern)
try:
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    FSDP = None
    StateDictOptions = None
    get_model_state_dict = None


@dataclass
class PyTorchTrainingBackend:
    """Future-based PyTorch training backend (D6v1).

    Implements TrainingBackend protocol for standard PyTorch models.

    Example:
        >>> model = GPT(config).to("cuda")
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> device = torch.device("cuda:0")
        >>> backend = PyTorchTrainingBackend(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=my_loss_fn,
        ...     checkpoint_dir=Path("/checkpoints"),
        ...     device=device,
        ... )
        >>>
        >>> # Training loop (batches automatically moved to device)
        >>> metrics = await backend.forward_backward(batch).result()
        >>> step_metrics = await backend.optim_step().result()
        >>>
        >>> # Save checkpoint (increments weight_version)
        >>> ckpt_path = await backend.save_checkpoint(step, metrics)
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    checkpoint_dir: Path  # TODO(ray): Replace with CheckpointStorage protocol for S3/distributed storage
    device: torch.device | None = None  # Tiger Style: Explicit device (optional for CPU-only)

    # State (SLIME-inspired)
    weight_version: int = 0
    current_step: int = 0

    # Execution state
    _nursery: trio.Nursery | None = field(default=None, init=False, repr=False)
    _poisoned: bool = field(default=False, init=False, repr=False)

    # FSDP checkpoint options (SLIME pattern - set in __post_init__)
    _fsdp_state_dict_opts: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate initialization (Tiger Style)."""
        assert self.model is not None, "model cannot be None"
        assert self.optimizer is not None, "optimizer cannot be None"
        assert self.loss_fn is not None, "loss_fn cannot be None"
        assert self.checkpoint_dir is not None, "checkpoint_dir cannot be None"

        # Validate device if specified (Tiger Style: fail fast with clear error)
        if self.device is not None:
            if self.device.type == "cuda":
                device_index = self.device.index if self.device.index is not None else 0
                num_gpus = torch.cuda.device_count()
                assert device_index < num_gpus, \
                    f"Device {self.device} is invalid: only {num_gpus} GPU(s) available " \
                    f"(indices 0-{num_gpus - 1}). Check your gpu_ranks config."

                # Verify device is actually accessible (not just in range)
                try:
                    # Small allocation to verify device works
                    _ = torch.zeros(1, device=self.device)
                except RuntimeError as e:
                    raise AssertionError(
                        f"Device {self.device} exists but is not accessible: {e}\n"
                        f"This may indicate the device is in use or misconfigured."
                    ) from e

        # Create checkpoint directory if needed
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Detect FSDP and configure state dict options (SLIME pattern)
        if FSDP_AVAILABLE and isinstance(self.model, FSDP):
            # Use new PyTorch checkpoint API with CPU offload for FSDP
            # This matches SLIME's approach in fsdp_utils/actor.py:73-75
            self._fsdp_state_dict_opts = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,  # Offload to CPU to avoid OOM during checkpoint
            )
        else:
            self._fsdp_state_dict_opts = None

    def forward_backward(self, batch: dict[str, Any]) -> TrainFuture[dict[str, float]]:
        """Compute loss and gradients (returns future immediately).

        Args:
            batch: {
                "input_ids": torch.Tensor,  # [batch, seq_len]
                "labels": torch.Tensor,     # [batch, seq_len]
                "loss_mask": torch.Tensor,  # [batch, seq_len]
            }

        Returns:
            Future resolving to {"loss": float, "grad_norm": float}

        Raises:
            AssertionError: If backend is poisoned or batch is invalid
        """
        # Tiger Style: Assert preconditions
        assert not self._poisoned, "Backend is poisoned (previous error)"
        assert "input_ids" in batch, "batch must have 'input_ids'"
        assert "labels" in batch, "batch must have 'labels'"
        assert "loss_mask" in batch, "batch must have 'loss_mask'"

        # For D6v1, simplify: run synchronously and return completed future
        # (True async with trio.to_thread can be added later if needed)
        try:
            # Zero gradients
            self.optimizer.zero_grad()

            # Move batch to device if specified (Tiger Style: explicit device handling)
            if self.device is not None:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

            # Forward pass
            output = self.model(batch["input_ids"])

            # Extract logits from model output (HuggingFace models return ModelOutput objects)
            # Handle both raw tensors and ModelOutput objects
            # Type check would catch if we passed wrong type to loss_fn
            logits: torch.Tensor
            if hasattr(output, 'logits'):
                logits = output.logits  # HuggingFace ModelOutput -> extract tensor
            else:
                logits = output  # Raw tensor (custom models)

            # Compute loss (SLIME pattern: pass logits and batch dict)
            # Type hint on loss_fn ensures logits is torch.Tensor, not ModelOutput
            loss = self.loss_fn(logits=logits, batch=batch)

            # Backward pass
            loss.backward()

            # Compute grad norm (before clipping)
            grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in self.model.parameters()
                if p.grad is not None
            ) ** 0.5

            # Create future with immediate result
            future: TrainFuture[dict[str, float]] = TrainFuture(operation="forward_backward")
            future.set_result({"loss": loss.item(), "grad_norm": grad_norm})
            return future

        except Exception as e:
            # Poison backend on error
            self._poisoned = True
            raise RuntimeError(f"Training step failed: {e}") from e

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        """Apply gradients and update weights (returns future).

        Returns:
            Future resolving to {"lr": float, "step": int}

        Raises:
            AssertionError: If backend is poisoned
        """
        # Tiger Style: Assert preconditions
        assert not self._poisoned, "Backend is poisoned (previous error)"

        try:
            # Apply gradients
            self.optimizer.step()
            self.current_step += 1

            # Get learning rate from first param group
            lr = self.optimizer.param_groups[0]["lr"]

            # Create future with immediate result
            future: TrainFuture[dict[str, float]] = TrainFuture(operation="optim_step")
            future.set_result({"lr": lr, "step": self.current_step})
            return future

        except Exception as e:
            # Poison backend on error
            self._poisoned = True
            raise RuntimeError(f"Optimizer step failed: {e}") from e

    async def save_checkpoint(
        self,
        step: int,
        metrics: dict[str, float] = {},
    ) -> Path:
        """Save checkpoint with version (increments weight_version).

        Args:
            step: Training step number
            metrics: Optional training metrics to save

        Returns:
            Path to checkpoint directory (e.g., checkpoint_dir/step_0100)

        Side effects:
            - Increments self.weight_version
            - Creates checkpoint_dir/step_{step:04d}/
            - Saves pytorch_model.bin, optimizer.bin, metadata.json

        FSDP support:
            - Uses new PyTorch checkpoint API (get_model_state_dict) for FSDP models
            - Only rank 0 saves to disk
            - All ranks participate in barrier for coordination

        Ray/distributed storage readiness:
            TODO(ray): This method currently couples state extraction with I/O.
            When adding Ray/miniray, refactor to use dependency injection:

            Pattern to use:
                class CheckpointStorage(Protocol):
                    async def save(self, state: Dict, metadata: Dict) -> str: ...
                    async def load(self, checkpoint_id: str) -> Dict: ...

                # Then inject storage at construction:
                backend = PyTorchTrainingBackend(
                    ...,
                    checkpoint_storage=S3CheckpointStorage(bucket="..."),
                )

            This decouples:
            - State extraction (get_model_state_dict) from I/O (torch.save)
            - Local filesystem assumptions from distributed storage (S3, GCS)
            - Matches Casey Muratori's decoupling principle
            - Matches ray_design.txt: "Abstract Storage - Don't assume local filesystem"
        """
        # Tiger Style: Assert preconditions
        assert step >= 0, f"step must be >= 0, got {step}"
        assert not self._poisoned, "Backend is poisoned (previous error)"

        # Get rank early for logging
        rank = dist.get_rank() if dist.is_initialized() else 0
        is_distributed = dist.is_initialized()

        # DEBUG: Log checkpoint start
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Starting checkpoint save for step {step}")
        logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Distributed={is_distributed}, FSDP={self._fsdp_state_dict_opts is not None}")

        # Increment weight version (SLIME pattern)
        self.weight_version += 1
        logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Incremented weight_version to {self.weight_version}")

        # Create checkpoint directory (only rank 0, then barrier)
        ckpt_dir = self.checkpoint_dir / f"step_{step:04d}"
        if rank == 0:
            logger.debug(f"[CHECKPOINT DEBUG] Rank 0: Creating checkpoint directory: {ckpt_dir}")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("[CHECKPOINT DEBUG] Rank 0: Directory created successfully")

        # Barrier to ensure directory exists before all ranks proceed
        if is_distributed:
            logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Entering barrier #1 (before state_dict)")
            dist.barrier()
            logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Passed barrier #1")

        # Get model state dict (FSDP-aware using SLIME pattern)
        # This is where SLIME's fsdp_utils/actor.py:667 pattern is used
        if self._fsdp_state_dict_opts is not None:
            logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Getting FSDP state dict (this may take time)...")
            # FSDP model: use new PyTorch checkpoint API
            # get_model_state_dict handles gathering across ranks
            state_dict = get_model_state_dict(self.model, options=self._fsdp_state_dict_opts)
            logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Got FSDP state dict successfully")
        else:
            logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Getting regular PyTorch state dict...")
            # Regular PyTorch model
            state_dict = self.model.state_dict()
            logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Got state dict successfully")

        # Only rank 0 saves to disk (SLIME pattern: checkpoint.py:143)
        if rank == 0:
            logger.debug("[CHECKPOINT DEBUG] Rank 0: Starting disk I/O...")

            # Save model state
            model_path = ckpt_dir / "pytorch_model.bin"
            logger.debug(f"[CHECKPOINT DEBUG] Rank 0: Saving model to {model_path}...")
            await trio.to_thread.run_sync(torch.save, state_dict, model_path)
            logger.debug("[CHECKPOINT DEBUG] Rank 0: Model saved successfully")

            # Save optimizer state
            optimizer_path = ckpt_dir / "optimizer.bin"
            logger.debug("[CHECKPOINT DEBUG] Rank 0: Getting optimizer state dict...")
            optimizer_state = self.optimizer.state_dict()
            logger.debug(f"[CHECKPOINT DEBUG] Rank 0: Saving optimizer to {optimizer_path}...")
            await trio.to_thread.run_sync(torch.save, optimizer_state, optimizer_path)
            logger.debug("[CHECKPOINT DEBUG] Rank 0: Optimizer saved successfully")

            # Save metadata (nanochat + SLIME pattern)
            metadata = {
                "step": step,
                "weight_version": self.weight_version,
                "timestamp": time.time(),
                "metrics": metrics,
            }
            metadata_path = ckpt_dir / "metadata.json"
            logger.debug(f"[CHECKPOINT DEBUG] Rank 0: Saving metadata to {metadata_path}...")
            await trio.to_thread.run_sync(self._write_json_metadata, metadata_path, metadata)
            logger.debug("[CHECKPOINT DEBUG] Rank 0: Metadata saved successfully")
        else:
            logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Skipping disk I/O (not rank 0)")

        # Barrier to ensure rank 0 finishes before other ranks proceed
        # This prevents races if checkpoint path is used immediately after
        if is_distributed:
            logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Entering barrier #2 (after save)")
            dist.barrier()
            logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Passed barrier #2")

        logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Checkpoint save completed successfully!")
        return ckpt_dir

    async def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load checkpoint and restore weight_version.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Metadata dict from checkpoint

        Side effects:
            - Loads model and optimizer state
            - Restores self.weight_version from metadata
            - Updates self.current_step

        FSDP support:
            - FSDP models can load full state dicts directly
            - Barrier ensures all ranks coordinate during load
        """
        # Tiger Style: Assert preconditions
        assert checkpoint_path.exists(), f"Checkpoint directory does not exist: {checkpoint_path}"
        assert checkpoint_path.is_dir(), f"Checkpoint path must be a directory: {checkpoint_path}"

        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        assert metadata_path.exists(), f"metadata.json not found in {checkpoint_path}"

        # Use trio.to_thread for async-safe file I/O
        metadata = await trio.to_thread.run_sync(self._read_json_metadata, metadata_path)

        # Load model state
        model_path = checkpoint_path / "pytorch_model.bin"
        assert model_path.exists(), f"pytorch_model.bin not found in {checkpoint_path}"

        # Load state dict to CPU first, then load into model
        # FSDP models can handle full state dicts via load_state_dict
        state_dict = await trio.to_thread.run_sync(torch.load, model_path, {"map_location": "cpu"})
        self.model.load_state_dict(state_dict)

        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.bin"
        assert optimizer_path.exists(), f"optimizer.bin not found in {checkpoint_path}"
        optimizer_state = await trio.to_thread.run_sync(torch.load, optimizer_path, {"map_location": "cpu"})
        self.optimizer.load_state_dict(optimizer_state)

        # Restore weight version and step (SLIME pattern)
        self.weight_version = metadata["weight_version"]
        self.current_step = metadata["step"]

        # Barrier for FSDP coordination
        if dist.is_initialized():
            dist.barrier()

        return metadata

    def get_weights(self) -> TrainFuture[dict[str, Any]]:
        """Get model weights for syncing to inference.

        Returns:
            Future resolving to model.state_dict()

        FSDP support:
            - Uses new PyTorch checkpoint API for FSDP models
            - Returns full state dict (gathered to rank 0 if needed)
        """
        # Tiger Style: Assert preconditions
        assert not self._poisoned, "Backend is poisoned (previous error)"

        # Get state dict (FSDP-aware, same pattern as save_checkpoint)
        if self._fsdp_state_dict_opts is not None:
            # FSDP model: use new PyTorch checkpoint API
            state_dict = get_model_state_dict(self.model, options=self._fsdp_state_dict_opts)
        else:
            # Regular PyTorch model
            state_dict = self.model.state_dict()

        # Create future with immediate result
        future: TrainFuture[dict[str, Any]] = TrainFuture(operation="get_weights")
        future.set_result(state_dict)
        return future

    def load_weights(self, weights: dict[str, Any]) -> TrainFuture[None]:
        """Load model weights from inference or checkpoint.

        Args:
            weights: state_dict to load

        Returns:
            Future resolving to None
        """
        # Tiger Style: Assert preconditions
        assert not self._poisoned, "Backend is poisoned (previous error)"
        assert weights is not None, "weights cannot be None"

        try:
            # Load state dict
            self.model.load_state_dict(weights)

            # Create future with immediate result
            future: TrainFuture[None] = TrainFuture(operation="load_weights")
            future.set_result(None)
            return future

        except Exception as e:
            # Poison backend on error
            self._poisoned = True
            raise RuntimeError(f"Failed to load weights: {e}") from e

    def get_state_snapshot(self) -> dict[str, Any]:
        """Get complete state for debugging/introspection.

        Returns all state that affects training behavior.
        Useful for debugging, logging, and understanding what's happening.

        Tiger Style: Make hidden state visible when needed.

        Returns:
            Dict with all backend state including model, optimizer, and training state
        """
        try:
            num_params = sum(p.numel() for p in self.model.parameters())
            param_dtype = next(self.model.parameters()).dtype
            param_device = next(self.model.parameters()).device
        except StopIteration:
            num_params = 0
            param_dtype = None
            param_device = None

        return {
            # Model state
            "model_num_parameters": num_params,
            "model_dtype": str(param_dtype) if param_dtype else None,
            "model_device": str(param_device) if param_device else None,
            "model_is_training": self.model.training,

            # Optimizer state
            "optimizer_type": type(self.optimizer).__name__,
            "optimizer_num_param_groups": len(self.optimizer.param_groups),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "betas": self.optimizer.param_groups[0].get("betas", None),
            "weight_decay": self.optimizer.param_groups[0].get("weight_decay", 0),

            # Training state
            "current_step": self.current_step,
            "weight_version": self.weight_version,

            # Execution state
            "is_poisoned": self._poisoned,
            "is_fsdp": self._fsdp_state_dict_opts is not None,
            "device": str(self.device) if self.device else None,
        }

    # Helper methods for async file I/O (Tiger Style: explicit sync methods)
    @staticmethod
    def _write_json_metadata(path: Path, data: dict[str, Any]) -> None:
        """Blocking helper to write JSON metadata (called via trio.to_thread)."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _read_json_metadata(path: Path) -> dict[str, Any]:
        """Blocking helper to read JSON metadata (called via trio.to_thread)."""
        with open(path) as f:
            return json.load(f)
