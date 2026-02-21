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

from ...training.types import TrainerConfig, TrainFuture

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
    loss_fn: Callable[[torch.Tensor, dict], torch.Tensor]
    checkpoint_dir: (
        Path  # TODO(ray): Replace with CheckpointStorage protocol for S3/distributed storage
    )
    device: torch.device | None = None  # Tiger Style: Explicit device (optional for CPU-only)
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    is_lora: bool = False  # Track if model is PEFT LoRA-wrapped

    # State (SLIME-inspired)
    weight_version: int = 0
    current_step: int = 0

    # Execution state
    _nursery: trio.Nursery | None = field(default=None, init=False, repr=False)
    _poisoned: bool = field(default=False, init=False, repr=False)

    # NCCL weight sync state (PipelineRL-inspired in-flight updates)
    _nccl_process_group: Any | None = field(default=None, init=False, repr=False)
    _nccl_inference_endpoints: list[str] = field(default_factory=list, init=False, repr=False)

    # FSDP checkpoint options (SLIME pattern - set in __post_init__)
    _fsdp_state_dict_opts: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
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
                assert device_index < num_gpus, (
                    f"Device {self.device} is invalid: only {num_gpus} GPU(s) available "
                    f"(indices 0-{num_gpus - 1}). Check your gpu_ranks config."
                )

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
        """Compute loss and gradients with gradient accumulation (returns future immediately).

        Supports micro-batching for memory efficiency (SLIME/Tinker pattern):
        - Splits batch into micro-batches based on trainer_config
        - Accumulates gradients across micro-batches
        - Scales loss by number of micro-batches for correct averaging

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

        try:
            # Zero gradients at the start
            self.optimizer.zero_grad()

            # Move batch to device if specified (Tiger Style: explicit device handling)
            if self.device is not None:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            # Compute number of micro-batches for gradient accumulation
            batch_size = batch["input_ids"].shape[0]
            num_minibatches = self.trainer_config.get_num_minibatches(batch_size)
            micro_batch_size = batch_size // num_minibatches

            # Accumulate loss and metrics across micro-batches
            total_loss = 0.0
            accumulated_metrics: dict[str, float] = {}

            for i in range(num_minibatches):
                start_idx = i * micro_batch_size
                end_idx = start_idx + micro_batch_size

                # Slice micro-batch
                micro_batch = {
                    k: v[start_idx:end_idx] if isinstance(v, torch.Tensor) and v.dim() > 0 else v
                    for k, v in batch.items()
                }

                # Forward pass on micro-batch
                output = self.model(micro_batch["input_ids"])

                # Extract logits from model output (HuggingFace models return ModelOutput objects)
                logits: torch.Tensor
                if hasattr(output, "logits"):
                    logits = output.logits
                else:
                    logits = output

                # Debug: check if logits have grad_fn (needed for backprop)
                if not logits.requires_grad:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"logits.requires_grad=False, logits.grad_fn={logits.grad_fn}")
                    # Check trainable params
                    trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
                    logger.warning(f"Model has {trainable} trainable parameters")

                # Compute loss (loss_fn returns (loss, metrics) or just loss)
                loss_result = self.loss_fn(logits, micro_batch)

                # Handle both (loss, metrics) tuple and bare loss tensor
                if isinstance(loss_result, tuple):
                    loss, metrics = loss_result
                else:
                    loss, metrics = loss_result, {}

                # Scale loss for gradient accumulation (SLIME pattern)
                # This ensures the total gradient is averaged correctly
                scaled_loss = loss / num_minibatches

                # Backward pass (gradients accumulate)
                scaled_loss.backward()

                # Track total loss for logging
                total_loss += loss.item()

                # Accumulate metrics (average across micro-batches)
                for k, v in metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v

            # Average loss and metrics
            avg_loss = total_loss / num_minibatches
            for k in accumulated_metrics:
                accumulated_metrics[k] /= num_minibatches

            # Compute grad norm (after accumulation, before clipping)
            grad_norm = (
                sum(
                    p.grad.norm().item() ** 2 for p in self.model.parameters() if p.grad is not None
                )
                ** 0.5
            )

            # Create future with immediate result
            # Merge accumulated metrics with standard metrics
            result = {
                "loss": avg_loss,
                "grad_norm": grad_norm,
                "num_minibatches": num_minibatches,
                "micro_batch_size": micro_batch_size,
                **accumulated_metrics,  # Include metrics from loss_fn
            }
            future: TrainFuture[dict[str, float]] = TrainFuture(operation="forward_backward")
            future.set_result(result)
            return future

        except Exception as e:
            # Poison backend on error
            self._poisoned = True
            raise RuntimeError(f"Training step failed: {e}") from e

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        """Apply gradients and update weights (returns future).

        Clips gradients if max_grad_norm is set in trainer_config (SLIME pattern).

        Returns:
            Future resolving to {"lr": float, "step": int, "grad_norm_clipped": float}

        Raises:
            AssertionError: If backend is poisoned
        """
        # Tiger Style: Assert preconditions
        assert not self._poisoned, "Backend is poisoned (previous error)"

        try:
            # Clip gradients if configured (SLIME pattern)
            grad_norm_clipped = None
            if self.trainer_config.max_grad_norm is not None:
                grad_norm_clipped = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.trainer_config.max_grad_norm,
                )
                grad_norm_clipped = float(grad_norm_clipped)

            # Apply gradients
            self.optimizer.step()
            self.current_step += 1

            # Get learning rate from first param group
            lr = self.optimizer.param_groups[0]["lr"]

            # Create future with immediate result
            result = {"lr": lr, "step": self.current_step}
            if grad_norm_clipped is not None:
                result["grad_norm_clipped"] = grad_norm_clipped

            future: TrainFuture[dict[str, float]] = TrainFuture(operation="optim_step")
            future.set_result(result)
            return future

        except Exception as e:
            # Poison backend on error
            self._poisoned = True
            raise RuntimeError(f"Optimizer step failed: {e}") from e

    async def save_checkpoint(
        self,
        step: int,
        metrics: dict[str, float] | None = None,
        prefix: str = "step_",
    ) -> Path:
        """Save checkpoint with version (increments weight_version).

        Args:
            step: Training step number
            metrics: Optional training metrics to save
            prefix: Directory prefix (default "step_", use "sync_" for temp sync checkpoints)

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
        if metrics is None:
            metrics = {}
        assert step >= 0, f"step must be >= 0, got {step}"
        assert not self._poisoned, "Backend is poisoned (previous error)"

        # Get rank early for logging
        rank = dist.get_rank() if dist.is_initialized() else 0
        is_distributed = dist.is_initialized()

        # DEBUG: Log checkpoint start
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"[CHECKPOINT DEBUG] Rank {rank}: Starting checkpoint save for step {step}")
        logger.debug(
            f"[CHECKPOINT DEBUG] Rank {rank}: Distributed={is_distributed}, FSDP={self._fsdp_state_dict_opts is not None}"
        )

        # Increment weight version (SLIME pattern)
        self.weight_version += 1
        logger.debug(
            f"[CHECKPOINT DEBUG] Rank {rank}: Incremented weight_version to {self.weight_version}"
        )

        # Create checkpoint directory (only rank 0, then barrier)
        ckpt_dir = self.checkpoint_dir / f"{prefix}{step:04d}"
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
            logger.debug(
                f"[CHECKPOINT DEBUG] Rank {rank}: Getting FSDP state dict (this may take time)..."
            )
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

    async def save_checkpoint_to_path(
        self,
        path: Path,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save checkpoint to a specific path (for fast sync to RAM disk).

        Unlike save_checkpoint(), this saves to an exact path (not checkpoint_dir).
        Used for syncing weights to /dev/shm for fast I/O.

        Args:
            path: Exact directory path to save checkpoint
            metrics: Optional training metrics to save

        Returns:
            Path to checkpoint directory (same as input path)
        """
        import logging
        import time

        import torch
        import torch.distributed as dist
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        if metrics is None:
            metrics = {}

        logger = logging.getLogger(__name__)
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Increment weight version
        self.weight_version += 1

        # Create directory
        path = Path(path)
        if rank == 0:
            path.mkdir(parents=True, exist_ok=True)

        # Barrier if distributed
        if dist.is_initialized():
            dist.barrier()

        # Get state dict
        if self._fsdp_state_dict_opts is not None:
            state_dict = get_model_state_dict(self.model, options=self._fsdp_state_dict_opts)
        else:
            state_dict = self.model.state_dict()

        # Only rank 0 saves
        if rank == 0:
            # Save model only (skip optimizer for sync - we just need weights)
            model_path = path / "pytorch_model.bin"
            await trio.to_thread.run_sync(torch.save, state_dict, model_path)

            # Save minimal metadata
            metadata = {
                "weight_version": self.weight_version,
                "timestamp": time.time(),
            }
            metadata_path = path / "metadata.json"
            await trio.to_thread.run_sync(self._write_json_metadata, metadata_path, metadata)

        # Barrier to ensure save completes
        if dist.is_initialized():
            dist.barrier()

        return path

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
        optimizer_state = await trio.to_thread.run_sync(
            torch.load, optimizer_path, {"map_location": "cpu"}
        )
        self.optimizer.load_state_dict(optimizer_state)

        # Restore weight version and step (SLIME pattern)
        self.weight_version = metadata["weight_version"]
        self.current_step = metadata["step"]

        # Barrier for FSDP coordination
        if dist.is_initialized():
            dist.barrier()

        return metadata

    async def save_hf_checkpoint(
        self,
        path: Path | str,
        tokenizer: Any | None = None,
    ) -> Path:
        """Save model in HuggingFace format for easy loading with from_pretrained().

        This is useful for the SFT → RL pipeline where SFT saves a checkpoint
        that RL can load directly as a model_name.

        Args:
            path: Directory to save the model
            tokenizer: Optional tokenizer to save alongside model

        Returns:
            Path to the saved directory

        Example:
            >>> await backend.save_hf_checkpoint("/tmp/sft_model")
            >>> # Later, in GRPO:
            >>> config = GRPOConfig(model_name="/tmp/sft_model", ...)
        """
        import logging

        logger = logging.getLogger(__name__)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        rank = dist.get_rank() if dist.is_initialized() else 0

        # Only rank 0 saves
        if rank == 0:
            logger.info(f"Saving HuggingFace checkpoint to {path}")

            # For FSDP, gather the full state dict first
            if self._fsdp_state_dict_opts is not None:
                state_dict = get_model_state_dict(self.model, options=self._fsdp_state_dict_opts)
                # Need to load into unwrapped model for save_pretrained
                # Get the underlying model from FSDP wrapper
                unwrapped = (
                    self.model._fsdp_wrapped_module
                    if hasattr(self.model, "_fsdp_wrapped_module")
                    else self.model
                )
                unwrapped.load_state_dict(state_dict)
                await trio.to_thread.run_sync(unwrapped.save_pretrained, path)
            else:
                await trio.to_thread.run_sync(self.model.save_pretrained, path)

            if tokenizer is not None:
                await trio.to_thread.run_sync(tokenizer.save_pretrained, path)

            logger.info(f"HuggingFace checkpoint saved to {path}")

        # Barrier for coordination
        if dist.is_initialized():
            dist.barrier()

        return path

    async def save_weights_for_sampler(
        self,
        path: Path | str,
    ) -> Path:
        """Save merged weights ready for inference engine (Tinker-inspired).

        If using LoRA, merges adapter weights into base model before saving.
        Saves as HuggingFace format for SGLang/vLLM compatibility.

        This is the weight sync primitive for TTT - it produces inference-ready
        weights that can be loaded via update_weights_from_disk.

        Uses atomic rename pattern: writes to temp dir, then renames to final path.
        This prevents race conditions where SGLang reads a partially-written checkpoint.

        Args:
            path: Directory to save merged weights

        Returns:
            Path to the saved directory

        Tiger Style: Assert postconditions.

        Example:
            >>> sync_dir = await backend.save_weights_for_sampler("/dev/shm/sync")
            >>> await inference_engine.update_weights_from_checkpoint(str(sync_dir))
        """
        import logging
        import shutil

        logger = logging.getLogger(__name__)
        path = Path(path)

        # Atomic rename pattern: write to temp dir, then rename
        # This prevents SGLang from reading a partially-written checkpoint
        temp_path = path.parent / f"{path.name}_tmp_{self.weight_version}"
        temp_path.mkdir(parents=True, exist_ok=True)

        rank = dist.get_rank() if dist.is_initialized() else 0

        # Only rank 0 saves
        if rank == 0:
            if self.is_lora:
                # LoRA: merge adapters temporarily for inference sync
                # W' = W + BA (low-rank merge)
                # IMPORTANT: Use merge_adapter() + unmerge_adapter() to preserve LoRA structure
                # merge_and_unload() would permanently destroy the LoRA adapters!
                logger.info(f"Merging LoRA weights for inference sync to {temp_path}")
                self.model.merge_adapter()  # Merge LoRA into base weights temporarily
                await trio.to_thread.run_sync(
                    lambda: self.model.base_model.model.save_pretrained(temp_path)
                )
                self.model.unmerge_adapter()  # Restore LoRA structure for continued training
                logger.info("LoRA merge and save complete")
            else:
                # Regular model: save directly
                logger.info(f"Saving weights for inference sync to {temp_path}")
                await trio.to_thread.run_sync(self.model.save_pretrained, temp_path)

            # Atomic rename: remove old dir, rename temp to final
            # shutil.rmtree + os.rename is not fully atomic, but close enough:
            # SGLang will either see old complete checkpoint or new complete checkpoint
            if path.exists():
                shutil.rmtree(path)
            temp_path.rename(path)
            logger.info(f"Atomically renamed {temp_path} -> {path}")

        # Barrier for coordination
        if dist.is_initialized():
            dist.barrier()

        # Tiger Style: assert postcondition
        config_path = path / "config.json"
        assert config_path.exists(), f"save_pretrained must create config.json at {config_path}"

        return path

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
            # Trainer config (gradient accumulation)
            "micro_batch_size": self.trainer_config.micro_batch_size,
            "num_minibatches": self.trainer_config.num_minibatches,
            "max_grad_norm": self.trainer_config.max_grad_norm,
            # Execution state
            "is_poisoned": self._poisoned,
            "is_fsdp": self._fsdp_state_dict_opts is not None,
            "device": str(self.device) if self.device else None,
        }

    # ══════════════════════════════════════════════════════════════
    # NCCL In-Flight Weight Sync (PipelineRL-inspired)
    # ══════════════════════════════════════════════════════════════

    async def init_nccl_weight_sync(
        self,
        inference_endpoints: list[str],
        master_addr: str | None = None,
        master_port: int = 29500,
    ) -> None:
        """Initialize NCCL process group for in-flight weight sync.

        Sets up NCCL communication between trainer and inference servers.
        Call once at startup before any sync_weights_nccl calls.

        Args:
            inference_endpoints: List of SGLang server URLs (e.g., ["http://localhost:30000"])
            master_addr: NCCL master address (default: localhost)
            master_port: NCCL master port (default: 29500)

        Example:
            >>> await backend.init_nccl_weight_sync(["http://localhost:30000"])
            >>> # Later, in training loop:
            >>> await backend.sync_weights_nccl()  # Non-blocking GPU-to-GPU sync
        """
        import logging
        import os

        import httpx

        logger = logging.getLogger(__name__)

        # Determine master address
        if master_addr is None:
            master_addr = os.environ.get("MASTER_ADDR", "localhost")

        # World size = trainer (1) + inference servers
        world_size = 1 + len(inference_endpoints)
        trainer_rank = 0

        logger.info(f"Initializing NCCL weight sync group (world_size={world_size})")
        logger.info(f"  Master: {master_addr}:{master_port}")
        logger.info(f"  Inference endpoints: {inference_endpoints}")

        # Store endpoints for later use
        self._nccl_inference_endpoints = list(inference_endpoints)

        # NCCL init requires all ranks to join concurrently.
        # SGLang's /init_weights_update_group blocks on dist.init_process_group,
        # so we must run the HTTP requests AND trainer join in parallel.
        import trio

        async def register_inference_endpoint(
            endpoint: str, rank: int, results: list[tuple[str, bool, str]]
        ) -> None:
            """Register an inference endpoint with the NCCL group."""
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    response = await client.post(
                        f"{endpoint}/init_weights_update_group",
                        json={
                            "master_address": master_addr,
                            "master_port": master_port,
                            "rank_offset": rank,
                            "world_size": world_size,
                            "group_name": "weight_sync",
                            "backend": "nccl",
                        },
                    )
                    response.raise_for_status()
                    results.append((endpoint, True, "OK"))
                except Exception as e:
                    results.append((endpoint, False, str(e)))

        async def trainer_join() -> None:
            """Trainer joins as rank 0 (runs in thread pool since it's blocking)."""
            if not dist.is_initialized():
                await trio.to_thread.run_sync(
                    lambda: dist.init_process_group(
                        backend="nccl",
                        init_method=f"tcp://{master_addr}:{master_port}",
                        rank=trainer_rank,
                        world_size=world_size,
                    )
                )

        # Launch all NCCL participants concurrently
        results: list[tuple[str, bool, str]] = []
        async with trio.open_nursery() as nursery:
            # Start inference endpoint registrations
            for i, endpoint in enumerate(inference_endpoints):
                inference_rank = i + 1
                logger.info(f"  Registering {endpoint} as rank {inference_rank}")
                nursery.start_soon(register_inference_endpoint, endpoint, inference_rank, results)

            # Trainer joins NCCL (this unblocks the inference endpoints)
            logger.info("  Trainer joining as rank 0...")
            nursery.start_soon(trainer_join)

        # Check results
        for endpoint, success, msg in results:
            if not success:
                raise RuntimeError(f"Failed to register {endpoint}: {msg}")
            logger.info(f"  {endpoint} registered successfully")

        # Create named group for weight sync
        self._nccl_process_group = dist.new_group(
            ranks=list(range(world_size)),
            backend="nccl",
        )

        logger.info("NCCL weight sync group initialized successfully")

    async def sync_weights_nccl(self) -> None:
        """Broadcast model weights to inference engines via NCCL.

        In-flight update: sampling can continue with slightly stale weights.
        GPU-to-GPU transfer - no disk I/O.

        Requires init_nccl_weight_sync() to be called first.

        Example:
            >>> # After each training step:
            >>> await backend.sync_weights_nccl()
            >>> # Inference engines now have updated weights
        """
        import logging

        import httpx

        logger = logging.getLogger(__name__)

        assert self._nccl_process_group is not None, (
            "NCCL weight sync not initialized. Call init_nccl_weight_sync() first."
        )

        # Increment weight version
        self.weight_version += 1

        # Get state dict (handles LoRA merging if needed)
        if self.is_lora:
            # Merge LoRA temporarily for sync
            self.model.merge_adapter()
            state_dict = self.model.base_model.model.state_dict()
        else:
            state_dict = self.model.state_dict()

        # Build parameter info for inference servers
        param_info = [
            {"name": name, "shape": list(p.shape), "dtype": str(p.dtype)}
            for name, p in state_dict.items()
        ]

        # 1. Tell inference servers to prepare for NCCL receive
        async with httpx.AsyncClient(timeout=300.0) as client:
            for endpoint in self._nccl_inference_endpoints:
                await client.post(
                    f"{endpoint}/update_weights_from_distributed",
                    json={
                        "names": [p["name"] for p in param_info],
                        "shapes": [p["shape"] for p in param_info],
                        "dtypes": [p["dtype"] for p in param_info],
                        "group_name": "weight_sync",
                        "weight_version": str(self.weight_version),
                    },
                )

        # 2. Broadcast each tensor via NCCL (GPU-to-GPU, no serialization)
        for name, param in state_dict.items():
            param_data = param.data.contiguous()
            if param_data.device.type != "cuda":
                param_data = param_data.cuda()
            dist.broadcast(param_data, src=0, group=self._nccl_process_group)

        # 3. Wait for completion
        dist.barrier(group=self._nccl_process_group)

        # Restore LoRA structure if needed
        if self.is_lora:
            self.model.unmerge_adapter()

        logger.debug(f"NCCL weight sync complete (version={self.weight_version})")

    async def cleanup_nccl_weight_sync(self) -> None:
        """Cleanup NCCL weight sync group.

        Call at shutdown to properly cleanup distributed resources.
        """
        import logging

        import httpx

        logger = logging.getLogger(__name__)

        if self._nccl_process_group is None:
            return

        logger.info("Cleaning up NCCL weight sync group...")

        # Tell SGLang servers to leave the group
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint in self._nccl_inference_endpoints:
                try:
                    await client.post(
                        f"{endpoint}/destroy_weights_update_group",
                        json={"group_name": "weight_sync"},
                    )
                except Exception:
                    pass  # Best effort cleanup

        # Destroy local process group
        if self._nccl_process_group is not None:
            dist.destroy_process_group(self._nccl_process_group)
            self._nccl_process_group = None

        self._nccl_inference_endpoints = []
        logger.info("NCCL weight sync group cleaned up")

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
