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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import time
import json

import torch
import trio

from rollouts.training.types import TrainFuture


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
    checkpoint_dir: Path
    device: Optional[torch.device] = None  # Tiger Style: Explicit device (optional for CPU-only)

    # State (SLIME-inspired)
    weight_version: int = 0
    current_step: int = 0

    # Execution state
    _nursery: Optional[trio.Nursery] = field(default=None, init=False, repr=False)
    _poisoned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate initialization (Tiger Style)."""
        assert self.model is not None, "model cannot be None"
        assert self.optimizer is not None, "optimizer cannot be None"
        assert self.loss_fn is not None, "loss_fn cannot be None"
        assert self.checkpoint_dir is not None, "checkpoint_dir cannot be None"

        # Create checkpoint directory if needed
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def forward_backward(self, batch: Dict[str, Any]) -> TrainFuture[Dict[str, float]]:
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
            logits = self.model(batch["input_ids"])

            # Compute loss (user-provided loss function)
            loss = self.loss_fn(logits, batch["labels"], batch["loss_mask"])

            # Backward pass
            loss.backward()

            # Compute grad norm (before clipping)
            grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in self.model.parameters()
                if p.grad is not None
            ) ** 0.5

            # Create future with immediate result
            future: TrainFuture[Dict[str, float]] = TrainFuture(operation="forward_backward")
            future.set_result({"loss": loss.item(), "grad_norm": grad_norm})
            return future

        except Exception as e:
            # Poison backend on error
            self._poisoned = True
            raise RuntimeError(f"Training step failed: {e}") from e

    def optim_step(self) -> TrainFuture[Dict[str, float]]:
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
            future: TrainFuture[Dict[str, float]] = TrainFuture(operation="optim_step")
            future.set_result({"lr": lr, "step": self.current_step})
            return future

        except Exception as e:
            # Poison backend on error
            self._poisoned = True
            raise RuntimeError(f"Optimizer step failed: {e}") from e

    async def save_checkpoint(
        self,
        step: int,
        metrics: Dict[str, float] = {},
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
        """
        # Tiger Style: Assert preconditions
        assert step >= 0, f"step must be >= 0, got {step}"
        assert not self._poisoned, "Backend is poisoned (previous error)"

        # Increment weight version (SLIME pattern)
        self.weight_version += 1

        # Create checkpoint directory (nanochat pattern)
        ckpt_dir = self.checkpoint_dir / f"step_{step:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = ckpt_dir / "pytorch_model.bin"
        torch.save(self.model.state_dict(), model_path)

        # Save optimizer state
        optimizer_path = ckpt_dir / "optimizer.bin"
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # Save metadata (nanochat + SLIME pattern)
        metadata = {
            "step": step,
            "weight_version": self.weight_version,
            "timestamp": time.time(),
            "metrics": metrics,
        }
        metadata_path = ckpt_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return ckpt_dir

    async def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint and restore weight_version.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Metadata dict from checkpoint

        Side effects:
            - Loads model and optimizer state
            - Restores self.weight_version from metadata
            - Updates self.current_step
        """
        # Tiger Style: Assert preconditions
        assert checkpoint_path.exists(), f"Checkpoint directory does not exist: {checkpoint_path}"
        assert checkpoint_path.is_dir(), f"Checkpoint path must be a directory: {checkpoint_path}"

        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        assert metadata_path.exists(), f"metadata.json not found in {checkpoint_path}"

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load model state
        model_path = checkpoint_path / "pytorch_model.bin"
        assert model_path.exists(), f"pytorch_model.bin not found in {checkpoint_path}"
        self.model.load_state_dict(torch.load(model_path))

        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.bin"
        assert optimizer_path.exists(), f"optimizer.bin not found in {checkpoint_path}"
        self.optimizer.load_state_dict(torch.load(optimizer_path))

        # Restore weight version and step (SLIME pattern)
        self.weight_version = metadata["weight_version"]
        self.current_step = metadata["step"]

        return metadata

    def get_weights(self) -> TrainFuture[Dict[str, Any]]:
        """Get model weights for syncing to inference.

        Returns:
            Future resolving to model.state_dict()
        """
        # Tiger Style: Assert preconditions
        assert not self._poisoned, "Backend is poisoned (previous error)"

        # Get state dict (simple synchronous operation)
        state_dict = self.model.state_dict()

        # Create future with immediate result
        future: TrainFuture[Dict[str, Any]] = TrainFuture(operation="get_weights")
        future.set_result(state_dict)
        return future

    def load_weights(self, weights: Dict[str, Any]) -> TrainFuture[None]:
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
