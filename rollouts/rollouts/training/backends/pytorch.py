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
        >>> backend = PyTorchTrainingBackend(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=my_loss_fn,
        ...     checkpoint_dir=Path("/checkpoints"),
        ... )
        >>>
        >>> # Training loop
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
        raise NotImplementedError("D6v1: PyTorch backend not yet implemented")

    def optim_step(self) -> TrainFuture[Dict[str, float]]:
        """Apply gradients and update weights (returns future).

        Returns:
            Future resolving to {"lr": float, "step": int}

        Raises:
            AssertionError: If backend is poisoned
        """
        raise NotImplementedError("D6v1: PyTorch backend not yet implemented")

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
        raise NotImplementedError("D6v1: PyTorch backend not yet implemented")

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
        raise NotImplementedError("D6v1: PyTorch backend not yet implemented")

    def get_weights(self) -> TrainFuture[Dict[str, Any]]:
        """Get model weights for syncing to inference.

        Returns:
            Future resolving to model.state_dict()
        """
        raise NotImplementedError("D6v1: PyTorch backend not yet implemented")

    def load_weights(self, weights: Dict[str, Any]) -> TrainFuture[None]:
        """Load model weights from inference or checkpoint.

        Args:
            weights: state_dict to load

        Returns:
            Future resolving to None
        """
        raise NotImplementedError("D6v1: PyTorch backend not yet implemented")
