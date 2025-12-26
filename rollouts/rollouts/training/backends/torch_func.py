"""torch.func training backend (D6v2 - Future).

Functional PyTorch training using torch.func + torchopt.

Features:
- JAX-style functional training in PyTorch
- Immutable parameter updates
- torch.func for grad/vmap transforms
- torchopt for functional optimizers

Status: STUB - Not yet implemented
Estimated effort: ~1-2 days
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch

# import torchopt  # External dependency, install when implementing
from rollouts.training.types import TrainFuture


@dataclass
class TorchFuncTrainingBackend:
    """Functional PyTorch training backend (D6v2).

    JAX-style functional training using torch.func + torchopt.

    Status: STUB - Not yet implemented.

    Dependencies:
        - torch >= 2.0 (torch.func built-in)
        - torchopt (pip install torchopt)

    Example (when implemented):
        >>> model = GPT(config).to("cuda")
        >>> backend = TorchFuncTrainingBackend.from_model(
        ...     model=model,
        ...     optimizer_fn=lambda: torchopt.adam(lr=1e-4),
        ...     loss_fn=my_loss_fn,
        ...     checkpoint_dir=Path("/checkpoints"),
        ... )
        >>>
        >>> # Functional training (immutable updates)
        >>> metrics = await backend.forward_backward(batch).result()
        >>> step_metrics = await backend.optim_step().result()
    """

    model: torch.nn.Module  # Template (for functional_call)
    params: tuple  # Immutable parameters
    buffers: dict  # Buffers (BatchNorm, etc.)
    optimizer: Any  # torchopt.Optimizer (functional)
    opt_state: Any  # Optimizer state (immutable)
    loss_fn: Callable
    checkpoint_dir: Path

    weight_version: int = 0
    current_step: int = 0

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        optimizer_fn: Callable,
        loss_fn: Callable,
        checkpoint_dir: Path,
    ) -> NoReturn:
        """Create from PyTorch model (extracts functional params).

        Args:
            model: PyTorch model
            optimizer_fn: Factory for torchopt optimizer (e.g., torchopt.adam(lr=1e-4))
            loss_fn: Loss function
            checkpoint_dir: Checkpoint directory

        Returns:
            TorchFuncTrainingBackend instance
        """
        raise NotImplementedError(
            "D6v2: torch.func backend not yet implemented. "
            "See docs/D6_TRAINING_BACKEND.md for implementation plan."
        )

    def forward_backward(self, batch: dict[str, Any]) -> TrainFuture[dict[str, float]]:
        """Compute loss and gradients (functional style).

        Uses torch.func.grad_and_value for JAX-style autodiff.
        """
        raise NotImplementedError("D6v2: Not yet implemented")

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        """Apply gradients (functional optimizer update)."""
        raise NotImplementedError("D6v2: Not yet implemented")

    async def save_checkpoint(
        self,
        step: int,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save checkpoint (functional params as PyTorch state_dict)."""
        raise NotImplementedError("D6v2: Not yet implemented")

    async def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load checkpoint and restore functional params."""
        raise NotImplementedError("D6v2: Not yet implemented")

    def get_weights(self) -> TrainFuture[dict[str, Any]]:
        """Get model weights (convert tuple to state_dict)."""
        raise NotImplementedError("D6v2: Not yet implemented")

    def load_weights(self, weights: dict[str, Any]) -> TrainFuture[None]:
        """Load weights (convert state_dict to tuple)."""
        raise NotImplementedError("D6v2: Not yet implemented")
