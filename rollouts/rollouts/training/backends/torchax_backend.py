"""torchax training backend (D6v4 - Experimental).

PyTorch syntax running on JAX/XLA backend via torchax.

Features:
- Write PyTorch code, get JAX performance
- TPU support via XLA
- Automatic JAX backend translation

Status: STUB - Not implemented (experimental, not recommended)
Maturity: torchax v0.0.4 (very early)
Recommendation: Skip for now, revisit in 6-12 months
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# import torchax  # Experimental, v0.0.4
from ...training.types import TrainFuture


@dataclass
class TorchaxTrainingBackend:
    """torchax training backend (D6v4 - EXPERIMENTAL).

    PyTorch syntax running on JAX/XLA backend.

    Status: STUB - Not implemented.
    Maturity: EXPERIMENTAL (torchax v0.0.4, relocated Oct 2025)

    WARNING: torchax is very early stage. Not recommended for production.
    Consider torch.func (D6v2) or raw JAX (D6v3) instead.

    Dependencies:
        - torchax (pip install torchax) - v0.0.4 as of Oct 2025
        - May hit unsupported operations or bugs

    Example (when implemented):
        >>> model = GPT(config)
        >>> model = model.to("jax")  # torchax magic!
        >>>
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>>
        >>> backend = TorchaxTrainingBackend(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=loss_fn,
        ...     checkpoint_dir=Path("/ckpts"),
        ... )
        >>>
        >>> # PyTorch training loop, runs on JAX/TPU!
        >>> metrics = await backend.forward_backward(batch).result()
    """

    model: torch.nn.Module  # PyTorch model on "jax" device
    optimizer: torch.optim.Optimizer  # PyTorch optimizer (runs on JAX)
    loss_fn: Callable
    checkpoint_dir: Path

    weight_version: int = 0
    current_step: int = 0

    _poisoned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate model is on JAX device."""
        raise NotImplementedError(
            "D6v4: torchax backend not implemented (experimental). "
            "\n\n"
            "torchax is very early stage (v0.0.4 as of Oct 2025). "
            "Not recommended for production use. "
            "\n\n"
            "Consider alternatives: "
            "\n  - D6v1: Standard PyTorch (stable, production-ready)"
            "\n  - D6v2: torch.func (functional PyTorch, JAX-style)"
            "\n  - D6v3: Raw JAX (pure functional, TPU support)"
            "\n\n"
            "See docs/D6_TRAINING_BACKEND.md for details."
        )

    def forward_backward(self, batch: dict[str, Any]) -> TrainFuture[dict[str, float]]:
        """Standard PyTorch training step (runs on JAX runtime)."""
        raise NotImplementedError("D6v4: Not implemented")

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        """PyTorch optimizer step (runs on JAX)."""
        raise NotImplementedError("D6v4: Not implemented")

    async def save_checkpoint(
        self,
        step: int,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save checkpoint (PyTorch format, should work via torchax)."""
        raise NotImplementedError("D6v4: Not implemented")

    async def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load checkpoint (PyTorch format)."""
        raise NotImplementedError("D6v4: Not implemented")

    def get_weights(self) -> TrainFuture[dict[str, Any]]:
        """Get model weights (PyTorch state_dict on JAX)."""
        raise NotImplementedError("D6v4: Not implemented")

    def load_weights(self, weights: dict[str, Any]) -> TrainFuture[None]:
        """Load weights (PyTorch state_dict)."""
        raise NotImplementedError("D6v4: Not implemented")
