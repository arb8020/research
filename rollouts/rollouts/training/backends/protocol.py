"""Training backend protocol

Minimal surface area (Tinker-inspired):
- forward_backward(): Loss + gradients
- optim_step(): Update weights
- get_weights(): Get state for syncing

Tiger Style: Protocol-based, explicit operations.
"""

from typing import Any, Protocol

from ...training.contracts import LossFnLike, StepResult, TrainingDatum
from ...training.types import TrainFuture


class TrainingBackend(Protocol):
    """Protocol for training backends

    Implementations: PyTorchBackend, FSDPBackend, DeepSpeedBackend, etc.

    Tinker: Minimal surface (just 2 core operations).
    Casey: Protocol over inheritance (low coupling).
    """

    def forward_backward(
        self,
        datum: TrainingDatum,
        *,
        loss_fn: LossFnLike | None = None,
    ) -> TrainFuture[StepResult]:
        """Compute loss and gradients

        Args:
            datum: Model-facing input plus objective-facing supervision/signals.
            loss_fn: Contract-native loss function over forward products and datum.

        Returns:
            Future resolving to StepResult.

        Tiger Style: Explicit datum contract, explicit return.
        Tinker: Returns future immediately (non-blocking).
        """
        ...

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        """Apply gradients and update weights

        Returns:
            Future resolving to {"lr": float, "step": int, ...}

        Tinker: Returns future immediately (non-blocking).
        """
        ...

    def get_weights(self) -> TrainFuture[dict[str, Any]]:
        """Get model weights for syncing to inference

        Returns:
            Future resolving to state_dict
        """
        ...

    def load_weights(self, weights: dict[str, Any]) -> TrainFuture[None]:
        """Load model weights from inference or checkpoint

        Args:
            weights: state_dict to load
        """
        ...
