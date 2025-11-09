"""Training backend protocol

Minimal surface area (Tinker-inspired):
- forward_backward(): Loss + gradients
- optim_step(): Update weights
- get_weights(): Get state for syncing

Tiger Style: Protocol-based, explicit operations.
"""

from typing import Protocol, Dict, Any
from rollouts.training.types import TrainFuture


class TrainingBackend(Protocol):
    """Protocol for training backends

    Implementations: PyTorchBackend, FSDPBackend, DeepSpeedBackend, etc.

    Tinker: Minimal surface (just 2 core operations).
    Casey: Protocol over inheritance (low coupling).
    """

    def forward_backward(self, batch: Dict[str, Any]) -> TrainFuture[Dict[str, float]]:
        """Compute loss and gradients

        Args:
            batch: {
                "input_ids": List[List[int]],
                "labels": List[List[int]],
                "attention_mask": List[List[int]],
            }

        Returns:
            Future resolving to {"loss": float, "grad_norm": float, ...}

        Tiger Style: Explicit batch format, explicit return.
        Tinker: Returns future immediately (non-blocking).
        """
        ...

    def optim_step(self) -> TrainFuture[Dict[str, float]]:
        """Apply gradients and update weights

        Returns:
            Future resolving to {"lr": float, "step": int, ...}

        Tinker: Returns future immediately (non-blocking).
        """
        ...

    def get_weights(self) -> TrainFuture[Dict[str, Any]]:
        """Get model weights for syncing to inference

        Returns:
            Future resolving to state_dict
        """
        ...

    def load_weights(self, weights: Dict[str, Any]) -> TrainFuture[None]:
        """Load model weights from inference or checkpoint

        Args:
            weights: state_dict to load
        """
        ...
