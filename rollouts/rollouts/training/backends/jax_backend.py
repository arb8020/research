"""JAX training backend (D6v3 - Future).

Raw JAX training with Flax/Optax.

Features:
- Pure functional training (JAX-native)
- JIT-compiled train step
- Immutable state management
- Orbax checkpointing
- TPU/XLA support

Status: STUB - Not yet implemented
Estimated effort: ~1-2 days
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# import jax
# import jax.numpy as jnp
# import optax
# import orbax.checkpoint as ocp

from rollouts.training.types import TrainFuture


@dataclass
class JAXTrainingBackend:
    """Raw JAX training backend (D6v3).

    Pure functional training using JAX/Flax/Optax.

    Status: STUB - Not yet implemented.

    Dependencies:
        - jax (pip install jax[cuda] or jax[tpu])
        - optax (pip install optax)
        - orbax-checkpoint (pip install orbax-checkpoint)

    Example (when implemented):
        >>> # Define pure function: (params, x) -> logits
        >>> def apply_fn(params, x):
        ...     return model_apply(params, x)
        >>>
        >>> # Create backend with JAX arrays
        >>> backend = JAXTrainingBackend(
        ...     apply_fn=apply_fn,
        ...     params=initial_params,
        ...     optimizer=optax.adam(1e-4),
        ...     loss_fn=loss_fn,
        ...     checkpoint_dir=Path("/ckpts"),
        ... )
        >>>
        >>> # Functional training (immutable updates)
        >>> metrics = await backend.forward_backward(batch).result()
    """

    apply_fn: Callable  # Pure function: (params, x) -> logits
    params: Any  # PyTree of JAX arrays (immutable)
    opt_state: Any  # Optimizer state (immutable)
    optimizer: Any  # optax.GradientTransformation
    loss_fn: Callable
    checkpoint_dir: Path

    weight_version: int = 0
    current_step: int = 0

    # JIT-compiled train step
    _train_step_fn: Optional[Callable] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize JIT-compiled train step."""
        raise NotImplementedError(
            "D6v3: JAX backend not yet implemented. "
            "See docs/D6_TRAINING_BACKEND.md for implementation plan."
        )

    def forward_backward(self, batch: Dict[str, Any]) -> TrainFuture[Dict[str, float]]:
        """Compute loss and gradients (JAX style).

        Note: In JAX, forward and backward are combined (jax.value_and_grad).
        Returns new params and opt_state (immutable update).
        """
        raise NotImplementedError("D6v3: Not yet implemented")

    def optim_step(self) -> TrainFuture[Dict[str, float]]:
        """Apply gradients (no-op in JAX, already done in forward_backward).

        This is a compatibility shim for the protocol. In JAX, the optimizer
        update happens in the same JIT-compiled function as the gradient
        computation.
        """
        raise NotImplementedError("D6v3: Not yet implemented")

    async def save_checkpoint(
        self,
        step: int,
        metrics: Dict[str, float] = {},
    ) -> Path:
        """Save checkpoint using Orbax."""
        raise NotImplementedError("D6v3: Not yet implemented")

    async def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint using Orbax."""
        raise NotImplementedError("D6v3: Not yet implemented")

    def get_weights(self) -> TrainFuture[Dict[str, Any]]:
        """Get model weights (JAX PyTree)."""
        raise NotImplementedError("D6v3: Not yet implemented")

    def load_weights(self, weights: Dict[str, Any]) -> TrainFuture[None]:
        """Load weights (JAX PyTree)."""
        raise NotImplementedError("D6v3: Not yet implemented")
