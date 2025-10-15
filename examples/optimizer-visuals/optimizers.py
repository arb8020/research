#!/usr/bin/env python3
"""JAX optimizer implementations for learning and visualization.

Following Tiger Style:
- Simple, explicit control flow
- Assert all invariants
- Pure functions where possible
- No hidden state
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable, Any
from functools import partial


# Type definitions
PyTree = Any  # JAX pytree of parameters
GradFn = Callable[[PyTree], tuple[float, PyTree]]  # (loss, grads)


class OptimizerState(NamedTuple):
    """Optimizer state.

    Following Casey Muratori: Data is transparent.
    All optimizer state is visible and explicit.
    """
    step: int  # Current optimization step (starts at 0)
    params: PyTree  # Model parameters


class SGDState(NamedTuple):
    """SGD optimizer state - just parameters, no momentum yet."""
    step: int
    params: PyTree


def sgd_init(params: PyTree) -> SGDState:
    """Initialize SGD optimizer.

    Args:
        params: Initial model parameters

    Returns:
        Initial optimizer state
    """
    return SGDState(step=0, params=params)


def sgd_update(
    state: SGDState,
    grads: PyTree,
    learning_rate: float
) -> SGDState:
    """Single SGD update step: params = params - lr * grads

    Following Tiger Style:
    - Assert all preconditions
    - Pure function (no side effects)
    - Simple, explicit control flow

    Args:
        state: Current optimizer state
        grads: Gradients (same structure as params)
        learning_rate: Step size (must be positive)

    Returns:
        New optimizer state
    """
    # Tiger Style: Assert all preconditions
    assert learning_rate > 0.0, f"learning_rate must be positive, got {learning_rate}"
    assert state.step >= 0, f"step must be non-negative, got {state.step}"

    # Update parameters: new = old - lr * grad
    new_params = jax.tree.map(
        lambda p, g: p - learning_rate * g,
        state.params,
        grads
    )

    return SGDState(
        step=state.step + 1,
        params=new_params
    )


def optimize(
    init_params: PyTree,
    grad_fn: GradFn,
    init_fn: Callable[[PyTree], Any],
    update_fn: Callable[[Any, PyTree, float], Any],
    learning_rate: float,
    num_steps: int,
    return_grads: bool = False
) -> tuple[list[float], list[PyTree], list[Any], list[PyTree] | None]:
    """Run optimization loop and collect trajectory.

    Following Casey Muratori:
    - Separate allocation from initialization
    - User controls the optimizer (pass in init/update fns)
    - No hidden coupling

    Following Tiger Style:
    - Put a limit on everything (num_steps is bounded)
    - Assert all invariants

    Args:
        init_params: Initial parameters
        grad_fn: Function that computes (loss, grads) from params
        init_fn: Optimizer initialization function
        update_fn: Optimizer update function (state, grads, lr) -> new_state
        learning_rate: Learning rate for optimizer
        num_steps: Maximum number of optimization steps (must be > 0)
        return_grads: If True, also return gradient history

    Returns:
        Tuple of (losses, params_history, states_history, grads_history)
        - losses: Loss at each step
        - params_history: Parameters at each step
        - states_history: Optimizer state at each step
        - grads_history: Gradients at each step (None if return_grads=False)
    """
    # Tiger Style: Assert all preconditions
    assert num_steps > 0, f"num_steps must be positive, got {num_steps}"
    assert learning_rate > 0.0, f"learning_rate must be positive, got {learning_rate}"
    assert callable(grad_fn), "grad_fn must be callable"
    assert callable(init_fn), "init_fn must be callable"
    assert callable(update_fn), "update_fn must be callable"

    # Initialize optimizer
    state = init_fn(init_params)

    # Preallocate storage (Tiger Style: static allocation where possible)
    losses = []
    params_history = []
    states_history = []
    grads_history = [] if return_grads else None

    # Optimization loop (Tiger Style: bounded loop)
    for step in range(num_steps):
        # Get current params from optimizer state
        params = state.params

        # Compute loss and gradients
        loss, grads = grad_fn(params)

        # Store trajectory
        losses.append(float(loss))
        params_history.append(params)
        states_history.append(state)
        if return_grads:
            grads_history.append(grads)

        # Update optimizer state
        state = update_fn(state, grads, learning_rate)

    # Tiger Style: Assert postconditions
    assert len(losses) == num_steps
    assert len(params_history) == num_steps
    assert len(states_history) == num_steps
    if return_grads:
        assert len(grads_history) == num_steps

    return losses, params_history, states_history, grads_history


# Example loss function for testing
def quadratic_loss(params: jnp.ndarray) -> jnp.ndarray:
    """Simple quadratic loss: f(x) = x^T x / 2

    This is a convex function with a unique global minimum at x = 0.
    Gradient: grad = x

    Args:
        params: Parameters (any shape)

    Returns:
        Scalar loss value (0-dimensional array)
    """
    return jnp.sum(params ** 2) / 2.0


def main():
    """Test SGD on a simple 2D quadratic problem."""
    import matplotlib.pyplot as plt

    print("=" * 80)
    print("Testing SGD Optimizer")
    print("=" * 80)

    # Problem setup
    init_params = jnp.array([3.0, 2.0])
    learning_rate = 0.1
    num_steps = 20

    # Define grad function (loss + gradients)
    grad_fn = jax.value_and_grad(quadratic_loss)

    print(f"\nProblem: minimize f(x) = x^T x / 2")
    print(f"Initial params: {init_params}")
    print(f"Learning rate: {learning_rate}")
    print(f"Steps: {num_steps}\n")

    # Run optimization
    losses, params_history, states = optimize(
        init_params=init_params,
        grad_fn=grad_fn,
        init_fn=sgd_init,
        update_fn=sgd_update,
        learning_rate=learning_rate,
        num_steps=num_steps
    )

    # Print results
    print("Optimization trajectory:")
    for i in range(min(5, len(losses))):
        print(f"  Step {i}: loss={losses[i]:.6f}, params={params_history[i]}")
    print("  ...")
    print(f"  Step {len(losses)-1}: loss={losses[-1]:.6f}, params={params_history[-1]}")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Loss over time
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. Optimization Step')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter trajectory in 2D
    x_vals = [p[0] for p in params_history]
    y_vals = [p[1] for p in params_history]

    # Create contour plot of loss landscape
    x_range = jnp.linspace(-4, 4, 100)
    y_range = jnp.linspace(-4, 4, 100)
    X, Y = jnp.meshgrid(x_range, y_range)
    Z = (X**2 + Y**2) / 2.0

    ax2.contour(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
    ax2.plot(x_vals, y_vals, 'ro-', linewidth=2, markersize=6, label='SGD trajectory')
    ax2.plot(x_vals[0], y_vals[0], 'go', markersize=10, label='Start')
    ax2.plot(x_vals[-1], y_vals[-1], 'r*', markersize=15, label='End')
    ax2.set_xlabel('x[0]')
    ax2.set_ylabel('x[1]')
    ax2.set_title('Parameter Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig('/Users/chiraagbalu/research/examples/optimizer-visuals/sgd_test.png', dpi=150)
    print(f"\nVisualization saved to: examples/optimizer-visuals/sgd_test.png")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
