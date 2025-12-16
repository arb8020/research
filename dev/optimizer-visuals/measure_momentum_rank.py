#!/usr/bin/env python3
"""Measure if Adam's momentum matrix becomes low-rank during training.

Experiment:
- Simple linear regression: y = W @ x + noise
- Track singular values of momentum matrix M during Adam optimization
- Visualize: do most singular values → 0 while a few stay large?
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def linear_regression_loss(W: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """MSE loss for linear regression: ||y - W @ X||^2 / n

    Args:
        W: Weight matrix (output_dim, input_dim)
        X: Input data (input_dim, n_samples)
        y: Target data (output_dim, n_samples)

    Returns:
        Scalar MSE loss
    """
    predictions = W @ X
    return jnp.mean((predictions - y) ** 2)


def generate_linear_data(
    input_dim: int, output_dim: int, n_samples: int, key: jax.random.PRNGKey
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate synthetic linear regression data.

    Args:
        input_dim: Dimension of input
        output_dim: Dimension of output
        n_samples: Number of samples
        key: Random key

    Returns:
        (W_true, X, y) where y = W_true @ X + noise
    """
    key1, key2, key3 = jax.random.split(key, 3)

    # True weight matrix
    W_true = jax.random.normal(key1, (output_dim, input_dim))

    # Input data
    X = jax.random.normal(key2, (input_dim, n_samples))

    # Output with noise
    noise = 0.1 * jax.random.normal(key3, (output_dim, n_samples))
    y = W_true @ X + noise

    return W_true, X, y


def adam_step_with_state(
    W: jnp.ndarray,
    m: jnp.ndarray,
    v: jnp.ndarray,
    grad: jnp.ndarray,
    t: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single Adam step, returning updated (W, m, v).

    Args:
        W: Current weights
        m: Current momentum
        v: Current second moment
        grad: Current gradient
        t: Step number (1-indexed)
        lr: Learning rate
        beta1: Momentum decay
        beta2: Second moment decay
        eps: Numerical stability

    Returns:
        (W_new, m_new, v_new)
    """
    # Update biased moments
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad**2)

    # Bias correction (optional, but standard)
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)

    # Update weights
    W_new = W - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    return W_new, m_new, v_new


def main():
    print("=" * 80)
    print("Measuring Momentum Matrix Rank During Adam Optimization")
    print("=" * 80)

    # Setup problem
    input_dim = 20
    output_dim = 20
    n_samples = 100
    num_steps = 500
    lr = 0.01

    print(f"\nProblem: Linear regression with {output_dim}x{input_dim} weight matrix")
    print(f"Training samples: {n_samples}")
    print(f"Optimization steps: {num_steps}")
    print(f"Learning rate: {lr}\n")

    # Generate data
    key = jax.random.PRNGKey(42)
    W_true, X, y = generate_linear_data(input_dim, output_dim, n_samples, key)

    # Initialize weights
    W = jax.random.normal(jax.random.PRNGKey(0), (output_dim, input_dim)) * 0.1
    m = jnp.zeros_like(W)
    v = jnp.zeros_like(W)

    # Storage for singular values over time
    singular_values_history = []
    losses = []

    # Define loss function with data baked in
    def loss_fn(W_):
        return linear_regression_loss(W_, X, y)

    # Optimization loop
    print("Running Adam optimization...")
    for t in range(1, num_steps + 1):
        # Compute loss and gradient
        loss, grad = jax.value_and_grad(loss_fn)(W)
        losses.append(float(loss))

        # Adam step
        W, m, v = adam_step_with_state(W, m, v, grad, t, lr=lr)

        # Compute singular values of momentum matrix
        # Use numpy SVD for simplicity (could use JAX but numpy is fine here)
        m_np = np.array(m)
        singular_values = np.linalg.svd(m_np, compute_uv=False)
        singular_values_history.append(singular_values)

        if t % 100 == 0:
            print(
                f"  Step {t:4d}: loss={loss:.6f}, max_sv={singular_values[0]:.4f}, min_sv={singular_values[-1]:.4f}"
            )

    print("\nOptimization complete!")

    # Convert to array for plotting
    singular_values_history = np.array(
        singular_values_history
    )  # shape: (num_steps, min(output_dim, input_dim))

    # Visualize results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss curve
    ax1.plot(losses, "b-", linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: All singular values over time
    for i in range(singular_values_history.shape[1]):
        ax2.plot(singular_values_history[:, i], alpha=0.6, linewidth=1)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Singular Value")
    ax2.set_title("Momentum Matrix Singular Values Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Plot 3: Top 5 vs bottom 5 singular values
    top_k = 5
    for i in range(top_k):
        ax3.plot(singular_values_history[:, i], label=f"SV {i + 1}", linewidth=2)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Singular Value")
    ax3.set_title(f"Top {top_k} Singular Values")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")

    # Plot 4: Singular value spectrum at different times
    steps_to_plot = [50, 150, 300, 500]
    for step_idx in steps_to_plot:
        if step_idx <= num_steps:
            sv = singular_values_history[step_idx - 1]
            ax4.plot(sv, "o-", label=f"Step {step_idx}", alpha=0.7)
    ax4.set_xlabel("Singular Value Index")
    ax4.set_ylabel("Singular Value Magnitude")
    ax4.set_title("Singular Value Spectrum at Different Times")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale("log")

    plt.tight_layout()
    save_path = Path(__file__).parent / "momentum_rank_analysis.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to: {save_path}")

    # Compute and print rank statistics
    print("\n" + "=" * 80)
    print("Rank Analysis")
    print("=" * 80)

    # Define "effective rank" as number of singular values > 1% of max
    final_sv = singular_values_history[-1]
    threshold = 0.01 * final_sv[0]
    effective_rank = np.sum(final_sv > threshold)

    print("Final singular values (sorted):")
    for i, sv in enumerate(final_sv[:10]):
        print(f"  SV {i + 1:2d}: {sv:.6f} ({100 * sv / final_sv[0]:.1f}% of max)")

    print(f"\nMatrix shape: {output_dim}x{input_dim}")
    print(f"Maximum possible rank: {min(output_dim, input_dim)}")
    print(f"Effective rank (SV > 1% of max): {effective_rank}")
    print(f"Low-rank? {effective_rank < min(output_dim, input_dim) / 2}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
