#!/usr/bin/env python3
"""Test visualization backends with optimizer data."""

import sys
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from optimizers import sgd_init, sgd_update, optimize
from loss_landscapes import quadratic_loss, narrow_valley_loss, narrow_valley_loss_numpy
from viz_data import create_viz_data
from viz_matplotlib import plot_optimization_summary, compare_optimizers
from viz_terminal import plot_optimization_summary_terminal
from viz_raw import print_optimization_summary_raw


def main():
    # Parse args
    show_terminal = "--terminal" in sys.argv or "--all" in sys.argv
    show_matplotlib = "--matplotlib" in sys.argv or "--all" in sys.argv
    show_raw = "--raw" in sys.argv or "--all" in sys.argv
    compare_lrs = "--compare-lr" in sys.argv
    use_valley = "--valley" in sys.argv

    # Default to matplotlib only
    if not show_terminal and not show_matplotlib and not show_raw:
        show_matplotlib = True

    print("=" * 80)
    print("Optimizer Visualizations")
    print("=" * 80)

    # Setup optimization problem
    if use_valley:
        print("\nUsing narrow valley loss landscape (steep in x, shallow in y)")
        init_params = jnp.array([1.0, 3.0])  # Start with large y, small x
        grad_fn = jax.value_and_grad(narrow_valley_loss)
        loss_fn_numpy = narrow_valley_loss_numpy
        landscape_bounds = ((-2.0, 2.0), (-4.0, 4.0))
        default_lr = 0.01  # Need smaller LR for steep valley
    else:
        print("\nUsing quadratic (bowl) loss landscape")
        init_params = jnp.array([3.0, 2.0])
        grad_fn = jax.value_and_grad(quadratic_loss)

        def loss_fn_numpy(params: np.ndarray) -> float:
            """Numpy version of quadratic loss for landscape computation."""
            return float(np.sum(params ** 2) / 2.0)

        landscape_bounds = ((-4.0, 4.0), (-4.0, 4.0))
        default_lr = 0.1

    if compare_lrs:
        # Compare different learning rates
        learning_rates = [0.01, 0.1, 0.5]
        viz_data_list = []

        for lr in learning_rates:
            print(f"\nRunning SGD with lr={lr}...")
            losses, params_history, _ = optimize(
                init_params=init_params,
                grad_fn=grad_fn,
                init_fn=sgd_init,
                update_fn=sgd_update,
                learning_rate=lr,
                num_steps=20
            )

            viz_data = create_viz_data(
                losses=losses,
                params_history=params_history,
                loss_fn=loss_fn_numpy,
                optimizer_name="SGD",
                learning_rate=lr,
                landscape_bounds=landscape_bounds,
                landscape_resolution=100
            )
            viz_data_list.append(viz_data)

        # Show comparison
        if show_matplotlib:
            print("\n" + "=" * 80)
            print("Matplotlib Comparison")
            print("=" * 80)
            save_path = Path(__file__).parent / "lr_comparison.png"
            compare_optimizers(viz_data_list, save_path=save_path)

    else:
        # Single run
        learning_rate = default_lr
        num_steps = 50 if use_valley else 20

        print(f"\nRunning SGD with lr={learning_rate}...")
        losses, params_history, _, grads_history = optimize(
            init_params=init_params,
            grad_fn=grad_fn,
            init_fn=sgd_init,
            update_fn=sgd_update,
            learning_rate=learning_rate,
            num_steps=num_steps,
            return_grads=show_raw
        )

        viz_data = create_viz_data(
            losses=losses,
            params_history=params_history,
            loss_fn=loss_fn_numpy,
            optimizer_name="SGD",
            learning_rate=learning_rate,
            landscape_bounds=landscape_bounds,
            landscape_resolution=100
        )

        # Show visualizations based on flags
        if show_raw:
            print("\n" + "=" * 80)
            print("Raw Data")
            print("=" * 80)
            # Convert JAX arrays to numpy for printing
            grads_np = [np.array(g) for g in grads_history] if grads_history else None
            print_optimization_summary_raw(viz_data, gradients=grads_np, max_steps=20)

        if show_terminal:
            print("\n" + "=" * 80)
            print("Terminal Visualization")
            print("=" * 80)
            plot_optimization_summary_terminal(viz_data)

        if show_matplotlib:
            print("\n" + "=" * 80)
            print("Matplotlib Visualization")
            print("=" * 80)
            save_path = Path(__file__).parent / "test_viz_output.png"
            plot_optimization_summary(viz_data, save_path=save_path)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
