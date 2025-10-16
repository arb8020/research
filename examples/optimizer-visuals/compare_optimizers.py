#!/usr/bin/env python3
"""Compare SGD vs Adam on narrow valley problem."""

import sys
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from optimizers import sgd_init, sgd_update, adam_init, adam_update, optimize
from loss_landscapes import narrow_valley_loss, narrow_valley_loss_numpy
from viz_data import create_viz_data
from viz_matplotlib import compare_optimizers
from viz_raw import print_optimization_summary_raw


def main():
    show_raw = "--raw" in sys.argv
    print("=" * 80)
    print("Comparing SGD vs Adam on Narrow Valley")
    print("=" * 80)

    # Setup problem
    init_params = jnp.array([1.0, 3.0])
    grad_fn = jax.value_and_grad(narrow_valley_loss)
    num_steps = 50

    # Run SGD
    print("\nRunning SGD (lr=0.01)...")
    sgd_losses, sgd_params, _, sgd_grads = optimize(
        init_params=init_params,
        grad_fn=grad_fn,
        init_fn=sgd_init,
        update_fn=sgd_update,
        learning_rate=0.01,
        num_steps=num_steps,
        return_grads=show_raw
    )

    # Run Adam
    print("Running Adam (lr=0.1)...")
    adam_losses, adam_params, _, adam_grads = optimize(
        init_params=init_params,
        grad_fn=grad_fn,
        init_fn=adam_init,
        update_fn=adam_update,
        learning_rate=0.1,
        num_steps=num_steps,
        return_grads=show_raw
    )

    # Create viz data
    print("Generating visualizations...")

    sgd_viz = create_viz_data(
        losses=sgd_losses,
        params_history=sgd_params,
        loss_fn=narrow_valley_loss_numpy,
        optimizer_name="SGD",
        learning_rate=0.01,
        landscape_bounds=((-2.0, 2.0), (-4.0, 4.0)),
        landscape_resolution=100
    )

    adam_viz = create_viz_data(
        losses=adam_losses,
        params_history=adam_params,
        loss_fn=narrow_valley_loss_numpy,
        optimizer_name="Adam",
        learning_rate=0.1,
        landscape_bounds=((-2.0, 2.0), (-4.0, 4.0)),
        landscape_resolution=100
    )

    if show_raw:
        # Print raw data
        print("\n" + "=" * 80)
        print("SGD Raw Data")
        print("=" * 80)
        sgd_grads_np = [np.array(g) for g in sgd_grads] if sgd_grads else None
        print_optimization_summary_raw(sgd_viz, gradients=sgd_grads_np, max_steps=10)

        print("\n" + "=" * 80)
        print("Adam Raw Data")
        print("=" * 80)
        adam_grads_np = [np.array(g) for g in adam_grads] if adam_grads else None
        print_optimization_summary_raw(adam_viz, gradients=adam_grads_np, max_steps=10)
    else:
        # Compare visualizations
        save_path = Path(__file__).parent / "sgd_vs_adam.png"
        compare_optimizers([sgd_viz, adam_viz], save_path=save_path)

    # Print final losses
    print(f"\nFinal losses after {num_steps} steps:")
    print(f"  SGD:  {sgd_losses[-1]:.6f}")
    print(f"  Adam: {adam_losses[-1]:.6f}")
    print(f"\nAdam is {sgd_losses[-1] / adam_losses[-1]:.2f}x better!")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
