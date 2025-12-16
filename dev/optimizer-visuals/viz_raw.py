#!/usr/bin/env python3
"""Raw data printer - just arrays and numbers.

Following Casey Muratori: Simplest possible output.
"""

import numpy as np
from viz_data import OptimizationVizData, TrajectoryData


def print_trajectory_raw(
    trajectory: TrajectoryData, gradients: list[np.ndarray] | None = None, max_steps: int = 10
):
    """Print raw trajectory data.

    Args:
        trajectory: Trajectory data
        gradients: Optional list of gradients at each step
        max_steps: Maximum number of steps to print (default 10)
    """
    n_steps = min(len(trajectory.steps), max_steps)

    if gradients is not None:
        print("Step | Loss      | Params                              | Gradients")
        print("-----|-----------|-------------------------------------|" + "-" * 40)
    else:
        print("Step | Loss      | Params")
        print("-----|-----------|" + "-" * 50)

    for i in range(n_steps):
        params_str = str(trajectory.params_history[i])
        if gradients is not None and i < len(gradients):
            grads_str = str(gradients[i])
            print(
                f"{trajectory.steps[i]:4d} | {trajectory.losses[i]:9.6f} | {params_str:35s} | {grads_str}"
            )
        else:
            print(f"{trajectory.steps[i]:4d} | {trajectory.losses[i]:9.6f} | {params_str}")

    if len(trajectory.steps) > n_steps:
        print("...")
        i = len(trajectory.steps) - 1
        params_str = str(trajectory.params_history[i])
        if gradients is not None and i < len(gradients):
            grads_str = str(gradients[i])
            print(
                f"{trajectory.steps[i]:4d} | {trajectory.losses[i]:9.6f} | {params_str:35s} | {grads_str}"
            )
        else:
            print(f"{trajectory.steps[i]:4d} | {trajectory.losses[i]:9.6f} | {params_str}")


def print_optimization_summary_raw(
    viz_data: OptimizationVizData, gradients: list[np.ndarray] | None = None, max_steps: int = 10
):
    """Print optimization summary as raw data.

    Args:
        viz_data: Complete visualization data
        gradients: Optional list of gradients at each step
        max_steps: Maximum trajectory steps to show
    """
    print("=" * 80)
    print(f"Optimizer: {viz_data.optimizer_name}")
    print(f"Learning Rate: {viz_data.learning_rate}")
    print(f"Total Steps: {len(viz_data.trajectory.steps)}")
    print(f"Initial Loss: {viz_data.trajectory.losses[0]:.6f}")
    print(f"Final Loss: {viz_data.trajectory.losses[-1]:.6f}")
    print("=" * 80)
    print()

    print_trajectory_raw(viz_data.trajectory, gradients=gradients, max_steps=max_steps)
