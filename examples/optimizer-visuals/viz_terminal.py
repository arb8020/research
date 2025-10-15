#!/usr/bin/env python3
"""Terminal visualization backend using plotext.

Following Casey Muratori: Presentation layer that consumes raw data.
"""

import plotext as plt
from viz_data import OptimizationVizData, TrajectoryData


def plot_loss_curve_terminal(trajectory: TrajectoryData, title: str = "Loss vs. Step"):
    """Plot loss curve in terminal.

    Args:
        trajectory: Trajectory data
        title: Plot title
    """
    plt.clf()
    plt.plot(trajectory.steps, trajectory.losses, marker="braille")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.show()


def plot_2d_trajectory_terminal(
    trajectory: TrajectoryData,
    title: str = "Parameter Trajectory"
):
    """Plot 2D parameter trajectory in terminal.

    Args:
        trajectory: Trajectory data (must be 2D)
        title: Plot title
    """
    # Tiger Style: Assert preconditions
    assert len(trajectory.params_history) > 0, "Trajectory must have at least one point"
    assert trajectory.params_history[0].shape == (2,), "Params must be 2D"

    x_vals = [float(p[0]) for p in trajectory.params_history]
    y_vals = [float(p[1]) for p in trajectory.params_history]

    plt.clf()
    plt.plot(x_vals, y_vals, marker="braille")
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.title(title)
    plt.show()


def plot_optimization_summary_terminal(viz_data: OptimizationVizData):
    """Show optimization summary in terminal.

    Args:
        viz_data: Complete visualization data
    """
    print("=" * 80)
    print(f"Optimizer: {viz_data.optimizer_name}")
    print(f"Learning Rate: {viz_data.learning_rate}")
    print(f"Steps: {len(viz_data.trajectory.steps)}")
    print(f"Initial Loss: {viz_data.trajectory.losses[0]:.6f}")
    print(f"Final Loss: {viz_data.trajectory.losses[-1]:.6f}")
    print("=" * 80)
    print()

    # Loss curve
    plot_loss_curve_terminal(
        viz_data.trajectory,
        title=f"{viz_data.optimizer_name} - Loss"
    )

    # Trajectory if 2D
    if viz_data.landscape is not None:
        print()
        plot_2d_trajectory_terminal(
            viz_data.trajectory,
            title=f"{viz_data.optimizer_name} - Trajectory"
        )
