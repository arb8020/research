#!/usr/bin/env python3
"""Matplotlib visualization backend for optimizer data.

Following Casey Muratori: Presentation layer that consumes raw data.
No business logic here - just rendering.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from viz_data import ContourData, OptimizationVizData, TrajectoryData


def plot_loss_curve(
    trajectory: TrajectoryData,
    ax: plt.Axes | None = None,
    title: str = "Loss vs. Step"
) -> plt.Axes:
    """Plot loss curve over optimization steps.

    Args:
        trajectory: Trajectory data
        ax: Matplotlib axes (None to create new)
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(trajectory.steps, trajectory.losses, 'b-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_2d_trajectory(
    trajectory: TrajectoryData,
    landscape: ContourData,
    ax: plt.Axes | None = None,
    title: str = "Parameter Trajectory"
) -> plt.Axes:
    """Plot 2D parameter trajectory on loss landscape.

    Args:
        trajectory: Trajectory data (must be 2D params)
        landscape: Loss landscape contour data
        ax: Matplotlib axes (None to create new)
        title: Plot title

    Returns:
        Matplotlib axes
    """
    # Tiger Style: Assert preconditions
    assert len(trajectory.params_history) > 0, "Trajectory must have at least one point"
    assert trajectory.params_history[0].shape == (2,), "Params must be 2D"

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Extract x, y coordinates from trajectory
    x_vals = [p[0] for p in trajectory.params_history]
    y_vals = [p[1] for p in trajectory.params_history]

    # Plot landscape contours
    ax.contour(
        landscape.x,
        landscape.y,
        landscape.z,
        levels=20,
        alpha=0.3,
        cmap='viridis'
    )

    # Plot trajectory
    ax.plot(x_vals, y_vals, 'ro-', linewidth=2, markersize=6, label='Trajectory')
    ax.plot(x_vals[0], y_vals[0], 'go', markersize=10, label='Start')
    ax.plot(x_vals[-1], y_vals[-1], 'r*', markersize=15, label='End')

    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    return ax


def plot_optimization_summary(
    viz_data: OptimizationVizData,
    save_path: Path | None = None
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """Create complete optimization summary plot.

    Following Casey: One function to create the "standard" visualization.

    Args:
        viz_data: Complete visualization data
        save_path: Path to save figure (None to not save)

    Returns:
        (figure, axes) tuple
    """
    # Create figure
    if viz_data.landscape is not None:
        # 2D problem - show loss curve + trajectory
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        plot_loss_curve(
            viz_data.trajectory,
            ax=ax1,
            title=f"{viz_data.optimizer_name} - Loss (lr={viz_data.learning_rate})"
        )

        plot_2d_trajectory(
            viz_data.trajectory,
            viz_data.landscape,
            ax=ax2,
            title=f"{viz_data.optimizer_name} - Trajectory"
        )

        axes = (ax1, ax2)
    else:
        # Non-2D problem - just loss curve
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        plot_loss_curve(
            viz_data.trajectory,
            ax=ax,
            title=f"{viz_data.optimizer_name} - Loss (lr={viz_data.learning_rate})"
        )

        axes = (ax,)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to: {save_path}")

    return fig, axes


def compare_optimizers(
    viz_data_list: list[OptimizationVizData],
    save_path: Path | None = None
) -> tuple[plt.Figure, Any]:
    """Compare multiple optimizers side by side.

    Args:
        viz_data_list: List of optimizer visualization data
        save_path: Path to save figure (None to not save)

    Returns:
        (figure, axes) tuple
    """
    # Tiger Style: Assert preconditions
    assert len(viz_data_list) > 0, "Must provide at least one optimizer"

    n_optimizers = len(viz_data_list)

    # Check if 2D problem
    has_landscape = viz_data_list[0].landscape is not None

    if has_landscape:
        # Create 2 x n_optimizers grid
        fig, axes = plt.subplots(2, n_optimizers, figsize=(6 * n_optimizers, 10))
        if n_optimizers == 1:
            axes = axes.reshape(2, 1)

        for i, viz_data in enumerate(viz_data_list):
            # Top row: loss curves
            plot_loss_curve(
                viz_data.trajectory,
                ax=axes[0, i],
                title=f"{viz_data.optimizer_name} (lr={viz_data.learning_rate})"
            )

            # Bottom row: trajectories
            if viz_data.landscape is not None:
                plot_2d_trajectory(
                    viz_data.trajectory,
                    viz_data.landscape,
                    ax=axes[1, i],
                    title=f"{viz_data.optimizer_name}"
                )
    else:
        # Just loss curves in one row
        fig, axes = plt.subplots(1, n_optimizers, figsize=(6 * n_optimizers, 5))
        if n_optimizers == 1:
            axes = [axes]

        for i, viz_data in enumerate(viz_data_list):
            plot_loss_curve(
                viz_data.trajectory,
                ax=axes[i],
                title=f"{viz_data.optimizer_name} (lr={viz_data.learning_rate})"
            )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved comparison to: {save_path}")

    return fig, axes
