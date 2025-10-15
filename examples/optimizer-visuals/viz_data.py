#!/usr/bin/env python3
"""Pure data generation for optimizer visualizations.

Following Casey Muratori: Separate data generation from presentation.
This module produces raw data that can be consumed by any visualization backend.

Tiger Style:
- Pure functions (no side effects)
- Assert all invariants
- Explicit data structures
"""

import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Callable, Any
from dataclasses import dataclass


PyTree = Any


@dataclass
class TrajectoryData:
    """Raw optimizer trajectory data.

    Following Casey: Data is transparent, not opaque.
    All fields are public and directly accessible.
    """
    losses: list[float]  # Loss at each step
    params_history: list[np.ndarray]  # Parameters at each step (converted to numpy)
    steps: list[int]  # Step numbers

    def __post_init__(self):
        # Tiger Style: Assert all invariants
        assert len(self.losses) == len(self.params_history) == len(self.steps), \
            "All trajectory arrays must have same length"
        assert len(self.losses) > 0, "Trajectory must have at least one point"
        assert all(s >= 0 for s in self.steps), "Steps must be non-negative"


@dataclass
class ContourData:
    """Raw contour plot data for loss landscape.

    Following Casey: Separate concerns - trajectory vs landscape.
    """
    x: np.ndarray  # X coordinates (1D array)
    y: np.ndarray  # Y coordinates (1D array)
    z: np.ndarray  # Loss values (2D array, shape [len(y), len(x)])

    def __post_init__(self):
        # Tiger Style: Assert all invariants
        assert self.x.ndim == 1, f"x must be 1D, got {self.x.ndim}D"
        assert self.y.ndim == 1, f"y must be 1D, got {self.y.ndim}D"
        assert self.z.ndim == 2, f"z must be 2D, got {self.z.ndim}D"
        assert self.z.shape == (len(self.y), len(self.x)), \
            f"z.shape {self.z.shape} must match (len(y)={len(self.y)}, len(x)={len(self.x)})"


@dataclass
class OptimizationVizData:
    """Complete visualization data for optimizer comparison.

    Following Casey: One struct to rule them all - keep related data together.
    """
    trajectory: TrajectoryData
    landscape: ContourData | None  # None for non-2D problems
    optimizer_name: str
    learning_rate: float

    def __post_init__(self):
        assert len(self.optimizer_name) > 0, "optimizer_name must be non-empty"
        assert self.learning_rate > 0.0, f"learning_rate must be positive, got {self.learning_rate}"


def extract_trajectory_data(
    losses: list[float],
    params_history: list[PyTree],
) -> TrajectoryData:
    """Extract trajectory data from optimizer output.

    Following Casey: Explicit conversion from JAX to numpy for portability.

    Args:
        losses: Loss values from optimization
        params_history: Parameter history from optimization

    Returns:
        TrajectoryData with numpy arrays
    """
    # Tiger Style: Assert preconditions
    assert len(losses) == len(params_history), \
        "losses and params_history must have same length"
    assert len(losses) > 0, "Must have at least one data point"

    # Convert JAX arrays to numpy for portability
    params_numpy = []
    for params in params_history:
        if isinstance(params, jnp.ndarray):
            params_numpy.append(np.array(params))
        else:
            params_numpy.append(params)  # Already numpy or other type

    steps = list(range(len(losses)))

    return TrajectoryData(
        losses=losses,
        params_history=params_numpy,
        steps=steps
    )


def compute_2d_landscape(
    loss_fn: Callable[[np.ndarray], float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: int = 100
) -> ContourData:
    """Compute 2D loss landscape for visualization.

    Following Casey: User controls granularity (resolution parameter).
    Following Tiger Style: Bounded computation (resolution has upper limit).

    Args:
        loss_fn: Function that takes 2D numpy array and returns scalar loss
        x_range: (min, max) for x-axis
        y_range: (min, max) for y-axis
        resolution: Number of points along each axis (default 100, max 1000)

    Returns:
        ContourData with loss landscape
    """
    # Tiger Style: Assert all preconditions
    assert len(x_range) == 2, "x_range must be (min, max)"
    assert len(y_range) == 2, "y_range must be (min, max)"
    assert x_range[0] < x_range[1], f"x_range must be increasing, got {x_range}"
    assert y_range[0] < y_range[1], f"y_range must be increasing, got {y_range}"
    assert 0 < resolution <= 1000, f"resolution must be in (0, 1000], got {resolution}"

    # Generate grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)

    # Compute loss at each grid point
    # Tiger Style: Explicit nested loops for clarity
    z = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            params = np.array([x[j], y[i]])
            z[i, j] = loss_fn(params)

    return ContourData(x=x, y=y, z=z)


def create_viz_data(
    losses: list[float],
    params_history: list[PyTree],
    loss_fn: Callable[[np.ndarray], float] | None,
    optimizer_name: str,
    learning_rate: float,
    landscape_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    landscape_resolution: int = 100
) -> OptimizationVizData:
    """Create complete visualization data from optimizer output.

    Following Casey: One-stop shop for creating viz data.
    User controls all parameters explicitly.

    Args:
        losses: Loss values from optimization
        params_history: Parameter history
        loss_fn: Loss function (needed for landscape, None if not 2D problem)
        optimizer_name: Name of optimizer for labeling
        learning_rate: Learning rate used
        landscape_bounds: ((x_min, x_max), (y_min, y_max)) or None to auto-detect
        landscape_resolution: Grid resolution for landscape

    Returns:
        Complete OptimizationVizData
    """
    # Extract trajectory
    trajectory = extract_trajectory_data(losses, params_history)

    # Compute landscape if requested and possible
    landscape = None
    if loss_fn is not None and len(params_history) > 0:
        first_params = trajectory.params_history[0]
        if isinstance(first_params, np.ndarray) and first_params.shape == (2,):
            # 2D problem - we can visualize landscape

            if landscape_bounds is None:
                # Auto-detect bounds from trajectory
                all_x = [p[0] for p in trajectory.params_history]
                all_y = [p[1] for p in trajectory.params_history]
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)

                # Add 20% margin
                x_margin = (x_max - x_min) * 0.2
                y_margin = (y_max - y_min) * 0.2
                x_range = (x_min - x_margin, x_max + x_margin)
                y_range = (y_min - y_margin, y_max + y_margin)
            else:
                x_range, y_range = landscape_bounds

            landscape = compute_2d_landscape(
                loss_fn=loss_fn,
                x_range=x_range,
                y_range=y_range,
                resolution=landscape_resolution
            )

    return OptimizationVizData(
        trajectory=trajectory,
        landscape=landscape,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate
    )


def main():
    """Test data generation."""
    # Simple test case
    def loss_fn(params: np.ndarray) -> float:
        return float(np.sum(params ** 2) / 2.0)

    # Create fake trajectory
    losses = [6.5, 5.2, 4.1, 3.2, 2.5, 2.0, 1.5, 1.0]
    params_history = [
        np.array([3.0, 2.0]),
        np.array([2.7, 1.8]),
        np.array([2.4, 1.6]),
        np.array([2.1, 1.4]),
        np.array([1.8, 1.2]),
        np.array([1.5, 1.0]),
        np.array([1.2, 0.8]),
        np.array([0.9, 0.6]),
    ]

    viz_data = create_viz_data(
        losses=losses,
        params_history=params_history,
        loss_fn=loss_fn,
        optimizer_name="SGD",
        learning_rate=0.1
    )

    print("=" * 80)
    print("Visualization Data Generation Test")
    print("=" * 80)
    print(f"\nOptimizer: {viz_data.optimizer_name}")
    print(f"Learning rate: {viz_data.learning_rate}")
    print(f"Trajectory points: {len(viz_data.trajectory.losses)}")
    print(f"Landscape available: {viz_data.landscape is not None}")

    if viz_data.landscape:
        print(f"Landscape resolution: {len(viz_data.landscape.x)} x {len(viz_data.landscape.y)}")
        print(f"Landscape bounds: x=[{viz_data.landscape.x[0]:.2f}, {viz_data.landscape.x[-1]:.2f}], "
              f"y=[{viz_data.landscape.y[0]:.2f}, {viz_data.landscape.y[-1]:.2f}]")

    print("\nTest passed!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
