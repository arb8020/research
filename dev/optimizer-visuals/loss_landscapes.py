#!/usr/bin/env python3
"""Different loss landscapes for testing optimizers.

Following Tiger Style: Simple, explicit functions.
Following Casey: User composes loss functions as needed.
"""

import jax.numpy as jnp
import numpy as np


def quadratic_loss(params: jnp.ndarray) -> jnp.ndarray:
    """Isotropic quadratic: f(x) = x^T x / 2

    Minimum at origin, same curvature in all directions.
    """
    return jnp.sum(params ** 2) / 2.0


def narrow_valley_loss(params: jnp.ndarray, x_scale: float = 10.0, y_scale: float = 1.0) -> jnp.ndarray:
    """Narrow valley: steep in x, shallow in y.

    f(x, y) = (x_scale * x)^2 / 2 + (y_scale * y)^2 / 2

    Args:
        params: [x, y] parameters
        x_scale: Curvature in x direction (larger = steeper)
        y_scale: Curvature in y direction (larger = steeper)

    Returns:
        Scalar loss
    """
    assert params.shape == (2,), "narrow_valley_loss requires 2D params"
    x, y = params[0], params[1]
    return ((x_scale * x) ** 2 + (y_scale * y) ** 2) / 2.0


def narrow_valley_loss_numpy(params: np.ndarray, x_scale: float = 10.0, y_scale: float = 1.0) -> float:
    """Numpy version for landscape visualization."""
    assert params.shape == (2,), "narrow_valley_loss requires 2D params"
    x, y = params[0], params[1]
    return float(((x_scale * x) ** 2 + (y_scale * y) ** 2) / 2.0)


def rosenbrock_loss(params: jnp.ndarray, a: float = 1.0, b: float = 100.0) -> jnp.ndarray:
    """Rosenbrock function: f(x, y) = (a - x)^2 + b(y - x^2)^2

    Classic optimization test function with narrow curved valley.
    Minimum at (a, a^2), typically (1, 1).

    Args:
        params: [x, y] parameters
        a: First parameter (default 1)
        b: Second parameter (default 100, controls valley narrowness)
    """
    assert params.shape == (2,), "rosenbrock_loss requires 2D params"
    x, y = params[0], params[1]
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def rosenbrock_loss_numpy(params: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """Numpy version for landscape visualization."""
    assert params.shape == (2,), "rosenbrock_loss requires 2D params"
    x, y = params[0], params[1]
    return float((a - x) ** 2 + b * (y - x ** 2) ** 2)
