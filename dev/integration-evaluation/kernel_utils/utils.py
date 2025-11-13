"""Verification and benchmarking utilities for GPU kernels.

Provides tolerance-based matching and performance measurement,
following backend-bench patterns but with explicit error handling.
"""
from typing import Callable
import torch
import time


def allclose_with_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> tuple[bool, float, float]:
    """Check if tensors match within tolerance and return error metrics.

    Args:
        actual: Computed result
        expected: Reference result
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        (is_match, max_abs_error, max_rel_error)
    """
    # Tiger Style: Assert preconditions
    assert actual.shape == expected.shape, \
        f"Shape mismatch: {actual.shape} vs {expected.shape}"
    assert rtol > 0, f"rtol must be positive, got {rtol}"
    assert atol > 0, f"atol must be positive, got {atol}"

    # Compute errors
    abs_diff = torch.abs(actual - expected)
    max_abs_error = float(abs_diff.max().item())

    # Avoid division by zero in relative error
    expected_abs = torch.abs(expected)
    rel_diff = abs_diff / torch.where(
        expected_abs > 1e-8,
        expected_abs,
        torch.ones_like(expected_abs)
    )
    max_rel_error = float(rel_diff.max().item())

    # Check tolerance
    is_match = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))

    return is_match, max_abs_error, max_rel_error


def make_match_reference(
    reference_fn: Callable,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Callable:
    """Create a checker function that compares implementation to reference.

    Args:
        reference_fn: Ground truth implementation
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Checker function: impl_fn -> (is_correct, error_msg | None)
    """
    # Tiger Style: Assert function arguments
    assert callable(reference_fn), "reference_fn must be callable"
    assert rtol > 0, f"rtol must be positive, got {rtol}"
    assert atol > 0, f"atol must be positive, got {atol}"

    def checker(impl_fn: Callable, test_input):
        """Check if implementation matches reference.

        Returns:
            (is_correct, error_msg | None)
            error_msg is None on success
        """
        assert callable(impl_fn), "impl_fn must be callable"

        try:
            # Run reference
            ref_output = reference_fn(test_input)
        except Exception as e:
            return False, f"Reference failed: {type(e).__name__}: {e}"

        try:
            # Run implementation (must not mutate test_input!)
            impl_output = impl_fn(test_input)
        except Exception as e:
            return False, f"Implementation failed: {type(e).__name__}: {e}"

        # Compare outputs
        is_match, max_abs_err, max_rel_err = allclose_with_error(
            impl_output, ref_output, rtol=rtol, atol=atol
        )

        if is_match:
            return True, None
        else:
            error_msg = (
                f"Output mismatch: "
                f"max_abs_error={max_abs_err:.6f} (tol={atol}), "
                f"max_rel_error={max_rel_err:.6f} (tol={rtol})"
            )
            return False, error_msg

    return checker


def benchmark_kernel(
    kernel_fn: Callable,
    test_input,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> tuple[float, str | None]:
    """Benchmark kernel execution time.

    Args:
        kernel_fn: Kernel to benchmark
        test_input: Input data (will be called multiple times)
        num_warmup: Warmup iterations
        num_runs: Measurement iterations

    Returns:
        (avg_time_ms, error_msg | None)
        error_msg is None on success
    """
    # Tiger Style: Assert arguments
    assert callable(kernel_fn), "kernel_fn must be callable"
    assert num_warmup > 0, f"num_warmup must be positive, got {num_warmup}"
    assert num_runs > 0, f"num_runs must be positive, got {num_runs}"

    try:
        # Warmup
        for _ in range(num_warmup):
            _ = kernel_fn(test_input)

        # Ensure CUDA synchronization if on GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = kernel_fn(test_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / num_runs) * 1000.0
        return avg_time_ms, None

    except Exception as e:
        return -1.0, f"Benchmark failed: {type(e).__name__}: {e}"
