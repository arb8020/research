"""Verification and benchmarking utilities for GPU kernels.

Provides tolerance-based matching and performance measurement,
following backend-bench patterns but with explicit error handling.
"""
from typing import Callable
import torch
import time
import traceback


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
            tb = traceback.format_exc()
            return False, f"Reference failed: {type(e).__name__}: {e}\n{tb}"

        try:
            # Run implementation (must not mutate test_input!)
            impl_output = impl_fn(test_input)
        except Exception as e:
            tb = traceback.format_exc()
            return False, f"Implementation failed: {type(e).__name__}: {e}\n{tb}"

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
    use_triton: bool = True,
) -> tuple[float, str | None]:
    """Benchmark kernel execution time.

    Args:
        kernel_fn: Kernel to benchmark
        test_input: Input data (will be called multiple times)
        num_warmup: Warmup iterations
        num_runs: Measurement iterations
        use_triton: Use Triton's do_bench for GPU (more accurate)

    Returns:
        (avg_time_ms, error_msg | None)
        error_msg is None on success
    """
    # Tiger Style: Assert arguments
    assert callable(kernel_fn), "kernel_fn must be callable"
    assert num_warmup > 0, f"num_warmup must be positive, got {num_warmup}"
    assert num_runs > 0, f"num_runs must be positive, got {num_runs}"

    try:
        # Try Triton benchmarking for GPU (more accurate)
        if use_triton and torch.cuda.is_available():
            try:
                import triton.testing
                # Triton's do_bench handles warmup and synchronization
                time_ms = triton.testing.do_bench(
                    lambda: kernel_fn(test_input),
                    warmup=num_warmup,
                    rep=num_runs,
                )
                return time_ms, None
            except ImportError:
                # Fall through to manual timing if Triton not available
                pass

        # Fallback: Manual timing (original implementation)
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
        tb = traceback.format_exc()
        return -1.0, f"Benchmark failed: {type(e).__name__}: {e}\n{tb}"


def benchmark_vs_reference(
    impl_fn: Callable,
    reference_fn: Callable,
    test_input,
    num_warmup: int = 10,
    num_runs: int = 100,
    use_triton: bool = True,
) -> tuple[float, float, float, str | None]:
    """Benchmark implementation against reference and compute speedup.

    Args:
        impl_fn: Implementation kernel to benchmark
        reference_fn: Reference kernel for comparison
        test_input: Test data
        num_warmup: Warmup iterations
        num_runs: Measurement iterations
        use_triton: Use Triton's do_bench if available

    Returns:
        (speedup, impl_time_ms, ref_time_ms, error_msg | None)
        speedup = ref_time / impl_time
        error_msg is None on success
    """
    try:
        # Benchmark reference
        ref_time, ref_err = benchmark_kernel(
            reference_fn, test_input, num_warmup, num_runs, use_triton
        )
        if ref_err:
            return 0.0, -1.0, -1.0, f"Reference benchmark failed: {ref_err}"

        # Benchmark implementation
        impl_time, impl_err = benchmark_kernel(
            impl_fn, test_input, num_warmup, num_runs, use_triton
        )
        if impl_err:
            return 0.0, impl_time, ref_time, f"Implementation benchmark failed: {impl_err}"

        # Calculate speedup
        speedup = ref_time / impl_time if impl_time > 0 else 0.0

        return speedup, impl_time, ref_time, None

    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, -1.0, -1.0, f"Speedup comparison failed: {type(e).__name__}: {e}\n{tb}"


def compare_backends(
    backend_names: list[str],
    test_input,
    reference_backend: str = "reference",
    num_warmup: int = 10,
    num_runs: int = 100,
    use_triton: bool = True,
) -> dict[str, tuple[float, float]]:
    """Compare performance of multiple backends against reference.

    Args:
        backend_names: List of backend names to compare
        test_input: Test data
        reference_backend: Name of reference backend for speedup calculation
        num_warmup: Warmup iterations
        num_runs: Measurement iterations
        use_triton: Use Triton's do_bench if available

    Returns:
        Dict mapping backend_name -> (speedup, time_ms)
        speedup is relative to reference_backend
    """
    from kernel_utils.backends import BACKENDS

    # Benchmark reference first
    ref_backend = BACKENDS[reference_backend]
    ref_time, ref_err = benchmark_kernel(
        ref_backend, test_input, num_warmup, num_runs, use_triton
    )
    if ref_err:
        raise RuntimeError(f"Reference '{reference_backend}' benchmark failed: {ref_err}")

    results = {}

    for name in backend_names:
        if name == reference_backend:
            # Reference has speedup of 1.0
            results[name] = (1.0, ref_time)
            continue

        backend = BACKENDS[name]
        time_ms, err = benchmark_kernel(
            backend, test_input, num_warmup, num_runs, use_triton
        )

        if err:
            results[name] = (0.0, -1.0)
        else:
            speedup = ref_time / time_ms if time_ms > 0 else 0.0
            results[name] = (speedup, time_ms)

    return results
