"""Verification and benchmarking utilities for GPU kernels.

Provides tolerance-based matching and performance measurement,
following backend-bench patterns but with explicit error handling.
"""
from typing import Callable
import torch
import time
import traceback
from pathlib import Path


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


def profile_kernel(
    kernel_fn: Callable,
    test_input,
    output_dir: Path,
    backend_name: str,
    test_name: str,
    num_warmup: int = 5,
    num_profile_runs: int = 1,
) -> tuple[str, str | None]:
    """Profile kernel execution using torch.profiler.

    Args:
        kernel_fn: Kernel to profile
        test_input: Input data
        output_dir: Directory to save profile traces
        backend_name: Name of backend being profiled
        test_name: Name of test case
        num_warmup: Warmup iterations before profiling
        num_profile_runs: Number of profiling runs to capture

    Returns:
        (trace_path, error_msg | None)
        trace_path is the path to the saved trace JSON file
        error_msg is None on success
    """
    assert callable(kernel_fn), "kernel_fn must be callable"
    assert num_warmup >= 0, f"num_warmup must be non-negative, got {num_warmup}"
    assert num_profile_runs > 0, f"num_profile_runs must be positive, got {num_profile_runs}"

    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Warmup
        for _ in range(num_warmup):
            _ = kernel_fn(test_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Profile with torch.profiler
        trace_filename = f"{backend_name}_{test_name}_profile"
        chrome_trace_path = str(output_dir / f"{trace_filename}.json")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(num_profile_runs):
                kernel_fn(test_input)
                if num_profile_runs > 1:
                    prof.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Export chrome trace (don't use tensorboard handler to avoid double-save)
        prof.export_chrome_trace(chrome_trace_path)

        return chrome_trace_path, None

    except Exception as e:
        tb = traceback.format_exc()
        return "", f"Profiling failed: {type(e).__name__}: {e}\n{tb}"


def ncu_profile_kernel(
    kernel_fn: Callable,
    test_input,
    output_dir: Path,
    backend_name: str,
    test_name: str,
    ncu_args: str = "--metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed",
) -> tuple[str, str | None]:
    """Profile kernel execution using NVIDIA Nsight Compute (ncu).

    Note: This function creates a Python script that will be profiled by NCU.
    NCU must be available in the system PATH.

    Args:
        kernel_fn: Kernel to profile
        test_input: Input data (tuple from TestCase.generate())
        output_dir: Directory to save NCU reports
        backend_name: Name of backend being profiled
        test_name: Name of test case
        ncu_args: Arguments to pass to ncu (default: basic metrics)

    Returns:
        (report_path, error_msg | None)
        report_path is the path to the saved .ncu-rep file
        error_msg is None on success
    """
    import subprocess
    import sys
    import json

    assert callable(kernel_fn), "kernel_fn must be callable"

    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a temporary script that runs the kernel
        script_path = output_dir / f"_ncu_temp_{backend_name}_{test_name}.py"
        params_path = output_dir / f"_ncu_temp_{backend_name}_{test_name}.json"
        report_filename = f"{backend_name}_{test_name}_ncu"
        report_path = output_dir / f"{report_filename}.ncu-rep"

        # Extract test parameters from SMOKE_TESTS by matching test_name
        # This avoids pickling torch tensors
        from kernel_utils.task import SMOKE_TESTS
        test_case = None
        for tc in SMOKE_TESTS:
            if tc.name == test_name:
                test_case = tc
                break

        if test_case is None:
            return "", f"Test case '{test_name}' not found in SMOKE_TESTS"

        # Save test parameters as JSON
        test_params = {
            "m": test_case.m,
            "k": test_case.k,
            "l": test_case.l,
            "seed": test_case.seed,
        }
        with open(params_path, 'w') as f:
            json.dump(test_params, f)

        # Get the project directory (parent of kernel_utils)
        project_dir = Path(__file__).parent.parent.absolute()

        # Create a standalone script that can be profiled by NCU
        script_content = f"""
import sys
import json
import torch
from pathlib import Path

# Add project directory to path so imports work
project_dir = Path('{project_dir}')
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# Load test parameters
with open('{params_path}', 'r') as f:
    params = json.load(f)

# Import kernel modules
import nvfp4.reference_kernel

# Generate test input
test_input = nvfp4.reference_kernel.generate_input(
    m=params['m'],
    k=params['k'],
    l=params['l'],
    seed=params['seed']
)

# Import BACKENDS and register backends explicitly (no auto-registration anymore)
from kernel_utils.backends import BACKENDS

BACKENDS.register(
    name="reference",
    kernel_fn=nvfp4.reference_kernel.ref_kernel,
    description="PyTorch reference using torch._scaled_mm",
    language="pytorch",
)

try:
    import nvfp4.optimized.triton_kernel
except ImportError:
    pass

try:
    import nvfp4.optimized.cute_kernel
except ImportError:
    pass

# Get the kernel function from registry
kernel_fn = BACKENDS['{backend_name}']

# Run the kernel once (NCU will profile this execution)
result = kernel_fn(test_input)

# Synchronize to ensure kernel completes
if torch.cuda.is_available():
    torch.cuda.synchronize()

print("Kernel execution completed")
"""

        # Write the script
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Build NCU command
        # Try common NCU paths if 'ncu' is not in PATH
        import shutil
        ncu_path = shutil.which("ncu")
        if ncu_path is None:
            # Check common CUDA installation paths
            for cuda_path in ["/usr/local/cuda/bin/ncu", "/usr/local/cuda-12/bin/ncu", "/opt/nvidia/nsight-compute/ncu"]:
                if Path(cuda_path).exists():
                    ncu_path = cuda_path
                    break

        if ncu_path is None:
            return "", "NCU not found - ensure NVIDIA Nsight Compute is installed and in PATH"

        # NCU needs sudo for GPU performance counters access
        # Check if we can use sudo
        use_sudo = False
        sudo_check = subprocess.run(["sudo", "-n", "true"], capture_output=True)
        if sudo_check.returncode == 0:
            use_sudo = True

        ncu_cmd = []
        if use_sudo:
            ncu_cmd.extend(["sudo", "-E"])  # -E preserves environment

        ncu_cmd.extend([
            ncu_path,
            "--export", str(report_path),
            "--force-overwrite",
        ])

        # Add custom metrics if provided
        if ncu_args:
            ncu_cmd.extend(ncu_args.split())

        # Add the Python command
        ncu_cmd.extend([
            sys.executable,
            str(script_path)
        ])

        # Run NCU
        result = subprocess.run(
            ncu_cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        # Clean up temporary files
        script_path.unlink(missing_ok=True)
        params_path.unlink(missing_ok=True)

        if result.returncode != 0:
            error_msg = f"NCU exit code {result.returncode}\n"
            if result.stdout:
                error_msg += f"stdout: {result.stdout}\n"
            if result.stderr:
                error_msg += f"stderr: {result.stderr}"
            return "", error_msg

        return str(report_path), None

    except subprocess.TimeoutExpired:
        return "", "NCU profiling timed out (>5 minutes)"
    except FileNotFoundError:
        return "", "NCU not found - ensure NVIDIA Nsight Compute is installed and in PATH"
    except Exception as e:
        tb = traceback.format_exc()
        return "", f"NCU profiling failed: {type(e).__name__}: {e}\n{tb}"
