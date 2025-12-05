#!/usr/bin/env python3
"""Quick smoke test for NVFP4 kernel implementations.

Runs basic correctness and performance checks on all registered backends
before deploying to remote GPUs.

Usage:
    # Test all backends
    python -m kernel_utils.smoke_test

    # Test specific backends
    python -m kernel_utils.smoke_test reference triton cute
"""
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nvfp4.reference_kernel import ref_kernel

from kernel_utils.backends import BACKENDS
from kernel_utils.results import (
    BackendResults,
    CorrectnessResult,
    PerformanceResult,
    TestSuiteResults,
)
from kernel_utils.task import SMOKE_TESTS
from kernel_utils.utils import (
    benchmark_kernel,
    make_match_reference,
    ncu_profile_kernel,
    profile_kernel,
)

# Register reference backend explicitly
BACKENDS.register(
    name="reference",
    kernel_fn=ref_kernel,
    description="PyTorch reference using torch._scaled_mm",
    language="pytorch",
)

# Auto-register custom backends by importing them
# Add your kernel implementations here:
try:
    from nvfp4.optimized import triton_kernel  # Registers "triton" backend
except ImportError:
    pass  # Triton kernel not implemented yet

try:
    from nvfp4.optimized import cute_kernel  # Registers "cute" backend
except ImportError:
    pass  # CuTe kernel not implemented yet


def run_smoke_tests(
    backend_names: list[str] | None = None,
    reference_backend: str = "reference",
    save_results: bool = False,
    enable_profiling: bool = False,
    enable_ncu: bool = False,
) -> tuple[bool, str]:
    """Run smoke tests on specified backends.

    Args:
        backend_names: List of backend names to test, or None for all
        reference_backend: Backend to use as reference for correctness/speedup
        save_results: Whether to save results to JSON
        enable_profiling: Whether to profile kernels with torch.profiler
        enable_ncu: Whether to profile kernels with NVIDIA Nsight Compute

    Returns:
        (all_passed, summary_msg)
    """
    import torch

    if not torch.cuda.is_available():
        return False, "CUDA not available - smoke tests require GPU"

    # Default to all backends
    if backend_names is None:
        backend_names = BACKENDS.list()

    # Validate backends exist
    for name in backend_names:
        if name not in BACKENDS:
            available = BACKENDS.list()
            return False, f"Backend '{name}' not found. Available: {available}"

    print("🔥 Running NVFP4 Smoke Tests")
    print(f"   Test Suite: {len(SMOKE_TESTS)} tests")
    print(f"   Backends: {', '.join(backend_names)}")
    print(f"   Reference: {reference_backend}")
    if enable_profiling:
        print("   Torch Profiling: ENABLED")
    if enable_ncu:
        print("   NCU Profiling: ENABLED")
    print("=" * 80)

    # Reference for correctness checking
    ref_backend_fn = BACKENDS[reference_backend]
    checker = make_match_reference(ref_backend_fn, rtol=1e-3, atol=1e-3)

    # Collect results for all backends
    all_backend_results = []

    # Profile traces
    profile_traces = []
    ncu_reports = []

    for backend_name in backend_names:
        backend = BACKENDS[backend_name]
        print(f"\n🔧 Backend: {backend_name} ({backend.language})")
        print(f"   {backend.description}")
        print("-" * 80)

        correctness_results = []
        performance_results = []

        for test in SMOKE_TESTS:
            print(f"\n📊 Test: {test.name}")
            print(f"   Dimensions: m={test.m}, k={test.k}, l={test.l}, seed={test.seed}")

            # Generate fresh test data (lazy evaluation)
            test_input = test.generate()

            # Correctness check
            is_correct, error_msg = checker(backend, test_input)

            # Record correctness result
            correctness_results.append(
                CorrectnessResult(
                    test_name=test.name,
                    backend_name=backend_name,
                    is_correct=is_correct,
                    test_params=test.serialize(),
                    error_msg=error_msg if not is_correct else None,
                )
            )

            if is_correct:
                print("   ✅ Correctness: PASS")

                # Benchmark this backend
                test_input_bench = test.generate()  # Fresh data for benchmark
                avg_time, bench_err = benchmark_kernel(
                    backend, test_input_bench, num_warmup=5, num_runs=20
                )

                # Calculate speedup vs reference
                speedup = None
                ref_time = None
                if backend_name != reference_backend and bench_err is None:
                    # Get reference time
                    test_input_ref = test.generate()
                    ref_time, ref_err = benchmark_kernel(
                        ref_backend_fn, test_input_ref, num_warmup=5, num_runs=20
                    )
                    if ref_err is None:
                        speedup = ref_time / avg_time if avg_time > 0 else 0.0

                # Record performance result
                performance_results.append(
                    PerformanceResult(
                        test_name=test.name,
                        backend_name=backend_name,
                        successfully_ran=(bench_err is None),
                        test_params=test.serialize(),
                        avg_time_ms=avg_time if bench_err is None else -1.0,
                        speedup=speedup,
                        reference_time_ms=ref_time,
                        error_msg=bench_err,
                    )
                )

                if bench_err is None:
                    speedup_str = f" ({speedup:.2f}x)" if speedup else ""
                    print(f"   ⏱️  Performance: {avg_time:.3f}ms{speedup_str}")

                    # Profile with torch.profiler if enabled
                    if enable_profiling:
                        # Profile this backend
                        test_input_prof = test.generate()
                        profile_dir = Path("profiles")
                        trace_path, prof_err = profile_kernel(
                            backend, test_input_prof, profile_dir,
                            backend_name, test.name,
                            num_warmup=5, num_profile_runs=1
                        )
                        if prof_err is None:
                            print(f"   📊 Torch profile saved: {trace_path}")
                            profile_traces.append((backend_name, test.name, trace_path))
                        else:
                            print(f"   ⚠️  Torch profiling failed: {prof_err}")

                        # Also profile reference for comparison (if not already reference)
                        if backend_name != reference_backend:
                            test_input_ref_prof = test.generate()
                            ref_trace_path, ref_prof_err = profile_kernel(
                                ref_backend_fn, test_input_ref_prof, profile_dir,
                                reference_backend, test.name,
                                num_warmup=5, num_profile_runs=1
                            )
                            if ref_prof_err is None:
                                print(f"   📊 Reference torch profile saved: {ref_trace_path}")
                                profile_traces.append((reference_backend, test.name, ref_trace_path))
                            else:
                                print(f"   ⚠️  Reference torch profiling failed: {ref_prof_err}")

                    # Profile with NCU if enabled
                    if enable_ncu:
                        # Profile this backend
                        test_input_ncu = test.generate()
                        ncu_dir = Path("ncu_reports")
                        ncu_report_path, ncu_err = ncu_profile_kernel(
                            backend, test_input_ncu, ncu_dir,
                            backend_name, test.name
                        )
                        if ncu_err is None:
                            print(f"   📊 NCU report saved: {ncu_report_path}")
                            ncu_reports.append((backend_name, test.name, ncu_report_path))
                        else:
                            print(f"   ⚠️  NCU profiling failed: {ncu_err}")

                        # Also profile reference for comparison (if not already reference)
                        if backend_name != reference_backend:
                            test_input_ref_ncu = test.generate()
                            ref_ncu_path, ref_ncu_err = ncu_profile_kernel(
                                ref_backend_fn, test_input_ref_ncu, ncu_dir,
                                reference_backend, test.name
                            )
                            if ref_ncu_err is None:
                                print(f"   📊 Reference NCU report saved: {ref_ncu_path}")
                                ncu_reports.append((reference_backend, test.name, ref_ncu_path))
                            else:
                                print(f"   ⚠️  Reference NCU profiling failed: {ref_ncu_err}")
                else:
                    print(f"   ⚠️  Benchmark failed: {bench_err}")
            else:
                print("   ❌ Correctness: FAIL")
                if error_msg:
                    # Print first line of error
                    first_line = error_msg.split('\n')[0]
                    print(f"   Error: {first_line}")

                # Still record performance (as failed)
                performance_results.append(
                    PerformanceResult(
                        test_name=test.name,
                        backend_name=backend_name,
                        successfully_ran=False,
                        test_params=test.serialize(),
                        avg_time_ms=-1.0,
                        error_msg="Skipped due to correctness failure",
                    )
                )

        # Create backend results
        backend_results = BackendResults(
            backend_name=backend_name,
            correctness_tests=correctness_results,
            performance_tests=performance_results,
        )
        all_backend_results.append(backend_results)

        print(f"\n   {backend_results.summary()}")

    # Create complete test suite results
    suite_results = TestSuiteResults(
        suite_name="SMOKE_TESTS",
        backends=all_backend_results,
    )

    # Print comparison table
    print("\n" + "=" * 80)
    print(suite_results.summary_table())

    # Save results if requested
    if save_results:
        output_path = Path("results/smoke_test_results.json")
        suite_results.to_json(output_path)
        print(f"\n💾 Results saved to: {output_path}")

    # Print profile summary if profiling was enabled
    if enable_profiling and profile_traces:
        print("\n📊 Torch Profiling Summary:")
        print(f"   {len(profile_traces)} profile trace(s) generated")
        print("   Location: profiles/")
        print("\n   To view torch profiles:")
        print("   1. Chrome trace: Open chrome://tracing and load .json files")
        print("   2. TensorBoard: tensorboard --logdir=profiles/")

    # Print NCU summary if NCU profiling was enabled
    if enable_ncu and ncu_reports:
        print("\n📊 NCU Profiling Summary:")
        print(f"   {len(ncu_reports)} NCU report(s) generated")
        print("   Location: ncu_reports/")
        print("\n   To view NCU reports:")
        print("   CSV files in ncu_reports/")

    # Determine overall success
    all_passed = all(br.all_correct for br in all_backend_results)
    summary = f"{len(backend_names)} backends tested"

    return all_passed, summary


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run NVFP4 smoke tests on registered backends"
    )
    parser.add_argument(
        "backends",
        nargs="*",
        help="Backend names to test (default: all registered backends)",
    )
    parser.add_argument(
        "--reference",
        default="reference",
        help="Reference backend for correctness/speedup (default: reference)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available backends and exit",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch.profiler to generate detailed performance traces",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Enable NVIDIA Nsight Compute (ncu) profiling for kernel-level metrics",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        print("Available backends:")
        for name, backend in BACKENDS.items():
            print(f"  {name:<15} ({backend.language:<10}) - {backend.description}")
        return 0

    # Run tests
    backend_names = args.backends if args.backends else None

    try:
        success, summary = run_smoke_tests(
            backend_names=backend_names,
            reference_backend=args.reference,
            save_results=args.save,
            enable_profiling=args.profile,
            enable_ncu=args.ncu,
        )

        if success:
            print(f"\n✅ All smoke tests passed: {summary}")
            return 0
        else:
            print(f"\n❌ Some smoke tests failed: {summary}")
            return 1

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
