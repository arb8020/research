#!/usr/bin/env python3
"""Quick smoke test for NVFP4 kernel implementations.

Runs basic correctness checks locally before deploying to remote GPUs.
Usage:
    python -m kernel_utils.smoke_test
"""
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernel_utils.task import SMOKE_TESTS
from kernel_utils.utils import make_match_reference, benchmark_kernel
from nvfp4_reference_kernel import ref_kernel, generate_input


def run_smoke_tests() -> tuple[bool, str]:
    """Run smoke tests on reference kernel.

    Returns:
        (all_passed, summary_msg)
    """
    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        return False, "CUDA not available - smoke tests require GPU"

    print("🔥 Running NVFP4 Smoke Tests")
    print("=" * 60)

    checker = make_match_reference(ref_kernel, rtol=1e-3, atol=1e-3)

    passed_tests = 0
    failed_tests = 0

    for test in SMOKE_TESTS:
        print(f"\n📊 Test: {test.name}")
        print(f"   Dimensions: m={test.m}, k={test.k}, l={test.l}")

        # Generate test data
        test_input = generate_input(test.m, test.k, test.l, test.seed)

        # Verify correctness (reference against itself - sanity check)
        is_correct, error_msg = checker(ref_kernel, test_input)

        if is_correct:
            print(f"   ✅ Correctness: PASS")
            passed_tests += 1

            # Benchmark
            avg_time, bench_err = benchmark_kernel(
                ref_kernel, test_input, num_warmup=5, num_runs=20
            )
            if bench_err is None:
                print(f"   ⏱️  Performance: {avg_time:.3f}ms avg")
            else:
                print(f"   ⚠️  Benchmark failed: {bench_err}")
        else:
            print(f"   ❌ Correctness: FAIL")
            print(f"   Error: {error_msg}")
            failed_tests += 1

    print("\n" + "=" * 60)
    print(f"📈 Results: {passed_tests} passed, {failed_tests} failed")

    all_passed = (failed_tests == 0)
    summary = f"{passed_tests}/{len(SMOKE_TESTS)} tests passed"

    return all_passed, summary


def main() -> int:
    """Main entry point."""
    success, summary = run_smoke_tests()

    if success:
        print(f"\n✅ Smoke tests passed: {summary}")
        return 0
    else:
        print(f"\n❌ Smoke tests failed: {summary}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
