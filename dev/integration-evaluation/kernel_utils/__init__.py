"""Kernel utilities for NVFP4 testing.

Provides type definitions, verification utilities, and benchmarking
tools for GPU kernel testing.
"""

from kernel_utils.task import (
    input_t,
    output_t,
    TestCase,
    SMOKE_TESTS,
    CORRECTNESS_TESTS,
    PERFORMANCE_TESTS,
)

from kernel_utils.utils import (
    allclose_with_error,
    make_match_reference,
    benchmark_kernel,
)

__all__ = [
    # Type aliases
    "input_t",
    "output_t",
    # Test structures
    "TestCase",
    "SMOKE_TESTS",
    "CORRECTNESS_TESTS",
    "PERFORMANCE_TESTS",
    # Verification utilities
    "allclose_with_error",
    "make_match_reference",
    "benchmark_kernel",
]
