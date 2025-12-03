"""Kernel utilities for NVFP4 testing.

Provides type definitions, verification utilities, and benchmarking
tools for GPU kernel testing.
"""

from kernel_utils.backends import (
    BACKENDS,
    BackendInfo,
    BackendRegistry,
)
from kernel_utils.results import (
    BackendResults,
    CorrectnessResult,
    PerformanceResult,
    TestSuiteResults,
)
from kernel_utils.task import (
    CORRECTNESS_TESTS,
    PERFORMANCE_TESTS,
    SMOKE_TESTS,
    TestCase,
    input_t,
    output_t,
)
from kernel_utils.utils import (
    allclose_with_error,
    benchmark_kernel,
    benchmark_vs_reference,
    compare_backends,
    make_match_reference,
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
    "benchmark_vs_reference",
    "compare_backends",
    # Result structures
    "CorrectnessResult",
    "PerformanceResult",
    "BackendResults",
    "TestSuiteResults",
    # Backend registry
    "BACKENDS",
    "BackendRegistry",
    "BackendInfo",
]
