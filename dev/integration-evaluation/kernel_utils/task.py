"""Type definitions for NVFP4 kernel testing.

Defines the input/output tensor types and test case structure
for NVFP4 block-scaled GEMV kernels.
"""
from dataclasses import dataclass
from typing import TypeAlias
import torch

# Type aliases for kernel I/O
# Input: (a, b, scale_a, scale_b, scale_a_permuted, scale_b_permuted, c)
input_t: TypeAlias = tuple[
    torch.Tensor,  # a: [m, k, l] float4_e2m1fn_x2
    torch.Tensor,  # b: [1, k, l] float4_e2m1fn_x2
    torch.Tensor,  # scale_a: [m, k, l] float8_e4m3fn (CPU)
    torch.Tensor,  # scale_b: [1, k, l] float8_e4m3fn (CPU)
    torch.Tensor,  # scale_a_permuted: GPU version
    torch.Tensor,  # scale_b_permuted: GPU version
    torch.Tensor,  # c: [m, 1, l] float16 (output buffer)
]

# Output: Modified c tensor
output_t: TypeAlias = torch.Tensor  # [m, 1, l] float16


@dataclass(frozen=True)
class TestCase:
    """Single test case for kernel verification.

    Immutable to follow single-assignment principle.
    """
    m: int          # Matrix rows
    k: int          # Matrix cols / vector length
    l: int          # Batch size
    seed: int       # Random seed for reproducibility
    name: str       # Human-readable test name

    def __post_init__(self):
        """Validate test case parameters."""
        # Tiger Style: Assert preconditions
        assert self.m > 0, f"m must be positive, got {self.m}"
        assert self.k > 0, f"k must be positive, got {self.k}"
        assert self.l > 0, f"l must be positive, got {self.l}"
        assert self.m % 128 == 0, f"m must be multiple of 128, got {self.m}"
        assert self.k % 4 == 0, f"k must be multiple of 4, got {self.k}"
        assert self.seed >= 0, f"seed must be non-negative, got {self.seed}"
        assert len(self.name) > 0, "name cannot be empty"


# Predefined test suites
SMOKE_TESTS: list[TestCase] = [
    TestCase(m=128, k=256, l=1, seed=42, name="tiny_single"),
    TestCase(m=256, k=512, l=2, seed=43, name="small_batch"),
]

CORRECTNESS_TESTS: list[TestCase] = [
    TestCase(m=128, k=256, l=1, seed=100, name="min_size"),
    TestCase(m=256, k=512, l=4, seed=101, name="medium"),
    TestCase(m=512, k=1024, l=8, seed=102, name="large"),
    TestCase(m=1024, k=2048, l=16, seed=103, name="xlarge"),
]

PERFORMANCE_TESTS: list[TestCase] = [
    TestCase(m=512, k=2048, l=32, seed=200, name="perf_medium"),
    TestCase(m=1024, k=4096, l=64, seed=201, name="perf_large"),
]
