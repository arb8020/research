# Kernel Utils - NVFP4 Testing Infrastructure

Utilities for testing and benchmarking NVFP4 block-scaled GEMV kernels.

## Structure

```
kernel_utils/
├── __init__.py      # Package exports
├── task.py          # Type definitions and test cases
├── utils.py         # Verification and benchmarking utilities
└── smoke_test.py    # Local smoke test runner
```

## Design Principles

Following code style guidelines:
- **Explicit error handling**: Functions return `(result, error)` tuples
- **Tiger Style assertions**: Heavy validation of preconditions and invariants
- **Single assignment**: Variables are immutable where possible
- **Simple control flow**: No complex abstractions or recursion

## Usage

### Local Smoke Test (requires CUDA)

```bash
cd dev/integration-evaluation
python -m kernel_utils.smoke_test
```

Expected output (with CUDA):
```
🔥 Running NVFP4 Smoke Tests
============================================================

📊 Test: tiny_single
   Dimensions: m=128, k=256, l=1
   ✅ Correctness: PASS
   ⏱️  Performance: 2.345ms avg

📊 Test: small_batch
   Dimensions: m=256, k=512, l=2
   ✅ Correctness: PASS
   ⏱️  Performance: 5.678ms avg

============================================================
📈 Results: 2 passed, 0 failed

✅ Smoke tests passed: 2/2 tests passed
```

### Remote Deployment (via Bifrost)

```bash
# Use GPU 0 (default)
python smoke_deploy.py --ssh root@host:port

# Use specific GPU
python smoke_deploy.py --ssh root@host:port --gpu 4

# Custom SSH key
python smoke_deploy.py --ssh root@host:port --gpu 4 --ssh-key ~/.ssh/custom_key
```

This will:
1. Deploy code to remote GPU instance
2. Setup PyTorch dependencies
3. Run smoke tests on specified GPU (via `CUDA_VISIBLE_DEVICES`)
4. Return results

## Test Suites

### `SMOKE_TESTS`
Quick sanity checks:
- `tiny_single`: m=128, k=256, l=1
- `small_batch`: m=256, k=512, l=2

### `CORRECTNESS_TESTS`
Comprehensive correctness validation:
- `min_size`: m=128, k=256, l=1
- `medium`: m=256, k=512, l=4
- `large`: m=512, k=1024, l=8
- `xlarge`: m=1024, k=2048, l=16

### `PERFORMANCE_TESTS`
Performance benchmarking:
- `perf_medium`: m=512, k=2048, l=32
- `perf_large`: m=1024, k=4096, l=64

## API Reference

### `task.py`

**Type Aliases:**
```python
input_t = tuple[
    torch.Tensor,  # a: [m, k, l] float4_e2m1fn_x2
    torch.Tensor,  # b: [1, k, l] float4_e2m1fn_x2
    torch.Tensor,  # scale_a: [m, k, l] float8_e4m3fn
    torch.Tensor,  # scale_b: [1, k, l] float8_e4m3fn
    torch.Tensor,  # scale_a_permuted
    torch.Tensor,  # scale_b_permuted
    torch.Tensor,  # c: [m, 1, l] float16
]

output_t = torch.Tensor  # [m, 1, l] float16
```

**TestCase:**
```python
@dataclass(frozen=True)
class TestCase:
    m: int      # Matrix rows (must be multiple of 128)
    k: int      # Matrix cols (must be multiple of 4)
    l: int      # Batch size
    seed: int   # Random seed
    name: str   # Test name
```

### `utils.py`

**allclose_with_error:**
```python
def allclose_with_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> tuple[bool, float, float]:
    """Returns: (is_match, max_abs_error, max_rel_error)"""
```

**make_match_reference:**
```python
def make_match_reference(
    reference_fn: Callable,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Callable:
    """Create checker: impl_fn -> (is_correct, error_msg | None)"""
```

**benchmark_kernel:**
```python
def benchmark_kernel(
    kernel_fn: Callable,
    test_input,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> tuple[float, str | None]:
    """Returns: (avg_time_ms, error_msg | None)"""
```

## Example: Testing Custom Kernel

```python
from kernel_utils import TestCase, make_match_reference, benchmark_kernel
from nvfp4_reference_kernel import ref_kernel, generate_input

# Create test case
test = TestCase(m=256, k=512, l=4, seed=42, name="custom_test")

# Generate test data
test_input = generate_input(test.m, test.k, test.l, test.seed)

# Create checker
checker = make_match_reference(ref_kernel, rtol=1e-3, atol=1e-3)

# Test your implementation
def my_kernel_impl(data):
    # Your implementation here
    return ref_kernel(data)  # placeholder

is_correct, error_msg = checker(my_kernel_impl, test_input)
if is_correct:
    print("✅ Correctness: PASS")

    # Benchmark
    avg_time, bench_err = benchmark_kernel(my_kernel_impl, test_input)
    if bench_err is None:
        print(f"⏱️  Performance: {avg_time:.3f}ms")
else:
    print(f"❌ Correctness: FAIL - {error_msg}")
```

## Integration with Backend-Bench

This infrastructure follows backend-bench patterns:
- Test data generators (lazy evaluation)
- Tolerance-based correctness checking
- Performance benchmarking with warmup
- Structured test result reporting

Ready to extend with full backend-bench integration for model evaluation.
