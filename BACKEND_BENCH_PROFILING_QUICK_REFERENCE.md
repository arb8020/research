# BackendBench Profiling Quick Reference Guide

## Quick Lookup: Profiling Files & Functions

### 1. Core Performance Benchmarking
**File:** `/BackendBench/eval.py`

```python
# Simple CPU benchmarking
from BackendBench.eval import cpu_bench

time_per_run = cpu_bench(lambda: kernel(*args, **kwargs), num_runs=100)
# Returns: float (seconds per run)

# Full performance evaluation
from BackendBench.eval import eval_performance, PerformanceTestResult

speedup, results = eval_performance(reference_op, custom_kernel, test_cases)
# speedup: float (geometric mean)
# results: List[PerformanceTestResult]
#   - speedup: float (reference_time / kernel_time)
#   - benchmark_time_ms: float
#   - reference_time_ms: float
#   - successfully_ran: bool
```

### 2. CUDA Utilities
**File:** `/BackendBench/utils.py`

```python
# GPU memory cleanup
from BackendBench.utils import cleanup_memory_and_gpu

cleanup_memory_and_gpu()
# Does: gc.collect(), torch.cuda.synchronize(), torch.cuda.empty_cache()

# Check if kernel uses CUDA streams
from BackendBench.utils import uses_cuda_stream

if uses_cuda_stream(kernel_func):
    print("Kernel uses CUDA streams - may be skipped in benchmarking")

# Compute accuracy metrics
from BackendBench.utils import compute_errors

abs_error, rel_error = compute_errors(reference, result, eps=1e-10)
# abs_error: max(|reference - result|)
# rel_error: max(|reference - result| / (|reference| + eps))
```

### 3. Multiprocessing Evaluation
**File:** `/BackendBench/multiprocessing_eval.py`

```python
from BackendBench.multiprocessing_eval import MultiprocessingEvaluator, EvalResult

with MultiprocessingEvaluator(num_workers=4) as evaluator:
    for op_test in suite:
        evaluator.submit_task(
            op_test.op,
            custom_impl,
            op_test.correctness_tests,
            op_test.performance_tests
        )
    
    evaluator.start_evaluation()
    results = evaluator.get_results()

for result in results:
    print(f"Task {result.task_id}:")
    print(f"  Correctness Score: {result.correctness_score}")
    print(f"  Performance Score: {result.performance_score}")
    print(f"  Performance Results: {result.performance_results}")
```

### 4. Kernel Performance Testing with LLM Backend
**File:** `/BackendBench/backends/llm.py`

```python
from BackendBench.backends.llm import LLMBackend, FeedbackInfo

backend = LLMBackend(model="claude-3-sonnet", llm_client=client)

# Test correctness
is_correct, feedback = backend.test_kernel_correctness(
    op=torch.ops.aten.add.Tensor,
    kernel_code=kernel_source_code,
    test_cases=correctness_tests
)

# Test performance (only if correct)
if is_correct:
    performance_score, perf_results = backend.test_kernel_performance(
        op=torch.ops.aten.add.Tensor,
        kernel_code=kernel_source_code,
        performance_tests=perf_tests
    )

# Get feedback
feedback_str = feedback.format_for_llm()
print(f"Overall Speedup: {feedback.overall_speedup}")
print(f"Correctness Score: {feedback.correctness_score}")
```

### 5. Results Analysis
**File:** `/BackendBench/output.py`

```python
from BackendBench.output import _prepare_results_data

all_results, failed_tests, op_summaries = _prepare_results_data(
    correctness_results,
    performance_results
)

for op_name, summary in op_summaries.items():
    print(f"{op_name}:")
    print(f"  Correctness Rate: {summary['correctness_rate']}")
    print(f"  Average Speedup: {summary['avg_speedup']}")
    print(f"  Geomean Speedup: {summary['geomean_speedup']}")
    print(f"  Max Abs Error: {summary['max_abs_error']}")
    print(f"  Max Rel Error: {summary['max_rel_error']}")
```

---

## Data Structures

### PerformanceTestResult
```python
@dataclass
class PerformanceTestResult:
    op_name: str                    # Operation name
    args: str                       # Serialized test arguments
    speedup: float                  # Reference time / kernel time
    benchmark_time_ms: float        # Kernel execution time (ms)
    reference_time_ms: float        # Reference op execution time (ms)
    error_msg: str = ""
    successfully_ran: bool = False
    test_type: str = "performance"
```

### CorrectnessTestResult
```python
@dataclass
class CorrectnessTestResult:
    op_name: str
    args: str
    is_correct: bool = False
    error_msg: str = ""
    error_type: str = ""
    traceback: str = ""
    max_abs_error: float = -math.inf
    max_rel_error: float = -math.inf
    test_type: str = "correctness"
```

### FeedbackInfo
```python
@dataclass
class FeedbackInfo:
    compilation_error: Optional[str]
    correctness_results: List[CorrectnessTestResult]
    performance_results: List[PerformanceTestResult]
    summary: str
    kernel_code: str
    
    @property
    def overall_speedup(self) -> float:
        """Geometric mean speedup"""
        speedups = [r.speedup for r in self.performance_results if r.successfully_ran]
        return torch.tensor(speedups).log().mean().exp().item()
    
    @property
    def correctness_score(self) -> float:
        """Fraction of passed tests"""
```

### EvalResult
```python
@dataclass
class EvalResult:
    task_id: int
    correctness_score: float        # 0.0 to 1.0
    performance_score: float        # Speedup ratio
    correctness_results: List[CorrectnessTestResult]
    performance_results: List[PerformanceTestResult]
    error: Optional[str]
```

---

## Profiling Workflow Examples

### Example 1: Simple Kernel Benchmarking
```python
import torch
from BackendBench.eval import eval_performance

# Define reference and custom kernels
def reference_kernel(x):
    return torch.matmul(x, x.t())

def custom_kernel(x):
    # Your optimized implementation
    return torch.matmul(x, x.t())

# Create test cases
test_cases = [...]

# Benchmark
speedup, results = eval_performance(
    reference_kernel,
    custom_kernel,
    test_cases
)

print(f"Geomean Speedup: {speedup:.2f}x")
for result in results:
    print(f"  Args: {result.args}")
    print(f"    Speedup: {result.speedup:.2f}x")
    print(f"    Time: {result.benchmark_time_ms:.3f}ms (ref: {result.reference_time_ms:.3f}ms)")
```

### Example 2: Multiprocess Evaluation
```python
from BackendBench.multiprocessing_eval import MultiprocessingEvaluator

# Evaluate multiple operations in parallel
with MultiprocessingEvaluator(num_workers=torch.cuda.device_count()) as evaluator:
    for op_test in test_suite:
        evaluator.submit_task(
            op=op_test.op,
            impl=backend[op_test.op],
            correctness_tests=op_test.correctness_tests,
            performance_tests=op_test.performance_tests
        )
    
    evaluator.start_evaluation()
    results = evaluator.get_results()
    
    # Analyze results
    for result in results:
        op_name = str(result.task_id)
        print(f"{op_name}: Speedup {result.performance_score:.2f}x")
```

### Example 3: Get Performance Feedback for LLM
```python
from BackendBench.backends.llm import LLMBackend, FeedbackInfo

backend = LLMBackend(model="claude", llm_client=client)

is_correct, feedback = backend.test_kernel_correctness(
    op=torch.ops.aten.relu.default,
    kernel_code=custom_code,
    test_cases=tests
)

if is_correct:
    _, perf_results = backend.test_kernel_performance(
        op=torch.ops.aten.relu.default,
        kernel_code=custom_code,
        performance_tests=perf_tests
    )
    feedback.performance_results = perf_results

# Generate LLM-readable feedback
feedback_for_llm = feedback.format_for_llm()
print(feedback_for_llm)
# Output includes:
#   - Compilation errors (if any)
#   - Correctness test failures
#   - Performance metrics (speedup, times)
#   - Suggestions for improvement
```

---

## Key Metrics & Calculations

### Speedup Calculation
```python
# Per test case
speedup = reference_time_ms / benchmark_time_ms

# Aggregate (geometric mean - better for multiplicative metrics)
geomean_speedup = exp(mean(log(speedups)))
# Or in PyTorch:
geomean_speedup = torch.tensor(speedups).log().mean().exp().item()

# Aggregate (arithmetic mean - not recommended for speedups)
avg_speedup = mean(speedups)
```

### Error Metrics
```python
# Absolute error
abs_error = max(|reference - result|)

# Relative error
rel_error = max(|reference - result| / (|reference| + eps))
```

### Scoring
```python
# Correctness score: percentage of passed tests
correctness_score = num_passed / total_tests

# Performance score: geometric mean speedup
performance_score = exp(mean(log(speedups)))

# Combined score
combined_score = correctness_score * performance_score
```

---

## Benchmarking Internals

### Benchmark Function Selection
```python
try:
    if torch.cuda.is_available():
        import triton.testing
        bench_fn = triton.testing.do_bench  # GPU benchmarking
    else:
        raise ImportError("Triton not available")
except ImportError:
    from BackendBench.eval import cpu_bench
    bench_fn = cpu_bench  # CPU fallback
```

### Benchmark Execution Pattern
```python
# Warm-up (10 runs)
for _ in range(10):
    fn()

# Measurement (100 runs)
start = time.perf_counter()
for _ in range(100):
    fn()
time_per_run = (time.perf_counter() - start) / 100
```

### Triton Benchmarking
```python
import triton.testing
time_ms = triton.testing.do_bench(
    lambda: kernel(*args, **kwargs),
    warmup=10,  # Warm-up iterations
    rep=100     # Measurement repetitions
)
# Returns time in milliseconds
```

---

## GPU Memory Management

### Standard Cleanup Pattern
```python
import gc
import torch

def cleanup():
    gc.collect()                    # Free Python objects
    torch.cuda.synchronize()        # Wait for GPU
    torch.cuda.empty_cache()        # Clear cache

# Use between test cases to avoid OOM
```

### Multiprocessing Memory Optimization
```python
# Problem: Serializing CUDA tensors causes OOM
# Solution: Convert to CPU, move to GPU in worker process

# Main process (before queueing)
test.args = move_to_device(test.args, torch.device("cpu"))

# Worker process
test.args = move_to_device(test.args, torch.device("cuda:worker_id"))
```

---

## Common Pitfalls & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| OOM errors in multiprocessing | CUDA tensor serialization | Convert to CPU before queueing |
| Unreliable benchmark times | No warm-up | Use bench_fn with warm-up iterations |
| Worker crashes silently | CUDA errors | Use MultiprocessingEvaluator for recovery |
| High variance in measurements | Cache effects | Use geometric mean, run multiple times |
| Skipped kernels | Uses CUDA streams | Check `uses_cuda_stream()`, modify kernel |

---

## Performance Profiling Checklist

- [ ] Choose benchmark function (Triton for GPU, time.perf_counter for CPU)
- [ ] Prepare test cases with diverse input sizes/shapes
- [ ] Verify kernel correctness before benchmarking
- [ ] Run warm-up iterations (10+) before measurement
- [ ] Run enough repetitions (100+) for stable averages
- [ ] Use geometric mean for speedup aggregation
- [ ] Clean up GPU memory between tests
- [ ] Handle exceptions gracefully
- [ ] Report both raw times and speedups
- [ ] Use multiprocessing for robustness

---

## File Locations Summary

```
BackendBench Package
├── eval.py                          # Core benchmarking functions
├── utils.py                         # CUDA utilities, error computation
├── multiprocessing_eval.py          # Parallel evaluation framework
├── backends/
│   ├── llm.py                       # LLM-driven kernel generation + profiling
│   └── base.py                      # Backend base class
├── output.py                        # Results aggregation & metrics
├── opregistry.py                    # Operation registry
└── suite/
    ├── base.py                      # Test suite base
    └── torchbench.py                # TorchBench suite
```

---

## References

- **Triton Benchmarking**: Uses `triton.testing.do_bench()` for GPU timing
- **CUDA Documentation**: Memory management via torch.cuda
- **Meta PyTorch**: GitHub: meta-pytorch/BackendBench
