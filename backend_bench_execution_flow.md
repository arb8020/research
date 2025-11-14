# Backend-Bench Test Execution Flow - Complete Analysis

## Overview
This document traces the complete execution path of correctness and performance tests in backend-bench, from code submission through test result generation.

---

## 1. What is `self.callable` in CodeEvaluator?

### Definition Location
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/code_evaluator.py`
**Lines:** 113-137

### Code Context
```python
class CodeEvaluator:
    """Class encapsulating the logic to evaluate code either locally or via Modal."""

    def __init__(self, cfg: BackendBenchConfig):
        self.cfg = cfg
        self._gpu = cfg.gpu

        self._optests = {
            extract_operator_name(str(op_test.op)): op_test for op_test in cfg._suite
        }

        if not self.is_modal:
            from src.utils import run_code
            
            self.callable = run_code  # <-- THIS IS THE KEY LINE
            
        else:
            import modal
            
            fn = modal.Function.from_name(
                "backend-bench-env-runner", f"eval_code_{self.cfg.gpu.lower()}"
            )
            
            self.callable = lambda **kwargs: fn.remote(**kwargs)
```

### What `self.callable` Is
- **For local execution** (gpu="local"): Points to the `run_code` function
- **For modal/cloud execution**: Points to a lambda that calls a remote Modal function
- Either way, it's a **synchronous function** that executes the test and returns results

---

## 2. The `run_code` Function - How It Works

### Definition Location
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/utils.py`
**Lines:** 124-140

### Complete Code
```python
def run_code(
    op_test: OpTest,
    code: str,
    op_name: str,
    file_name: str,
    expected_fn_name: str,
    module_name: str,
) -> tuple[float, float, CorrectnessTestResult, PerformanceTestResult]:
    kernel_fn = compile_kernel_from_string(
        code, op_name, file_name, expected_fn_name, module_name
    )
    return eval_one_op(
        op_test.op,
        kernel_fn,
        op_test.correctness_tests,
        op_test.performance_tests,
    )
```

### What It Does (3-Step Process)

**Step 1: Compile the kernel code**
- Calls `compile_kernel_from_string()` to convert user-submitted code string into an executable Python function
- The compiled function is named `{op_name}_kernel_impl`

**Step 2: Extract tests from OpTest**
- Gets `op_test.correctness_tests` - list of Test objects for correctness validation
- Gets `op_test.performance_tests` - list of Test objects for performance benchmarking

**Step 3: Run tests**
- Calls `eval_one_op()` which runs both correctness and performance tests
- Returns tuple: (correctness_score, performance_score, correctness_results, performance_results)

---

## 3. Code Compilation - `compile_kernel_from_string`

### Definition Location
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/utils.py`
**Lines:** 410-443

### Complete Code
```python
def compile_kernel_from_string(
    kernel_code: str, op_name: str, kernel_file_path: str, expected_fn_name: str, module_name: str
) -> tuple[Callable | None, list[str]]:
    def _find_kernel_function(module, folder_name: str) -> Callable:
        """Find the main kernel function in the compiled module."""
        expected_name = f"{folder_name}_kernel_impl"

        if hasattr(module, expected_name):
            return getattr(module, expected_name)

        available_functions = [
            name
            for name in dir(module)
            if callable(getattr(module, name)) and not name.startswith("_")
        ]

        raise ValueError(
            f"Expected function '{expected_name}' not found in kernel code for {op_name}. "
            f"Available functions: {available_functions}. "
            f"Please ensure the LLM generated code follows the naming convention: {folder_name}_kernel_impl"
        )

    try:
        save_kernel_to_file(kernel_code, kernel_file_path)

        spec = importlib.util.spec_from_file_location(module_name, kernel_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel_func = _find_kernel_function(module, expected_fn_name)
        return kernel_func

    except Exception as e:
        raise RuntimeError(f"Failed to compile kernel for {op_name}: {str(e)}") from e
```

### How Code Gets Loaded
1. **Save to temp file**: `save_kernel_to_file()` writes user's code to disk
2. **Create module spec**: Uses `importlib.util.spec_from_file_location()` to create a module spec
3. **Execute module**: `spec.loader.exec_module()` executes the Python code
4. **Find kernel function**: Looks for function named `{op_name}_kernel_impl` in the module
5. **Return callable**: Returns the compiled kernel function

---

## 4. How Test kwargs Get Populated - `_get_op_test_kwargs`

### Definition Location
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/code_evaluator.py`
**Lines:** 142-151

### Code
```python
def _get_op_test_kwargs(self, op_name: str) -> dict:
    if self.is_modal:
        return {
            "suite_name": self.cfg.suite,
            "ops": self.cfg.ops,
        }
    else:
        return {
            "op_test": self._optests[op_name],
        }
```

### How It Works
- **Local execution**: Returns the entire `OpTest` object for the operator
  - `self._optests[op_name]` looks up the OpTest from the suite
  - This OpTest contains both correctness_tests and performance_tests lists
  
- **Modal execution**: Returns just suite metadata (suite loads tests on remote)
  - Suite name and optional op filter are sent to cloud
  - Remote side loads tests using same mechanism

### OpTest Data Structure
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/suite/base.py`
**Lines:** 22-26

```python
class OpTest:
    def __init__(self, op, correctness_tests, performance_tests):
        self.op = op                                # The PyTorch operation object
        self.correctness_tests = correctness_tests  # List[Test] for correctness
        self.performance_tests = performance_tests  # List[Test] for performance
```

---

## 5. Where Test Cases Are Defined - Test Suite Data

### Suite Architecture
There are 4 test suites available:

1. **Smoke** - Minimal tests for quick checks
2. **OpInfo** - PyTorch's OpInfo tests
3. **TorchBench** - Real workload traces from HuggingFace
4. **Facto** - Factored operator tests

### TorchBench Suite Details (Most Common)
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/suite/torchbench.py`
**Lines:** 73-100

```python
class TorchBenchTestSuite:
    def __init__(
        self,
        name,
        filename=None,
        filter=None,
        topn=None,
        check_overhead_dominated_ops=False,
    ):
        self.name = name
        self.topn = topn
        # Load operations using the shared data loader
        ops_list = load_ops_from_source(
            source=filename,
            format="auto",  # Auto-detect based on file extension
            filter=filter,
        )
        if check_overhead_dominated_ops:
            # Only include ops which are overhead dominated
            ops_list = [op for op in ops_list if op.get("is_overhead_dominated_op", False)]

        # Convert to dictionary format using utility function
        self.optests = op_list_to_benchmark_dict(ops_list)

        # Deduplicate the strings in self.optests
        for op in self.optests:
            self.optests[op] = list(set(self.optests[op]))
```

### Where Tests Come From
- **Data Source**: HuggingFace dataset `GPUMODE/backendbench_tests`
- **Format**: Serialized argument strings from real PyTorch execution traces
- **Loading Function**: `load_ops_from_source()` loads from parquet/JSON files
- **Deserialization**: Arguments are deserialized using `deserialize_args()` when tests run

### Test Object Structure
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/suite/base.py`
**Lines:** 8-19

```python
class Test:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @property
    def args(self):
        return [arg() for arg in self._args]  # Lazy evaluation - callables invoked here

    @property
    def kwargs(self):
        return {k: v() for k, v in self._kwargs.items()}
```

**Key Point**: Tests use lazy evaluation via callables. This means test data (tensors, etc.) is created fresh each time the test runs, not cached.

---

## 6. The Complete Test Execution Flow - `eval_one_op`

### Definition Location
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/eval.py`
**Lines:** 227-271

### Complete Code
```python
def eval_one_op(
    op, impl, correctness_tests, performance_tests
) -> Tuple[float, float, List[CorrectnessTestResult], List[PerformanceTestResult]]:
    """Evaluate impl of op against correctness_tests and performance_tests.

    Returns:
        Tuple of (correctness_score, performance_score, correctness_results, performance_results)
    """

    if uses_cuda_stream(impl):
        logger.warning(f"Skipping {op.__name__} because it uses CUDA stream")
        performance_results = []
        correctness_results = []
        for test in correctness_tests:
            args_str = serialize_args(test.args, test.kwargs)
            correctness_results.append(
                CorrectnessTestResult(
                    op_name=op.__name__,
                    args=args_str,
                    is_correct=False,
                    error_msg="Skipped: uses CUDA stream",
                )
            )
        for test in performance_tests:
            args_str = serialize_args(test.args, test.kwargs)
            performance_results.append(
                PerformanceTestResult(
                    op_name=op.__name__,
                    args=args_str,
                    speedup=0,
                    benchmark_time_ms=0,
                    reference_time_ms=0,
                    error_msg="Skipped: uses CUDA stream",
                )
            )
        return 0, 1.0, correctness_results, performance_results

    correctness_score, correctness_results = eval_correctness(op, impl, correctness_tests)
    performance_score, performance_results = eval_performance(op, impl, performance_tests)
    return (
        correctness_score,
        performance_score,
        correctness_results,
        performance_results,
    )
```

### Execution Steps

#### Step 1: CUDA Stream Check
- Inspects if kernel uses CUDA streams (which cause race conditions in benchmarking)
- If found, skips tests and returns failure with "Skipped: uses CUDA stream" message

#### Step 2: Correctness Tests
- Calls `eval_correctness(op, impl, correctness_tests)` for each test
- Returns: (correctness_score: float, correctness_results: List[CorrectnessTestResult])

#### Step 3: Performance Tests
- Calls `eval_performance(op, impl, performance_tests)` for each test
- Returns: (performance_score: float, performance_results: List[PerformanceTestResult])

---

## 7. Correctness Test Execution - `eval_correctness`

### Definition Location
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/eval.py`
**Lines:** 128-146

### Code
```python
def eval_correctness(op, impl, tests) -> Tuple[float, List[CorrectnessTestResult]]:
    """Evaluate correctness of impl against tests."""
    correct, total = 0, 0
    test_results: List[CorrectnessTestResult] = []
    for test in tests:
        args_str = serialize_args(test.args, test.kwargs)
        logging.debug(f"Testing {op.__name__} with args {args_str}")
        result = eval_correctness_test(op, impl, test)
        test_results.append(result)
        if result.is_correct:
            correct += 1
        total += 1

    # Handle the case where no tests are available
    if total == 0:
        logger.warning(f"No correctness tests available for {str(op)}")
        return 0.0, []

    return correct / total, test_results
```

### Single Test Execution - `eval_correctness_test`
**Lines:** 93-125

```python
def eval_correctness_test(op, impl, test) -> CorrectnessTestResult:
    """Evaluate impl of op against test.

    Returns:
        Tuple of (is_correct, error_message, absolute_error, relative_error)
    """
    args, kwargs = test.args, test.kwargs
    ref = op(*args, **kwargs)  # Run reference PyTorch op
    try:
        res = impl(*args, **kwargs)  # Run user's kernel
        is_correct = allclose(ref, res)  # Check if outputs match

        abs_error, rel_error = compute_errors(ref, res)
        result = CorrectnessTestResult(
            op_name=op.__name__,
            args=serialize_args(args, kwargs),
            is_correct=is_correct,
            max_abs_error=abs_error,
            max_rel_error=rel_error,
        )
        return result
    except Exception as e:
        error_msg = format_exception(e, op, args, kwargs, traceback.format_exc())
        result = CorrectnessTestResult(
            op_name=op.__name__,
            args=serialize_args(args, kwargs),
            is_correct=False,
            error_msg=error_msg,
            error_type=str(type(e)),
            traceback=traceback.format_exc(),
        )
        logger.warning(error_msg)
        return result
```

### Correctness Test Process
1. **Get test inputs**: `args, kwargs = test.args, test.kwargs`
2. **Run reference**: `ref = op(*args, **kwargs)` - PyTorch's native operation
3. **Run implementation**: `res = impl(*args, **kwargs)` - User's kernel
4. **Compare outputs**: `allclose(ref, res)` with default tolerances (atol=1e-2, rtol=1e-2)
5. **Compute error metrics**: Absolute and relative errors
6. **Catch exceptions**: If user code throws exception, mark as incorrect with error message
7. **Score**: `correct / total` = fraction of tests passed

---

## 8. Performance Test Execution - `eval_performance`

### Definition Location
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/eval.py`
**Lines:** 162-224

### Code Overview
```python
def eval_performance(op, impl, tests) -> Tuple[float, List[PerformanceTestResult]]:
    """Evaluate performance of impl against tests."""
    bench_fn = (
        triton.testing.do_bench if TRITON_AVAILABLE and torch.cuda.is_available() else cpu_bench
    )
    base_times = []
    test_times = []
    args_strs = []
    performance_results: List[PerformanceTestResult] = []

    for test in tests:
        # Cache the arguments to ensure consistency between reference and implementation
        cached_args = test.args
        cached_kwargs = test.kwargs
        args_str = serialize_args(cached_args, cached_kwargs)
        args_strs.append(args_str)
        logging.debug(f"Benchmarking {op.__name__} with args {args_str}")
        
        # Benchmark the reference implementation
        base_time = bench_fn(lambda: op(*cached_args, **cached_kwargs))
        base_times.append(base_time)
        
        # Note: If the test fails we consider the speedup to be 1.0
        test_time = base_time
        try:
            ref = op(*cached_args, **cached_kwargs)
            res = impl(*cached_args, **cached_kwargs)
            if not allclose(ref, res):
                abs_error, rel_error = compute_errors(ref, res)
                raise ValueError(
                    f"Reference and result tensors are not close: max absolute error {abs_error}, max relative error {rel_error}"
                )
            test_time = bench_fn(lambda: impl(*cached_args, **cached_kwargs))
            performance_results.append(
                PerformanceTestResult(
                    op_name=op.__name__,
                    args=args_str,
                    speedup=base_time / test_time,
                    successfully_ran=True,
                    benchmark_time_ms=test_time,
                    reference_time_ms=base_time,
                )
            )
        except Exception as e:
            error_msg = format_exception(e, op, test.args, test.kwargs, traceback.format_exc())
            performance_results.append(
                PerformanceTestResult(
                    op_name=op.__name__,
                    args=args_str,
                    successfully_ran=False,
                    speedup=None,
                    benchmark_time_ms=None,
                    reference_time_ms=base_time,
                    error_msg=error_msg,
                )
            )
        finally:
            test_times.append(test_time)

    speedups = torch.tensor(base_times) / torch.tensor(test_times)

    return speedups.log().mean().exp(), performance_results
```

### Performance Test Process

**Step 1: Choose Benchmarking Function**
- If Triton available and CUDA: Use `triton.testing.do_bench()` (GPU benchmarking)
- Otherwise: Use `cpu_bench()` (CPU timing)

**Step 2: For Each Test Case**
1. Get cached args/kwargs (fresh instantiation via lazy evaluation)
2. Benchmark reference PyTorch op: `bench_fn(lambda: op(*cached_args, **cached_kwargs))`
3. Try to benchmark user kernel:
   - Verify output correctness first
   - If correct, benchmark: `bench_fn(lambda: impl(*cached_args, **cached_kwargs))`
   - If incorrect, record error
4. Calculate speedup: `base_time / test_time`

**Step 3: Aggregate Score**
- Compute speedup for each test case
- Return **geometric mean** of speedups: `speedups.log().mean().exp()`
- This is robust to outliers and makes sense for speedup metrics

### Benchmarking Details - `cpu_bench`
**Lines:** 149-159

```python
def cpu_bench(fn, num_runs=100):
    """Simple CPU benchmarking using time.perf_counter."""
    import time

    for _ in range(10):
        fn()  # Warmup

    start = time.perf_counter()
    for _ in range(num_runs):
        fn()
    return (time.perf_counter() - start) / num_runs
```

- 10 warmup iterations
- 100 measured iterations
- Returns average time per iteration

---

## 9. How CodeEvaluator Orchestrates Everything

### Full Async Call Flow
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/code_evaluator.py`
**Lines:** 153-221

### Code
```python
async def __call__(self, code: str, op_name: str) -> CodeEvaluationResult:
    """Evaluate the provided code for the given operator.

    Args:
        code (str): The code to evaluate.
        op_name (str): The name of the operator.

    Returns:
        CodeEvaluationResult: The result of the code evaluation.
    """
    op_test = cast(OpTest, self._optests[op_name])

    result = CodeEvaluationResult(code=code)
    correctness_score = 0.0
    performance_score = 0.0
    correctness_results: list[CorrectnessTestResult] = []
    performance_results: list[PerformanceTestResult] = []
    is_correct = False

    if not code:
        reason = "No code block found"
        correctness_results = construct_default_result(
            op_test, "correctness", reason=reason
        )
        performance_results = construct_default_result(
            op_test, "performance", reason=reason
        )

    else:
        try:
            file_name = f"{op_name}_kernel.py"

            (
                correctness_score,
                performance_score,
                correctness_results,
                performance_results,
            ) = await asyncio.to_thread(
                self.callable,
                **self._get_op_test_kwargs(op_name),
                code=code,
                op_name=op_name,
                file_name=file_name,
                expected_fn_name=op_name,
                module_name=f"module_{op_name}",
            )

            is_correct = all(r.is_correct for r in correctness_results) and all(
                r.successfully_ran for r in performance_results
            )

        except Exception as e:
            correctness_results = construct_default_result(
                op_test,
                "correctness",
                reason=str(e),
            )
            performance_results = construct_default_result(
                op_test,
                "performance",
                reason=str(e),
            )

    result.correctness_results = correctness_results
    result.performance_results = performance_results
    result.correctness_score = correctness_score
    result.performance_score = performance_score
    result.is_correct = is_correct
    return result.cleanup()
```

### Call Sequence

1. **Get OpTest**: Look up test object for the operator
2. **Check code**: If empty, return default failure results
3. **Run in thread**: Call `self.callable()` (run_code) in thread pool
   - Compile kernel
   - Run correctness tests
   - Run performance tests
   - Get results
4. **Determine correctness**: `all(correctness_correct) AND all(performance_ran_successfully)`
5. **Handle exceptions**: Any exception during test execution results in failure
6. **Cleanup**: Remove NaN values from scores
7. **Return CodeEvaluationResult**: Object with all scores and detailed results

---

## 10. Result Aggregation - CodeEvaluationResult

### Data Structure
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/code_evaluator.py`
**Lines:** 13-110

```python
@dataclass
class CodeEvaluationResult:
    """Contains the results of evaluating a piece of code."""

    code: str
    correctness_results: List[CorrectnessTestResult] = field(default_factory=list)
    performance_results: List[PerformanceTestResult] = field(default_factory=list)
    correctness_score: float = 0.0
    performance_score: float = 0.0
    is_correct: bool = False

    @property
    def score(self) -> float:
        return self.correctness_score * self.performance_score  # COMBINED SCORE

    @property
    def feedback(self) -> str:
        # Constructs detailed feedback based on test results
        # If correct: shows performance results and asks for optimization
        # If incorrect: shows which correctness/performance tests failed
        ...
```

### Key Score Calculation
- **Final Score** = `correctness_score * performance_score`
- Correctness score: 0.0 to 1.0 (fraction of tests passed)
- Performance score: 0.0 to infinity (geometric mean of speedups)
- Combined score heavily penalizes correctness failures

---

## 11. Key Differences Between Correctness and Performance Tests

| Aspect | Correctness | Performance |
|--------|-------------|-------------|
| **Purpose** | Verify output correctness | Measure execution speed |
| **Reference** | PyTorch's native op | Reference implementation |
| **Tolerance** | atol=1e-2, rtol=1e-2 | N/A (timings) |
| **Failure on** | Numerical mismatch or exception | Exception during exec |
| **Score Metric** | Fraction passed (0-1) | Geometric mean of speedups (0-inf) |
| **Scoring stops at** | First failure in aggregation | All tests still measured |
| **Feedback focus** | Shows first 2 failing tests | Shows speedup metrics |

### Critical Difference
- **Correctness is a gate**: Performance tests only run if correctness passes
  - Line 201: `is_correct = all(r.is_correct for r in correctness_results) and all(r.successfully_ran for r in performance_results)`
  - If ANY correctness test fails, the code is marked as incorrect regardless of performance
  - But performance tests still run (for debugging purposes)

---

## 12. The Complete Data Flow Diagram

```
User Code
   |
   v
CodeEvaluator.__call__(code, op_name)
   |
   +---> asyncio.to_thread(self.callable, ...)
   |       |
   |       v
   |    run_code(op_test, code, op_name, ...)
   |       |
   |       +---> compile_kernel_from_string()
   |       |       |
   |       |       v
   |       |    Save to file
   |       |       |
   |       |       v
   |       |    importlib.util.exec_module()
   |       |       |
   |       |       v
   |       |    Find "{op_name}_kernel_impl" function
   |       |       |
   |       |       v
   |       |    Return compiled function
   |       |
   |       +---> eval_one_op()
   |       |       |
   |       |       +---> eval_correctness()
   |       |       |       |
   |       |       |       For each test:
   |       |       |         - Run ref = op(*args, **kwargs)
   |       |       |         - Run res = impl(*args, **kwargs)
   |       |       |         - Compare with allclose()
   |       |       |         - Compute abs/rel errors
   |       |       |         - Create CorrectnessTestResult
   |       |       |
   |       |       +---> eval_performance()
   |       |               |
   |       |               For each test:
   |       |                 - Benchmark ref with triton.testing.do_bench()
   |       |                 - Verify correctness
   |       |                 - Benchmark impl
   |       |                 - Calculate speedup
   |       |                 - Create PerformanceTestResult
   |
   v
(correctness_score, performance_score, 
 correctness_results[], performance_results[])
   |
   v
CodeEvaluationResult
   |
   v
Feedback to user
```

---

## Summary

### The Core Execution Path

1. **Code arrives** → CodeEvaluator.__call__(code, op_name)
2. **Look up tests** → self._optests[op_name] gives OpTest with test lists
3. **Compile code** → Save to file, import as module, extract "{op_name}_kernel_impl"
4. **Run tests** → eval_one_op() with ref op, compiled impl, and test lists
5. **Test each case** → For each Test in lists, call op(*test.args, **test.kwargs)
6. **Compare outputs** → allclose() checks if outputs match within tolerance
7. **Benchmark** → triton.testing.do_bench() measures execution time
8. **Aggregate** → Correctness: fraction passed, Performance: geometric mean speedup
9. **Score** → Final = correctness_score * performance_score
10. **Feedback** → Generate human-readable feedback based on results

### Key Insight
- Tests are **data-driven** from HuggingFace dataset
- Tests use **lazy evaluation** (callables invoked at test time)
- Correctness is a **gate** (must pass to be considered correct)
- Performance is always **measured** (even if correctness fails)
- Scoring is **multiplicative** (correctness × performance)
