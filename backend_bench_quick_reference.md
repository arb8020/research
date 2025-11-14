# Backend-Bench Test Execution - Quick Reference Guide

## File Locations & Key Code Sections

### 1. CodeEvaluator Class
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/code_evaluator.py`

| Component | Lines | Purpose |
|-----------|-------|---------|
| Class definition | 113-141 | Main evaluator class, initializes `self.callable` |
| `__init__` | 116-136 | Sets up local or modal execution path |
| `_get_op_test_kwargs()` | 142-151 | Returns kwargs dict (OpTest object or suite metadata) |
| `__call__()` async method | 153-221 | Orchestrates entire test execution |

**Key Line:** 127 - `self.callable = run_code` (for local execution)

### 2. run_code Function
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/utils.py`

| Lines | Code |
|-------|------|
| 124-140 | Complete `run_code()` function - 2-step process (compile + eval) |
| 125-133 | Compiles kernel and gets test object |
| 135-140 | Calls `eval_one_op()` with tests |

### 3. Code Compilation
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/utils.py`

| Lines | Component |
|-------|-----------|
| 410-443 | `compile_kernel_from_string()` - Compiles user code |
| 413-430 | `_find_kernel_function()` - Looks for `{op_name}_kernel_impl` |
| 432-437 | Execution pipeline: save → import → find function |

**Critical:** Expects function named `{op_name}_kernel_impl` in user code

### 4. Test Definitions
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/suite/base.py`

| Lines | Class |
|-------|-------|
| 8-19 | `Test` class - Lazy evaluation via callables |
| 22-26 | `OpTest` class - Container for (op, correctness_tests, performance_tests) |
| 29-36 | `TestSuite` class - Iterable collection of OpTests |

### 5. TorchBench Suite (Data Source)
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/suite/torchbench.py`

| Lines | Component |
|-------|-----------|
| 39-70 | `TorchBenchOpTest` - Loads from serialized args |
| 61-70 | Properties that yield test objects |
| 73-100 | `TorchBenchTestSuite.__init__()` - Loads from HuggingFace |

**Data Source:** `https://huggingface.co/datasets/GPUMODE/backendbench_tests`

### 6. Test Execution Engine
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/eval.py`

| Lines | Function | Purpose |
|-------|----------|---------|
| 227-271 | `eval_one_op()` | Main orchestrator (CUDA check → correctness → performance) |
| 128-146 | `eval_correctness()` | Runs all correctness tests, returns fraction passed |
| 93-125 | `eval_correctness_test()` | Single test: compare outputs with tolerance |
| 162-224 | `eval_performance()` | Benchmarks and calculates speedups |
| 149-159 | `cpu_bench()` | CPU benchmarking: 10 warmup + 100 timed runs |

**Key Tolerance:** atol=1e-2, rtol=1e-2 in `allclose()`

### 7. Result Types
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/eval.py`

| Lines | Class | Fields |
|-------|-------|--------|
| 18-28 | `CorrectnessTestResult` | args, is_correct, error_msg, max_abs_error, max_rel_error |
| 31-40 | `PerformanceTestResult` | args, speedup, benchmark_time_ms, reference_time_ms, successfully_ran |

### 8. CodeEvaluationResult
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/code_evaluator.py`

| Lines | Component |
|-------|-----------|
| 13-110 | `CodeEvaluationResult` dataclass |
| 34-35 | `score` property: `correctness_score * performance_score` |
| 38-103 | `feedback` property: Generates human-readable feedback |

---

## Execution Flow - Step by Step

```
1. CodeEvaluator.__call__(code, op_name)  [line 153]
   └─ Get OpTest from self._optests[op_name]  [line 163]
   └─ asyncio.to_thread(self.callable, ...)  [line 190]
      
      2. run_code(op_test, code, ...)  [utils.py:124]
         └─ compile_kernel_from_string()  [BackendBench/utils.py:410]
            └─ Save code to file
            └─ importlib.util.exec_module()
            └─ Find {op_name}_kernel_impl function
         
         └─ eval_one_op(op, kernel_fn, correctness_tests, performance_tests)
            [BackendBench/eval.py:227]
            
            3. Check CUDA streams  [line 236]
            
            4. eval_correctness(op, impl, correctness_tests)  [line 264]
               For each test:
                 - Run ref = op(*args, **kwargs)
                 - Run res = impl(*args, **kwargs)
                 - Compare with allclose(atol=1e-2, rtol=1e-2)
                 - Record is_correct and errors
               Return: (correct/total, [CorrectnessTestResult])
            
            5. eval_performance(op, impl, performance_tests)  [line 265]
               For each test:
                 - Benchmark reference op
                 - Verify correctness
                 - Benchmark user kernel
                 - Calculate speedup = ref_time / user_time
               Return: (geometric_mean_speedup, [PerformanceTestResult])
   
   └─ Create CodeEvaluationResult  [line 165]
   └─ Populate with scores and results  [lines 216-221]
   └─ Return result.cleanup()  [line 221]
```

---

## Critical Code Sections

### How Tests Are Called
```python
# From eval.py:100-102 (correctness)
ref = op(*args, **kwargs)
res = impl(*args, **kwargs)
is_correct = allclose(ref, res)  # atol=1e-2, rtol=1e-2

# From eval.py:179-180 (performance)
base_time = bench_fn(lambda: op(*cached_args, **cached_kwargs))
test_time = bench_fn(lambda: impl(*cached_args, **cached_kwargs))
speedup = base_time / test_time
```

### Test Data Structure
```python
# From base.py:8-19
class Test:
    @property
    def args(self):
        return [arg() for arg in self._args]  # LAZY - callables invoked here
    
    @property
    def kwargs(self):
        return {k: v() for k, v in self._kwargs.items()}
```

### Score Calculation
```python
# From code_evaluator.py:34-35
@property
def score(self) -> float:
    return self.correctness_score * self.performance_score
    # 0-1 × 0-inf = multiplicative penalty for correctness failures
```

### Correctness Gate
```python
# From code_evaluator.py:200-202
is_correct = all(r.is_correct for r in correctness_results) and all(
    r.successfully_ran for r in performance_results
)
# MUST pass ALL correctness tests AND ALL perf tests must run
```

---

## Summary Table

| Aspect | Where | Details |
|--------|-------|---------|
| **Code compilation** | utils.py:410 | importlib → exec_module → find `{op_name}_kernel_impl` |
| **Test lookup** | code_evaluator.py:142 | Returns OpTest from self._optests dict |
| **Test structure** | base.py:8 | Lazy evaluation Test class with args/kwargs properties |
| **Test source** | torchbench.py:85 | load_ops_from_source() from HuggingFace parquet |
| **Correctness exec** | eval.py:100-102 | ref = op(*args); res = impl(*args); allclose() |
| **Performance exec** | eval.py:179-195 | Benchmark both, calculate speedup |
| **Scoring** | code_evaluator.py:34 | correctness_score × performance_score |
| **Error tolerance** | eval.py:68 | atol=1e-2, rtol=1e-2 |

---

## Key Insights

1. **`self.callable` points to `run_code`** - The function that handles compilation and test execution
2. **Tests are lazily evaluated** - Test data created fresh each time via callables
3. **Correctness is a gate** - Code marked correct only if ALL tests pass
4. **Performance uses geometric mean** - Speedups aggregated logarithmically
5. **Final score is multiplicative** - Correctness × Performance (strong penalty for incorrect code)
6. **Both test types always run** - Even if correctness fails, performance tests still execute
7. **Test data from HuggingFace** - Real-world PyTorch operation traces
8. **Compilation expects naming convention** - User code must define `{op_name}_kernel_impl` function
