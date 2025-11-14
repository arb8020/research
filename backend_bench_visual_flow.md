# Backend-Bench Test Execution - Visual Flow Diagrams

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BackendBench Environment                     │
│  (loads test suite, initializes CodeEvaluator)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              CodeEvaluator.__call__(code, op_name)              │
│  - Async entry point                                             │
│  - Looks up OpTest from suite                                    │
│  - Calls self.callable via asyncio.to_thread                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
        LOCAL EXECUTION              MODAL EXECUTION
     (gpu="local")                  (gpu in T4/L4/A100...)
        self.callable =                    self.callable =
          run_code()                   modal.Function.remote()
```

## 2. Code Compilation Pipeline

```
User's Code String
      │
      ▼
┌─────────────────────────────────────────┐
│  compile_kernel_from_string()           │
│  (BackendBench/utils.py:410)            │
└──────────┬──────────────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │ Save to temp file   │
    │ (e.g., add_kernel.py)
    └──────────┬──────────┘
               │
               ▼
    ┌────────────────────────────────┐
    │ importlib.util.spec_from_file  │
    │ Create module spec             │
    └──────────┬─────────────────────┘
               │
               ▼
    ┌────────────────────────────────┐
    │ spec.loader.exec_module()      │
    │ Execute Python code            │
    └──────────┬─────────────────────┘
               │
               ▼
    ┌────────────────────────────────────────────┐
    │ _find_kernel_function(module, op_name)     │
    │ Look for {op_name}_kernel_impl             │
    │ Raise ValueError if not found              │
    └──────────┬───────────────────────────────┘
               │
               ▼
        ┌──────────────────┐
        │ Return Callable  │
        │ (compiled kernel)│
        └──────────────────┘
```

## 3. Test Data Flow

```
Test Suite Initialization
          │
          ├─── smoke.py (minimal tests)
          ├─── opinfo.py (PyTorch OpInfo tests)
          ├─── facto.py (factored operators)
          └─── torchbench.py (real traces)
                    │
                    ▼
          ┌──────────────────────┐
          │ TorchBenchTestSuite  │
          │ __init__()           │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────────────────┐
          │ load_ops_from_source()           │
          │ Load from HuggingFace parquet    │
          │ GPUMODE/backendbench_tests      │
          └──────────┬───────────────────────┘
                     │
                     ▼
          ┌──────────────────────────────────┐
          │ op_list_to_benchmark_dict()      │
          │ Convert to OpTest objects        │
          └──────────┬───────────────────────┘
                     │
                     ▼
          ┌──────────────────────────────────┐
          │ CodeEvaluator.__init__()         │
          │ Store in self._optests dict      │
          │ key = op_name                    │
          │ value = OpTest                   │
          └──────────────────────────────────┘
```

## 4. Test Structure - Lazy Evaluation

```
OpTest Object
│
├─ op: torch.ops.aten.add.Tensor (reference implementation)
│
├─ correctness_tests: [Test1, Test2, Test3, ...]
│  │
│  └─ Test1:
│     ├─ _args = (lambda: tensor(...), lambda: tensor(...))
│     ├─ _kwargs = {}
│     │
│     ├─ .args property → [tensor(...), tensor(...)]  ← LAZY!
│     └─ .kwargs property → {}
│
└─ performance_tests: [Test1, Test2, Test3, ...]
   │
   └─ Similar structure to correctness_tests

Key: Test data (tensors) created FRESH each time properties are accessed
     via callable evaluation. This ensures no state sharing between runs.
```

## 5. Correctness Test Execution

```
For Each Test in correctness_tests:
│
├─ 1. Extract test inputs
│  │  args, kwargs = test.args, test.kwargs
│  │
│  ▼
├─ 2. Run reference PyTorch op
│  │  ref = op(*args, **kwargs)
│  │  (This is torch.ops.aten.add.Tensor or similar)
│  │
│  ▼
├─ 3. Run user's compiled kernel
│  │  try:
│  │      res = impl(*args, **kwargs)
│  │      is_correct = allclose(ref, res, atol=1e-2, rtol=1e-2)
│  │  except Exception as e:
│  │      is_correct = False
│  │      error_msg = str(e)
│  │
│  ▼
├─ 4. Compute error metrics
│  │  abs_error = max(|ref - res|)
│  │  rel_error = max(|(ref - res)| / (|ref| + 1e-10))
│  │
│  ▼
└─ 5. Create CorrectnessTestResult
     CorrectnessTestResult(
         op_name=op.__name__,
         args=serialize_args(args, kwargs),
         is_correct=is_correct,
         max_abs_error=abs_error,
         max_rel_error=rel_error,
         error_msg=error_msg (if exception),
         error_type=type(e) (if exception),
     )

AGGREGATION:
correctness_score = (number of tests passed) / (total tests)
Range: 0.0 (all failed) to 1.0 (all passed)
```

## 6. Performance Test Execution

```
For Each Test in performance_tests:
│
├─ 1. Cache test inputs
│  │  cached_args = test.args
│  │  cached_kwargs = test.kwargs
│  │
│  ▼
├─ 2. Choose benchmarking function
│  │  if TRITON_AVAILABLE and torch.cuda.is_available():
│  │      bench_fn = triton.testing.do_bench()
│  │  else:
│  │      bench_fn = cpu_bench()
│  │
│  ▼
├─ 3. Benchmark reference PyTorch op
│  │  base_time = bench_fn(lambda: op(*cached_args, **cached_kwargs))
│  │  Returns: average time in seconds
│  │
│  ▼
├─ 4. Verify user kernel correctness
│  │  ref = op(*cached_args, **cached_kwargs)
│  │  res = impl(*cached_args, **cached_kwargs)
│  │  if not allclose(ref, res):
│  │      raise ValueError("Output mismatch")
│  │
│  ▼
├─ 5. Benchmark user kernel
│  │  test_time = bench_fn(lambda: impl(*cached_args, **cached_kwargs))
│  │
│  ▼
├─ 6. Calculate speedup
│  │  speedup = base_time / test_time
│  │  (>1.0 is faster, <1.0 is slower)
│  │
│  ▼
└─ 7. Create PerformanceTestResult
     PerformanceTestResult(
         op_name=op.__name__,
         args=serialize_args(args, kwargs),
         speedup=speedup,
         benchmark_time_ms=test_time,
         reference_time_ms=base_time,
         successfully_ran=True,
     )

AGGREGATION:
speedups = [speedup1, speedup2, speedup3, ...]
performance_score = exp(mean(log(speedups)))  ← GEOMETRIC MEAN
Range: 0.0 to infinity
Example: speedups=[1.2, 0.8, 1.5] → geometric_mean = 1.146

cpu_bench() Details:
  1. 10 warmup iterations (ignored)
  2. 100 timed iterations
  3. Return: (total_time / 100) = average time per call
```

## 7. Complete Test Execution Pipeline

```
┌─────────────────────────────────┐
│ CodeEvaluator.__call__(code, op)│
└────────────┬────────────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Get OpTest from     │
    │ self._optests[op]   │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Check if code empty │
    └──┬──────────────────┘
       │
       ├─ YES: Return default failure results
       │
       └─ NO: Continue
           │
           ▼
    ┌──────────────────────────────┐
    │ asyncio.to_thread(           │
    │   self.callable,             │
    │   op_test=OpTest,            │
    │   code=code,                 │
    │   ... other kwargs           │
    │ )                            │
    └────────────┬─────────────────┘
                 │
                 ▼
    ┌──────────────────────────────┐
    │ run_code(op_test, code, ...)│
    └────────┬─────────────────────┘
             │
             ├─ compile_kernel_from_string()
             │  └─ Returns compiled kernel function
             │
             └─ eval_one_op(op, kernel_fn, tests)
                │
                ├─ Check for CUDA streams
                │
                ├─ eval_correctness()
                │  └─ For each test:
                │     ├─ Run reference
                │     ├─ Run user kernel
                │     ├─ Compare outputs
                │     └─ Record CorrectnessTestResult
                │
                └─ eval_performance()
                   └─ For each test:
                      ├─ Benchmark reference
                      ├─ Verify correctness
                      ├─ Benchmark user kernel
                      └─ Record PerformanceTestResult
                │
                ▼
    Return (correctness_score, performance_score,
            correctness_results[], performance_results[])
    │
    ▼
    ┌──────────────────────────────────┐
    │ Create CodeEvaluationResult      │
    │ - Store all results              │
    │ - Calculate is_correct           │
    │ - Calculate score = corr * perf  │
    │ - Generate feedback              │
    └──────────┬───────────────────────┘
               │
               ▼
          RETURN TO USER
```

## 8. Correctness Gate Logic

```
┌──────────────────────────────────────────────────────┐
│  is_correct = ALL(correctness_tests_passed) AND      │
│              ALL(performance_tests_ran_successfully) │
└──────────────┬───────────────────────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
    FALSE        TRUE
      │            │
      ▼            ▼
    ┌─────────┐  ┌────────────────────┐
    │ Marked  │  │ Marked as correct  │
    │ WRONG   │  │ (perfect score if  │
    │ (0 pts) │  │  performance=1.0)  │
    └─────────┘  └────────────────────┘

KEY: Correctness is a GATE
     - Pass all correctness tests: gate opens
     - Fail even one: gate closes
     - Performance tests still run (for debugging)
     - But code is marked as WRONG
```

## 9. Score Calculation

```
┌─────────────────────────────────────────┐
│ correctness_score: 0.0 to 1.0           │
│ (fraction of correctness tests passed)  │
└────────────┬────────────────────────────┘
             │
             ▼ MULTIPLY
             │
┌────────────┴────────────────────────────┐
│ performance_score: 0.0 to infinity      │
│ (geometric mean of speedups)            │
└────────────┬────────────────────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Final Score =       │
    │ correctness × perf  │
    └─────────────────────┘

Examples:
- correctness=1.0, perf=1.2 → score = 1.2
- correctness=1.0, perf=0.8 → score = 0.8
- correctness=0.5, perf=2.0 → score = 1.0 (penalized)
- correctness=0.0, perf=∞   → score = 0.0 (gated)
```

## 10. File Organization

```
src/
├─ code_evaluator.py
│  ├─ CodeEvaluator class (lines 113-141)
│  ├─ __init__ (lines 116-136)
│  ├─ _get_op_test_kwargs (lines 142-151)
│  ├─ __call__ async (lines 153-221)
│  └─ CodeEvaluationResult dataclass (lines 13-110)
│
└─ utils.py
   ├─ run_code (lines 124-140)
   ├─ get_suite (lines 101-121)
   └─ construct_default_result (lines 67-98)

BackendBench/
├─ utils.py
│  ├─ compile_kernel_from_string (lines 410-443)
│  ├─ compute_errors (lines 185-242)
│  └─ deserialize_args (lines 164-182)
│
├─ eval.py
│  ├─ eval_one_op (lines 227-271)
│  ├─ eval_correctness (lines 128-146)
│  ├─ eval_correctness_test (lines 93-125)
│  ├─ eval_performance (lines 162-224)
│  ├─ cpu_bench (lines 149-159)
│  ├─ CorrectnessTestResult (lines 18-28)
│  └─ PerformanceTestResult (lines 31-40)
│
└─ suite/
   ├─ base.py
   │  ├─ Test (lines 8-19)
   │  ├─ OpTest (lines 22-26)
   │  └─ TestSuite (lines 29-36)
   │
   ├─ torchbench.py (lines 73-100)
   ├─ opinfo.py
   ├─ smoke.py
   └─ facto.py
```

---

## Key Takeaways

1. **Linear flow**: User code → Compile → Correctness tests → Performance tests → Score
2. **Lazy evaluation**: Test data created fresh via callables, no state reuse
3. **Correctness gate**: Single failure blocks entire evaluation
4. **Dual measurement**: Both correctness (pass/fail) and performance (speedup)
5. **Geometric mean**: Performance uses log-mean-exp for robust aggregation
6. **Multiplicative scoring**: Final score = correctness × performance
7. **Local or cloud**: Can run locally (gpu="local") or on cloud (Modal)
8. **Data-driven**: Tests loaded from HuggingFace, not hardcoded
