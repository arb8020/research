# Backend-Bench Test Execution Analysis - Complete Documentation

This directory contains comprehensive analysis of how correctness and performance tests are executed in backend-bench.

## Files Included

### 1. **backend_bench_execution_flow.md** (Main Document - 26KB)
- Complete end-to-end explanation of the test execution mechanism
- 12 major sections covering all aspects:
  1. What is `self.callable` in CodeEvaluator?
  2. The `run_code` function and how it works
  3. Code compilation pipeline (`compile_kernel_from_string`)
  4. How test kwargs get populated (`_get_op_test_kwargs`)
  5. Where test cases are defined (test suites)
  6. Code loading and module importing
  7. Correctness test execution
  8. Performance test execution
  9. CodeEvaluator orchestration
  10. Result aggregation
  11. Differences between correctness and performance tests
  12. Complete data flow diagram

**Use this for:** Deep understanding of the entire flow with code snippets and detailed explanations.

### 2. **backend_bench_quick_reference.md** (Quick Lookup - 7.8KB)
- File locations with exact line numbers
- Function signatures and purposes
- Critical code sections with context
- Execution flow summary
- Summary table of key components
- File paths for every important piece

**Use this for:** Quick lookups when you need to find something specific.

### 3. **backend_bench_visual_flow.md** (Diagrams - 17KB)
- ASCII flow diagrams showing:
  - High-level architecture
  - Code compilation pipeline
  - Test data flow
  - Test structure with lazy evaluation
  - Correctness test execution
  - Performance test execution
  - Complete pipeline
  - Correctness gate logic
  - Score calculation
  - File organization tree

**Use this for:** Visual understanding of how components interact.

## Key Findings Summary

### The Core Mechanism

1. **`self.callable` = `run_code`** (line 127 of code_evaluator.py)
   - This is the function that actually executes code and runs tests
   - For local execution (gpu="local"), it points to `run_code` from utils.py
   - For cloud execution, it points to a Modal function

2. **`run_code` does 2 things:**
   - Compiles the kernel code string into an executable Python function
   - Runs both correctness and performance tests against the compiled code

3. **Test execution flow:**
   ```
   Code → Compile → Extract Tests → Run Correctness → Run Performance → Score
   ```

4. **Correctness tests:**
   - Compare user code output vs PyTorch reference output
   - Use tolerance: atol=1e-2, rtol=1e-2
   - Score = fraction of tests passed (0.0 to 1.0)
   - Act as a GATE: must pass ALL tests to be marked correct

5. **Performance tests:**
   - Benchmark user code vs PyTorch reference
   - Calculate speedup = reference_time / user_time
   - Score = geometric mean of speedups (0.0 to infinity)
   - Always runs, even if correctness fails (for debugging)

6. **Final score:**
   - Calculated as: `correctness_score × performance_score`
   - Strong multiplicative penalty if correctness fails
   - If correctness = 0, final score = 0 (correctness is a gate)

## Critical File Paths

All file paths are in `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/`:

| Component | File | Key Lines |
|-----------|------|-----------|
| CodeEvaluator | src/code_evaluator.py | 113-221 |
| run_code | src/utils.py | 124-140 |
| compile_kernel_from_string | BackendBench/utils.py | 410-443 |
| eval_one_op | BackendBench/eval.py | 227-271 |
| eval_correctness | BackendBench/eval.py | 128-146 |
| eval_performance | BackendBench/eval.py | 162-224 |
| Test class | BackendBench/suite/base.py | 8-19 |
| OpTest class | BackendBench/suite/base.py | 22-26 |
| TorchBenchTestSuite | BackendBench/suite/torchbench.py | 73-100 |

## Important Details

### Code Compilation
- User code is saved to a temporary file
- Loaded as a Python module using `importlib.util`
- Module is executed: `spec.loader.exec_module()`
- Function named `{op_name}_kernel_impl` is extracted
- **Critical:** If this function doesn't exist, compilation fails

### Test Data Structure
Tests use **lazy evaluation** via callables:
```python
class Test:
    @property
    def args(self):
        return [arg() for arg in self._args]  # Callables invoked here
```
This ensures test data (tensors, etc.) is created fresh each time.

### Test Source
- Data from HuggingFace: `https://huggingface.co/datasets/GPUMODE/backendbench_tests`
- Real-world PyTorch operation traces
- Serialized as argument strings
- Deserialized on-demand when tests run

### Benchmarking
- GPU: Uses `triton.testing.do_bench()` if available
- CPU: Uses `cpu_bench()` with 10 warmup + 100 timed iterations
- Returns average time per call
- Speedup = reference_time / test_time

## How to Use These Documents

1. **Start with:** `backend_bench_visual_flow.md` (understand the big picture)
2. **Deep dive:** `backend_bench_execution_flow.md` (understand each component)
3. **Reference:** `backend_bench_quick_reference.md` (find specific code)

## Example Trace

When code arrives for operator `add.Tensor`:

1. CodeEvaluator.__call__(code="def add__Tensor_kernel_impl(...)")
2. Lookup OpTest from self._optests["add.Tensor"]
3. run_code(op_test, code, ...) in thread
4. compile_kernel_from_string() → finds "add__Tensor_kernel_impl" function
5. eval_one_op(torch.ops.aten.add.Tensor, compiled_fn, tests)
6. For each test in correctness_tests:
   - ref = torch.ops.aten.add.Tensor(*args, **kwargs)
   - res = compiled_fn(*args, **kwargs)
   - Check: allclose(ref, res, atol=1e-2, rtol=1e-2)
7. For each test in performance_tests:
   - Benchmark ref and res
   - Calculate speedup = ref_time / res_time
8. Return CodeEvaluationResult with all scores and results

---

**Document Generated:** 2025-11-12
**Tool:** Claude Code Analysis
