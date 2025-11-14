# Backend-Bench GPU Execution Analysis

## Overview
Backend-bench supports two GPU execution modes:
1. **Local GPU** (`gpu="local"`): Direct execution on the local machine's GPU
2. **Remote GPU** (`gpu="T4"|"L4"|"A100"|"H100"|"H200"|"B200"`): Execution on cloud GPU via Modal

---

## 1. GPU Configuration

### Configuration File Location
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/config.py`

### BackendBenchConfig Class (Lines 8-42)
```python
@dataclass
class BackendBenchConfig:
    """Configuration class for BackendBench.
    
    Attributes:
        gpu (str): Which GPU to use. Options are `local` (uses local GPU for debugging - 
                   results aren't correct as no scheduling is in place). If option from 
                   `T4`, `L4`, `A100`, `H100`, `H200` or `B200`, uses Modal to run 
                   evaluation on that GPU type.
    """
    
    suite: TSuite = "smoke"
    ops: list[str] | None = None
    gpu: str = "T4"  # Default: T4 GPU on Modal
    num_turns: int = 1
    feedback_loop: TFeedbackLoop = "until_max_turns"
```

### GPU Parameter Options
| GPU Value | Execution Mode | Details |
|-----------|---|---|
| `"local"` | Local execution | Uses local machine's GPU (no scheduling) |
| `"T4"` | Remote via Modal | NVIDIA Tesla T4 |
| `"L4"` | Remote via Modal | NVIDIA L4 |
| `"A100"` | Remote via Modal | NVIDIA A100 (80GB) |
| `"H100"` | Remote via Modal | NVIDIA H100 |
| `"H200"` | Remote via Modal | NVIDIA H200 |
| `"B200"` | Remote via Modal | NVIDIA B200 |

---

## 2. Local vs Remote Execution Path

### CodeEvaluator Initialization
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/code_evaluator.py`  
**Lines:** 113-140

```python
class CodeEvaluator:
    """Class encapsulating the logic to evaluate code either locally or via Modal."""

    def __init__(self, cfg: BackendBenchConfig):
        self.cfg = cfg
        self._gpu = cfg.gpu

        self._optests = {
            extract_operator_name(str(op_test.op)): op_test for op_test in cfg._suite
        }

        # DECISION POINT: Local vs Remote
        if not self.is_modal:
            # LOCAL EXECUTION
            from src.utils import run_code
            self.callable = run_code  # Direct function reference
        else:
            # REMOTE EXECUTION
            import modal
            fn = modal.Function.from_name(
                "backend-bench-env-runner", f"eval_code_{self.cfg.gpu.lower()}"
            )
            self.callable = lambda **kwargs: fn.remote(**kwargs)  # Remote call

    @property
    def is_modal(self) -> bool:
        return self._gpu != "local"
```

### Execution Flow Determination
- **`is_modal` property** determines execution mode
- **Local**: `gpu == "local"` → `self.callable = run_code`
- **Remote**: `gpu != "local"` → `self.callable = lambda: fn.remote(**kwargs)`

---

## 3. Local GPU Execution

### The `run_code` Function
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/utils.py`  
**Lines:** 124-140

```python
def run_code(
    op_test: OpTest,
    code: str,
    op_name: str,
    file_name: str,
    expected_fn_name: str,
    module_name: str,
) -> tuple[float, float, CorrectnessTestResult, PerformanceTestResult]:
    # Step 1: Compile kernel code
    kernel_fn = compile_kernel_from_string(
        code, op_name, file_name, expected_fn_name, module_name
    )
    
    # Step 2: Evaluate compiled kernel on GPU
    return eval_one_op(
        op_test.op,
        kernel_fn,
        op_test.correctness_tests,
        op_test.performance_tests,
    )
```

### Execution in CodeEvaluator
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/code_evaluator.py`  
**Lines:** 185-198

```python
(
    correctness_score,
    performance_score,
    correctness_results,
    performance_results,
) = await asyncio.to_thread(
    self.callable,  # For local: this is run_code
    **self._get_op_test_kwargs(op_name),  # Passes: op_test
    code=code,
    op_name=op_name,
    file_name=file_name,
    expected_fn_name=op_name,
    module_name=f"module_{op_name}",
)
```

### Key Point for Local Execution
- Runs in a **thread pool** (`asyncio.to_thread`)
- Blocks async execution while GPU work happens
- No scheduling or resource management
- Test data is on GPU (see section 4)

---

## 4. Remote GPU Execution via Modal

### Modal Runner Setup
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/modal_runner.py`  
**Lines:** 1-104

```python
import torch
from modal import Image, App

app = App("backend-bench-env-runner")
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# GPU Docker Image with CUDA 12.8
cuda_image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git", "gcc-13", "g++-13", "clang-18")
    .pip_install(
        "ninja~=1.11", "wheel~=0.45", "requests~=2.32.4",
        "packaging~=25.0", "numpy~=2.3", "pytest", "PyYAML",
    )
    .pip_install(
        "torch>=2.7.0,<2.8.0", "torchvision~=0.22", "torchaudio>=2.7.0,<2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("git+https://github.com/meta-pytorch/BackendBench.git@main")
    .pip_install(
        "nvidia-cupynumeric~=25.3",
        "nvidia-cutlass-dsl~=4.0",
        "cuda-core[cu12]~=0.3",
        "cuda-python[all]==12.8",
    )
    .add_local_python_source("modal_runner")
)

# Remote eval_code Function
def eval_code(
    suite_name: str,
    ops: list[str] | None,
    code: str,
    op_name: str,
    file_name: str,
    expected_fn_name: str,
    module_name: str,
) -> tuple[float, float, CorrectnessTestResult, PerformanceTestResult]:
    # Same logic as local, but runs on remote GPU
    suite = get_suite(suite_name, ops)
    op_tests = {extract_operator_name(str(op_test.op)): op_test for op_test in suite}
    op_test = op_tests[op_name]
    
    kernel_fn = compile_kernel_from_string(
        code, op_name, file_name, expected_fn_name, module_name
    )
    return eval_one_op(
        op_test.op,
        kernel_fn,
        op_test.correctness_tests,
        op_test.performance_tests,
    )

# Register function for each GPU type
for gpu in {"T4", "L4", "A100-80GB", "H100!", "H200", "B200"}:
    gpu_slug = gpu.lower().split("-")[0].strip("!").replace(":", "x")
    app.function(
        gpu=gpu, 
        image=cuda_image, 
        name=f"eval_code_{gpu_slug}", 
        serialized=True
    )(eval_code)
```

### Remote Call from CodeEvaluator
```python
# When gpu != "local":
import modal
fn = modal.Function.from_name(
    "backend-bench-env-runner", f"eval_code_{self.cfg.gpu.lower()}"
)
self.callable = lambda **kwargs: fn.remote(**kwargs)
```

### Key Differences: Local vs Remote
| Aspect | Local (`gpu="local"`) | Remote (`gpu="T4"` etc) |
|--------|---|---|
| **Function** | `run_code` from `utils.py` | `eval_code` from `modal_runner.py` |
| **Location** | Local machine | Modal cloud GPU |
| **Execution** | Direct function call via thread pool | `fn.remote()` call |
| **Test kwargs** | `op_test` object | `suite_name`, `ops` metadata |
| **Docker image** | None (uses local env) | NVIDIA CUDA 12.8 container |
| **torch.cuda** | Local GPU | Modal-allocated GPU |
| **Scheduling** | None | Modal handles allocation |

---

## 5. GPU-Specific Code in Execution Path

### Benchmarking Function Selection
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/eval.py`  
**Lines:** 43-51, 162-166

```python
# Check if CUDA available at import time
try:
    if torch.cuda.is_available():
        import triton.testing
        TRITON_AVAILABLE = True
    else:
        TRITON_AVAILABLE = False
except ImportError:
    TRITON_AVAILABLE = False

# At performance test time
def eval_performance(op, impl, tests) -> Tuple[float, List[PerformanceTestResult]]:
    """Evaluate performance of impl against tests."""
    
    # CRITICAL GPU DECISION:
    bench_fn = (
        triton.testing.do_bench 
        if TRITON_AVAILABLE and torch.cuda.is_available() 
        else cpu_bench  # Fallback to CPU if no CUDA
    )
```

### torch.cuda Usage Points

1. **GPU Availability Check**
   ```python
   torch.cuda.is_available()  # Returns True if GPU available
   ```

2. **GPU Memory Management** (in utils.py lines 245-249)
   ```python
   def cleanup_memory_and_gpu():
       """Helper function to clean up GPU memory"""
       gc.collect()
       torch.cuda.synchronize()  # Synchronize GPU operations
       torch.cuda.empty_cache()  # Clear GPU memory cache
   ```

3. **Tensor Device Placement** (in utils.py lines 100-113)
   ```python
   def _deserialize_tensor(size, dtype, stride=None, device="cuda"):
       kwargs = {}
       if dtype in _FLOATING_TYPES:
           kwargs.update({"low": 0, "high": 1})
       
       # Fall back to CPU if CUDA is not available
       if device == "cuda" and not torch.cuda.is_available():
           device = "cpu"
       
       if stride is not None:
           extent = 1 + sum((size - 1) * stride for size, stride in zip(size, stride))
           data = make_tensor(extent, dtype=dtype, device=device, **kwargs)
           return data.as_strided(size, stride)
       return make_tensor(size, dtype=dtype, device=device, **kwargs)
   ```

### CUDA Stream Detection
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/utils.py`  
**Lines:** 50-97

```python
def uses_cuda_stream(func) -> bool:
    """
    Detects whether a Python function creates CUDA streams.
    
    Reasons: CUDA streams cause race conditions during benchmarking
    because they don't synchronize properly during timing measurements.
    """
    try:
        source = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(source))
    except (TypeError, OSError, IndentationError, SyntaxError):
        return False

    # Check for stream creation patterns
    patterns = [
        r"torch\.cuda\.Stream\(",  # torch.cuda.Stream() constructor
        r"cupy\.cuda\.Stream\(",   # cupy.cuda.Stream() constructor
        r"cuda\.Stream\(",         # Generic cuda.Stream() constructor
        r"pycuda.*Stream\(",       # PyCUDA stream creation
        r"\bStream\(",             # Stream() constructor calls
        r"make_stream\(",          # make_stream() factory function
        r"create_stream\(",        # create_stream() factory function
    ]

    if any(re.search(p, source, re.IGNORECASE) for p in patterns):
        return True

    # ... AST-based check for Stream() constructors ...
    return finder.found
```

### CUDA Stream Skipping in eval_one_op
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/eval.py`  
**Lines:** 236-262

```python
def eval_one_op(op, impl, correctness_tests, performance_tests):
    """Evaluate impl of op against correctness_tests and performance_tests."""
    
    if uses_cuda_stream(impl):
        logger.warning(f"Skipping {op.__name__} because it uses CUDA stream")
        
        # Return failure for both correctness and performance
        correctness_results = [
            CorrectnessTestResult(
                op_name=op.__name__,
                args=args_str,
                is_correct=False,
                error_msg="Skipped: uses CUDA stream",
            )
            for test in correctness_tests
        ]
        
        performance_results = [
            PerformanceTestResult(
                op_name=op.__name__,
                args=args_str,
                speedup=0,
                benchmark_time_ms=0,
                reference_time_ms=0,
                error_msg="Skipped: uses CUDA stream",
            )
            for test in performance_tests
        ]
        
        return 0, 1.0, correctness_results, performance_results
    
    # Normal execution (GPU or CPU)
    correctness_score, correctness_results = eval_correctness(op, impl, correctness_tests)
    performance_score, performance_results = eval_performance(op, impl, performance_tests)
    return correctness_score, performance_score, correctness_results, performance_results
```

---

## 6. Test Input/Output Device Placement

### Test Data Creation
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/utils.py`  
**Lines:** 100-113

```python
def _deserialize_tensor(size, dtype, stride=None, device="cuda"):
    """Deserialize tensor arguments for tests."""
    kwargs = {}
    if dtype in _FLOATING_TYPES:
        kwargs.update({"low": 0, "high": 1})

    # DEFAULT: CUDA. But fall back to CPU if unavailable.
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if stride is not None:
        extent = 1 + sum((size - 1) * stride for size, stride in zip(size, stride))
        data = make_tensor(extent, dtype=dtype, device=device, **kwargs)
        return data.as_strided(size, stride)
    
    return make_tensor(size, dtype=dtype, device=device, **kwargs)
```

### Test Execution
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/eval.py`  
**Lines:** 93-125

```python
def eval_correctness_test(op, impl, test) -> CorrectnessTestResult:
    """Evaluate impl of op against test."""
    
    # Get test data (tensors on GPU if torch.cuda.is_available())
    args, kwargs = test.args, test.kwargs
    
    # Reference: PyTorch's native operation
    ref = op(*args, **kwargs)  # Runs on GPU (tensors are GPU)
    
    try:
        # Implementation: User's kernel
        res = impl(*args, **kwargs)  # Runs on GPU (same tensors)
        
        # Compare outputs
        is_correct = allclose(ref, res)
        
        # Compute error metrics
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
        # Handle execution errors
        error_msg = format_exception(e, op, args, kwargs, traceback.format_exc())
        result = CorrectnessTestResult(
            op_name=op.__name__,
            args=serialize_args(args, kwargs),
            is_correct=False,
            error_msg=error_msg,
            error_type=str(type(e)),
            traceback=traceback.format_exc(),
        )
        return result
```

### Key Point: Device Placement
- **Test creation default**: `device="cuda"` in `_deserialize_tensor`
- **Fallback to CPU**: If `torch.cuda.is_available()` returns False
- **Both local and remote**: Tests run on GPU by default, CPU only if GPU unavailable
- **No explicit device transfers**: Tensors created on target device directly

---

## 7. Triton Kernel Compilation and Execution

### Triton Import and Availability
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/eval.py`  
**Lines:** 43-51

```python
try:
    if torch.cuda.is_available():
        import triton.testing
        TRITON_AVAILABLE = True
    else:
        TRITON_AVAILABLE = False
except ImportError:
    TRITON_AVAILABLE = False
```

### Triton Kernel Code Handling
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/utils.py`  
**Lines:** 355-407

```python
def save_kernel_to_file(kernel_code: str, kernel_file_path: str) -> None:
    """Save kernel code to a file."""

    def _prepare_triton_code(kernel_code: str) -> str:
        """Prepare Triton kernel code with necessary imports."""
        imports = """
import torch
import triton
import triton.language as tl
"""
        if "import torch" not in kernel_code:
            kernel_code = imports + kernel_code
        return kernel_code

    def _prepare_torch_code(kernel_code: str) -> str:
        """Prepare regular PyTorch kernel code with necessary imports."""
        imports = """
import torch
import torch.nn.functional as F
"""
        if "import torch" not in kernel_code:
            kernel_code = imports + kernel_code
        return kernel_code

    # Detect kernel type by code patterns
    is_triton = "triton.jit" in kernel_code or "@triton.jit" in kernel_code
    is_helion = "helion.kernel" in kernel_code or "@helion.kernel" in kernel_code

    if is_triton:
        full_code = _prepare_triton_code(kernel_code)
    elif is_helion:
        full_code = _prepare_helion_code(kernel_code)
    else:
        full_code = _prepare_torch_code(kernel_code)

    # Write to disk
    with open(kernel_file_path, "w") as f:
        f.write(full_code)
```

### Kernel Compilation
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/utils.py`  
**Lines:** 410-443

```python
def compile_kernel_from_string(
    kernel_code: str, op_name: str, kernel_file_path: str, 
    expected_fn_name: str, module_name: str
) -> Callable:
    """Compile kernel code string into executable Python function."""
    
    def _find_kernel_function(module, folder_name: str) -> Callable:
        """Find the main kernel function in the compiled module."""
        expected_name = f"{folder_name}_kernel_impl"

        if hasattr(module, expected_name):
            return getattr(module, expected_name)

        available_functions = [
            name for name in dir(module)
            if callable(getattr(module, name)) and not name.startswith("_")
        ]

        raise ValueError(
            f"Expected function '{expected_name}' not found in kernel code for {op_name}. "
            f"Available functions: {available_functions}."
        )

    try:
        # Step 1: Save kernel to file (with imports added)
        save_kernel_to_file(kernel_code, kernel_file_path)

        # Step 2: Load as Python module using importlib
        spec = importlib.util.spec_from_file_location(module_name, kernel_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # EXECUTE THE CODE (Triton JIT happens here)

        # Step 3: Extract the kernel function
        kernel_func = _find_kernel_function(module, expected_fn_name)
        return kernel_func

    except Exception as e:
        raise RuntimeError(f"Failed to compile kernel for {op_name}: {str(e)}") from e
```

### Key Points for Triton
1. **JIT Compilation**: Happens during `spec.loader.exec_module(module)` when the `@triton.jit` decorator is evaluated
2. **GPU Code Generation**: Triton generates GPU code and compiles to PTXAS
3. **Lazy Compilation**: Actual GPU kernel compilation happens first time kernel is called
4. **Requires CUDA**: Triton only works if `torch.cuda.is_available()` and dependencies installed

---

## 8. Triton Benchmarking

### GPU Benchmarking Function
**File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/BackendBench/eval.py`  
**Lines:** 149-159, 162-224

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


def eval_performance(op, impl, tests) -> Tuple[float, List[PerformanceTestResult]]:
    """Evaluate performance of impl against tests."""
    
    # SELECT BENCHMARK FUNCTION
    bench_fn = (
        triton.testing.do_bench  # GPU benchmarking with synchronization
        if TRITON_AVAILABLE and torch.cuda.is_available() 
        else cpu_bench  # CPU benchmarking fallback
    )
    
    base_times = []
    test_times = []

    for test in tests:
        # Cache arguments for consistency
        cached_args = test.args
        cached_kwargs = test.kwargs
        args_str = serialize_args(cached_args, cached_kwargs)
        
        # BENCHMARK REFERENCE
        base_time = bench_fn(lambda: op(*cached_args, **cached_kwargs))
        base_times.append(base_time)
        
        test_time = base_time
        try:
            # Verify correctness first
            ref = op(*cached_args, **cached_kwargs)
            res = impl(*cached_args, **cached_kwargs)
            
            if not allclose(ref, res):
                abs_error, rel_error = compute_errors(ref, res)
                raise ValueError(
                    f"Reference and result tensors are not close: "
                    f"max absolute error {abs_error}, max relative error {rel_error}"
                )
            
            # BENCHMARK IMPLEMENTATION
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
            # If benchmarking fails, record error
            performance_results.append(
                PerformanceTestResult(
                    op_name=op.__name__,
                    args=args_str,
                    successfully_ran=False,
                    speedup=None,
                    benchmark_time_ms=None,
                    reference_time_ms=base_time,
                    error_msg=str(e),
                )
            )
        finally:
            test_times.append(test_time)

    # AGGREGATE: Geometric mean of speedups
    speedups = torch.tensor(base_times) / torch.tensor(test_times)
    return speedups.log().mean().exp(), performance_results
```

### Triton's `do_bench` Function
- Uses `triton.testing.do_bench()` for GPU benchmarking
- Handles GPU synchronization properly
- Measures milliseconds with proper warmup
- Only available when:
  - Triton installed
  - CUDA available
  - GPU compute capability supported

### GPU Synchronization
- `triton.testing.do_bench()` internally calls `torch.cuda.synchronize()`
- Ensures all GPU operations complete before timing
- Prevents race conditions in benchmarking
- More accurate than manual timing on GPU

---

## 9. Complete GPU Execution Flow Diagram

### Local Execution Path (gpu="local")
```
User Code
    |
    v
CodeEvaluator.__call__(code, op_name)
    |
    +---> is_modal = False
    |     self.callable = run_code (direct function reference)
    |
    v
asyncio.to_thread(run_code, op_test, code, ...)
    |
    v
run_code(op_test, code, ...)
    |
    +---> compile_kernel_from_string()
    |     |
    |     +---> save_kernel_to_file(code)
    |     |
    |     +---> importlib.util.exec_module()  
    |     |     (Triton JIT here if @triton.jit present)
    |     |
    |     +---> Extract "{op_name}_kernel_impl"
    |     |
    |     v
    |     kernel_fn (Triton or PyTorch kernel)
    |
    +---> eval_one_op(torch.ops.aten.X, kernel_fn, tests)
    |
    +---> Check uses_cuda_stream(kernel_fn)
    |     (Skip if CUDA streams detected)
    |
    +---> eval_correctness(op, kernel_fn, correctness_tests)
    |     For each test:
    |       - Deserialize args/kwargs (device="cuda" by default)
    |       - ref = torch.ops.aten.X(*args, **kwargs)  [GPU]
    |       - res = kernel_fn(*args, **kwargs)         [GPU]
    |       - allclose(ref, res) with atol=1e-2, rtol=1e-2
    |
    +---> eval_performance(op, kernel_fn, performance_tests)
    |     For each test:
    |       - bench_fn = triton.testing.do_bench (GPU)
    |                    or cpu_bench (CPU fallback)
    |       - base_time = bench_fn(lambda: op(*args))
    |       - test_time = bench_fn(lambda: kernel_fn(*args))
    |       - speedup = base_time / test_time
    |
    v
(correctness_score, performance_score, results)
    |
    v
CodeEvaluationResult
```

### Remote Execution Path (gpu="T4"|"L4"|"A100"|etc)
```
User Code
    |
    v
CodeEvaluator.__call__(code, op_name)
    |
    +---> is_modal = True
    |     fn = modal.Function.from_name(
    |         "backend-bench-env-runner", 
    |         f"eval_code_{gpu.lower()}"
    |     )
    |     self.callable = lambda **kwargs: fn.remote(**kwargs)
    |
    v
asyncio.to_thread(fn.remote, suite_name, ops, code, ...)
    |
    v
Modal serializes args and sends to cloud
    |
    v
Modal GPU container (NVIDIA CUDA 12.8)
    |
    +---> eval_code(suite_name, ops, code, op_name, ...)
    |     (Same as run_code, but runs on Modal GPU)
    |
    +---> compile_kernel_from_string()
    |     (Compiles on GPU container)
    |
    +---> eval_one_op(op, kernel_fn, tests)
    |     (Tests run on Modal GPU)
    |
    v
Return results back to local machine
    |
    v
CodeEvaluationResult
```

---

## 10. Key Summary of GPU Usage

### GPU Configuration Decision
```python
# In CodeEvaluator.__init__
if gpu == "local":
    self.callable = run_code  # Local function
else:  # gpu in ["T4", "L4", "A100", "H100", "H200", "B200"]
    self.callable = lambda **kwargs: modal_fn.remote(**kwargs)  # Remote
```

### GPU Selection for Benchmarking
```python
# In eval_performance
bench_fn = (
    triton.testing.do_bench 
    if TRITON_AVAILABLE and torch.cuda.is_available() 
    else cpu_bench
)
```

### Test Device Placement
```python
# In _deserialize_tensor
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor = make_tensor(size, dtype=dtype, device=device)
```

### GPU Memory Management
```python
# Available in utils.py (but not called automatically)
def cleanup_memory_and_gpu():
    torch.cuda.synchronize()  # Sync GPU
    torch.cuda.empty_cache()  # Clear cache
```

### CUDA Stream Detection
```python
# In eval_one_op
if uses_cuda_stream(kernel_fn):
    # Skip this kernel (CUDA streams break benchmarking)
    return 0, 1.0, failure_results, failure_results
```

---

## 11. GPU-Related Code Snippets Reference

### Checking GPU Availability
```python
import torch
torch.cuda.is_available()  # True if GPU present
```

### Tensor Creation on GPU
```python
from torch.testing import make_tensor
tensor = make_tensor(
    size=[128, 256], 
    dtype=torch.float32, 
    device="cuda"  # or "cpu"
)
```

### GPU Synchronization
```python
torch.cuda.synchronize()  # Wait for all GPU ops to complete
```

### GPU Memory Management
```python
torch.cuda.empty_cache()  # Clear GPU memory cache
```

### Triton Benchmarking
```python
import triton.testing
time_ms = triton.testing.do_bench(
    lambda: my_kernel(x, y, z),
    warmup=25,
    rep=100
)
```

### CUDA Stream Detection Patterns
```python
# Detected by uses_cuda_stream():
torch.cuda.Stream()          # Pattern: torch\.cuda\.Stream\(
cupy.cuda.Stream()           # Pattern: cupy\.cuda\.Stream\(
cuda.Stream()                # Pattern: cuda\.Stream\(
Stream()                     # Pattern: \bStream\(
make_stream()                # Pattern: make_stream\(
create_stream()              # Pattern: create_stream\(
```

---

## 12. Execution Mode Comparison

| Aspect | Local (gpu="local") | Remote (gpu="T4"+"L4"+etc) |
|--------|---|---|
| **Callable** | `run_code` | `modal_fn.remote()` |
| **Location** | Local machine | Modal cloud |
| **Process** | Current process via thread | Remote process in container |
| **torch.cuda** | Uses local GPU | Uses Modal-allocated GPU |
| **Triton Compilation** | Compiles locally | Compiles on Modal container |
| **Test Data** | GPU (fallback to CPU) | GPU (fallback to CPU) |
| **Benchmarking** | triton.testing.do_bench | triton.testing.do_bench |
| **Scheduling** | None | Modal handles |
| **Docker** | None | NVIDIA CUDA 12.8 |
| **Cost** | Hardware already owned | Modal credits |
| **Note** | Results unreliable (no scheduling) | Results reliable (proper scheduling) |

---

**Generated:** 2025-11-12  
**Based on Analysis of Backend-Bench Source Code**
