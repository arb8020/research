# Backend-Bench GPU Execution - Quick Reference Guide

## Question Answers

### 1. How is the GPU configured?

**Config File:** `/Users/chiraagbalu/research/.venv/lib/python3.13/site-packages/src/config.py`

```python
@dataclass
class BackendBenchConfig:
    gpu: str = "T4"  # Options: "local", "T4", "L4", "A100", "H100", "H200", "B200"
```

**Configuration Determines:**
- `gpu="local"` → Uses local machine GPU via thread pool (no scheduling)
- `gpu="T4"|"L4"|etc` → Uses Modal cloud GPU with proper scheduling

---

### 2. What happens when gpu != "local"? (Remote Execution Path)

**CodeEvaluator Initialization:** Lines 113-140 of code_evaluator.py

```python
if not self.is_modal:  # if gpu == "local"
    from src.utils import run_code
    self.callable = run_code  # Direct function
else:  # if gpu != "local"
    import modal
    fn = modal.Function.from_name(
        "backend-bench-env-runner", 
        f"eval_code_{self.cfg.gpu.lower()}"  # e.g., "eval_code_t4"
    )
    self.callable = lambda **kwargs: fn.remote(**kwargs)  # Remote call
```

**Remote Execution Uses Modal:**
- Serializes code and test params
- Sends to Modal cloud
- Runs in NVIDIA CUDA 12.8 Docker container
- Returns results back to local machine

**Modal Registration:** modal_runner.py Lines 100-104
```python
for gpu in {"T4", "L4", "A100-80GB", "H100!", "H200", "B200"}:
    gpu_slug = gpu.lower().split("-")[0].strip("!").replace(":", "x")
    app.function(
        gpu=gpu, 
        image=cuda_image, 
        name=f"eval_code_{gpu_slug}", 
        serialized=True
    )(eval_code)
```

---

### 3. How does the compiled kernel actually run on GPU?

**Kernel Compilation Path:** utils.py Lines 410-443

```python
def compile_kernel_from_string(
    kernel_code: str, op_name: str, kernel_file_path: str, 
    expected_fn_name: str, module_name: str
) -> Callable:
    # Step 1: Save to file (adds imports if Triton/Helion detected)
    save_kernel_to_file(kernel_code, kernel_file_path)
    
    # Step 2: Load as Python module
    spec = importlib.util.spec_from_file_location(module_name, kernel_file_path)
    module = importlib.util.module_from_spec(spec)
    
    # Step 3: Execute module (Triton JIT happens here!)
    spec.loader.exec_module(module)
    
    # Step 4: Extract kernel function named "{op_name}_kernel_impl"
    kernel_func = _find_kernel_function(module, op_name)
    return kernel_func
```

**GPU Execution Path:** eval.py Lines 93-125

```python
def eval_correctness_test(op, impl, test) -> CorrectnessTestResult:
    args, kwargs = test.args, test.kwargs  # Test data on GPU
    
    # Run PyTorch reference
    ref = op(*args, **kwargs)  # GPU operation
    
    # Run user's kernel
    res = impl(*args, **kwargs)  # GPU operation (Triton or PyTorch)
    
    # Compare outputs
    is_correct = allclose(ref, res)  # Returns bool
    return CorrectnessTestResult(...)
```

**Key:** Both `op` and `impl` receive GPU tensors and execute on GPU. No explicit `.to("cuda")` needed.

---

### 4. Are test inputs/outputs on GPU or CPU?

**Test Tensor Creation:** utils.py Lines 100-113

```python
def _deserialize_tensor(size, dtype, stride=None, device="cuda"):
    # DEFAULT: Create on CUDA
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"  # Fallback if no GPU
    
    return make_tensor(size, dtype=dtype, device=device, **kwargs)
```

**Default Behavior:**
- **Tensors created with `device="cuda"` by default**
- Falls back to CPU only if `torch.cuda.is_available()` returns False
- Both reference and implementation receive same tensors
- No device transfers during test execution

**In Both Local and Remote Execution:**
- Tests run on GPU when available
- CPU fallback if GPU unavailable
- Device placement is transparent to user code

---

### 5. CUDA, torch.cuda, and GPU-specific code in execution path

**GPU Availability Check:** eval.py Lines 43-51

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

**GPU Benchmarking Selection:** eval.py Lines 164-166

```python
def eval_performance(op, impl, tests):
    # CRITICAL DECISION POINT
    bench_fn = (
        triton.testing.do_bench 
        if TRITON_AVAILABLE and torch.cuda.is_available() 
        else cpu_bench
    )
```

**torch.cuda Usage in Code:**

1. **Availability Check**
   ```python
   torch.cuda.is_available()  # At import and runtime
   ```

2. **GPU Synchronization** (not called automatically, available in utils)
   ```python
   def cleanup_memory_and_gpu():
       gc.collect()
       torch.cuda.synchronize()  # Wait for GPU ops
       torch.cuda.empty_cache()  # Clear GPU cache
   ```

3. **CUDA Stream Detection** - utils.py Lines 50-97
   ```python
   def uses_cuda_stream(func) -> bool:
       # Checks for patterns:
       # - torch.cuda.Stream()
       # - cupy.cuda.Stream()
       # - cuda.Stream()
       # - etc.
       return True if found else False
   ```

4. **CUDA Stream Skipping** - eval.py Lines 236-262
   ```python
   if uses_cuda_stream(impl):
       logger.warning(f"Skipping {op.__name__} because it uses CUDA stream")
       # Return failure for both correctness and performance
       return 0, 1.0, failure_results, failure_results
   ```

---

### 6. How does Triton kernel compilation work?

**Triton Detection:** utils.py Lines 390-398

```python
def save_kernel_to_file(kernel_code: str, kernel_file_path: str) -> None:
    # Detect Triton kernels by patterns
    is_triton = "triton.jit" in kernel_code or "@triton.jit" in kernel_code
    
    if is_triton:
        # Add Triton imports
        imports = """
import torch
import triton
import triton.language as tl
"""
        full_code = imports + kernel_code
    else:
        # Add PyTorch imports
        imports = """
import torch
import torch.nn.functional as F
"""
        full_code = imports + kernel_code
    
    with open(kernel_file_path, "w") as f:
        f.write(full_code)
```

**JIT Compilation Happens Here:** utils.py Lines 435-437

```python
spec = importlib.util.spec_from_file_location(module_name, kernel_file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # <-- TRITON JIT HAPPENS HERE
                                  # When @triton.jit decorator is evaluated
                                  # Triton compiles to GPU code (PTXAS)
```

**Timeline:**
1. User submits code with `@triton.jit` decorated function
2. Code saved to temp file with Triton imports added
3. `exec_module()` loads and parses the file
4. `@triton.jit` decorator is evaluated
5. Triton performs JIT compilation to GPU bytecode
6. Function returned as callable
7. First call triggers lazy GPU compilation

---

### 7. Difference between local vs non-local GPU execution

**Configuration Decision:**

```python
# CodeEvaluator.__init__ (code_evaluator.py Lines 124-136)
if self._gpu == "local":
    # LOCAL: Direct execution
    self.callable = run_code
else:  # gpu in ["T4", "L4", "A100", "H100", "H200", "B200"]
    # REMOTE: Modal execution
    self.callable = lambda **kwargs: fn.remote(**kwargs)
```

**Detailed Comparison:**

| Aspect | Local (gpu="local") | Remote (gpu="T4", etc) |
|--------|---|---|
| **Function** | `run_code` (utils.py) | `eval_code` (modal_runner.py) |
| **Location** | Local machine | Modal cloud GPU |
| **Execution** | Thread pool in current process | Remote function call |
| **Code Path** | Direct Python call | RPC via Modal |
| **Test Loading** | Full OpTest object | Suite name + ops metadata |
| **Docker** | None (uses local env) | NVIDIA CUDA 12.8 container |
| **GPU Device** | `torch.cuda.is_available()` | Modal-allocated GPU |
| **Scheduling** | None (can context switch) | Modal handles scheduling |
| **Benchmark** | triton.testing.do_bench | triton.testing.do_bench |
| **Result** | Returns to same process | Serialized back to local |

**Local Execution Flow:**
```python
asyncio.to_thread(run_code, op_test, code, ...)
    → run_code compiles kernel
    → eval_one_op runs tests on local GPU
    → Returns scores and results
```

**Remote Execution Flow:**
```python
asyncio.to_thread(fn.remote, suite_name, ops, code, ...)
    → Sends to Modal cloud
    → Modal container runs eval_code
    → Compiles kernel on cloud GPU
    → Runs tests on cloud GPU
    → Serializes results back
```

---

## Code Locations Quick Reference

| Component | File | Lines |
|-----------|------|-------|
| GPU config | `src/config.py` | 8-42 |
| Local vs Modal decision | `src/code_evaluator.py` | 113-140 |
| Local execution | `src/utils.py:run_code` | 124-140 |
| Remote execution | `src/modal_runner.py` | 1-104 |
| Kernel compilation | `BackendBench/utils.py:compile_kernel_from_string` | 410-443 |
| Kernel saving | `BackendBench/utils.py:save_kernel_to_file` | 355-407 |
| Test execution | `BackendBench/eval.py:eval_one_op` | 227-271 |
| Correctness tests | `BackendBench/eval.py:eval_correctness_test` | 93-125 |
| Performance tests | `BackendBench/eval.py:eval_performance` | 162-224 |
| GPU benchmarking | `BackendBench/eval.py:cpu_bench` | 149-159 |
| CUDA stream detection | `BackendBench/utils.py:uses_cuda_stream` | 50-97 |
| Tensor deserialization | `BackendBench/utils.py:_deserialize_tensor` | 100-113 |
| GPU memory cleanup | `BackendBench/utils.py:cleanup_memory_and_gpu` | 245-249 |

---

## Key Code Snippets

### Checking GPU Mode
```python
from src.code_evaluator import CodeEvaluator
from src.config import BackendBenchConfig

cfg = BackendBenchConfig(gpu="local")  # or "T4", "L4", etc
evaluator = CodeEvaluator(cfg)

print(f"Using Modal: {evaluator.is_modal}")  # False for "local", True for others
print(f"GPU: {evaluator._gpu}")  # "local" or "T4" or "L4" etc
```

### Tensor Creation on GPU
```python
from BackendBench.utils import _deserialize_tensor

# Creates tensor on CUDA if available, otherwise CPU
tensor = _deserialize_tensor(
    size=[128, 256], 
    dtype=torch.float32, 
    stride=None,
    device="cuda"  # Default
)
print(f"Device: {tensor.device}")  # cuda:0 or cpu
```

### CUDA Stream Detection
```python
from BackendBench.utils import uses_cuda_stream

def my_kernel(x):
    stream = torch.cuda.Stream()  # This will be detected
    with torch.cuda.stream(stream):
        y = x + 1
    return y

print(uses_cuda_stream(my_kernel))  # True
```

### Benchmarking Selection
```python
import torch
try:
    if torch.cuda.is_available():
        import triton.testing
        bench_fn = triton.testing.do_bench
    else:
        bench_fn = cpu_bench
except ImportError:
    bench_fn = cpu_bench

time_ms = bench_fn(lambda: kernel(x, y))
```

---

## Summary

1. **GPU Config**: Determined by `gpu` parameter ("local" vs "T4"/"L4"/etc)
2. **Local GPU**: Direct execution via `run_code` in thread pool
3. **Remote GPU**: Remote execution via Modal cloud containers
4. **Kernel Compilation**: Via `importlib.util.exec_module()`, Triton JIT at this point
5. **Test Data**: GPU tensors by default, CPU fallback if no CUDA
6. **GPU Checks**: `torch.cuda.is_available()` used at import and runtime
7. **Triton Compilation**: Happens in `exec_module()` when `@triton.jit` decorator evaluates
8. **CUDA Streams**: Detected and skipped (cause benchmarking race conditions)
9. **Benchmarking**: `triton.testing.do_bench()` for GPU, `cpu_bench()` fallback
10. **Synchronization**: `triton.testing.do_bench()` handles internal synchronization

