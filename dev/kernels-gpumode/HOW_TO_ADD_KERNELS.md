# How to Add Your Custom Kernels

## Quick Answer

**Where to write kernels?** Anywhere in the project root:
- `nvfp4_triton_kernel.py` (already created as template)
- `nvfp4_cute_kernel.py` (already created as template)
- `nvfp4_custom_v1.py`, `nvfp4_custom_v2.py`, etc.

**How are they found?** They auto-register when imported in `kernel_utils/smoke_test.py` (lines 33-41)

---

## The Simple 3-Step Process

### Step 1: Write Your Kernel Implementation

Edit `nvfp4_triton_kernel.py`:

```python
import triton
import triton.language as tl
from kernel_utils.task import input_t, output_t
from kernel_utils.backends import BACKENDS

@triton.jit
def my_triton_kernel(...):
    # Your Triton kernel implementation
    pass

def triton_kernel(data: input_t) -> output_t:
    """Wrapper function - must match this signature."""
    a, b, scale_a, scale_b, scale_a_perm, scale_b_perm, c = data

    # Launch your kernel
    my_triton_kernel[grid](...)

    return c  # Must return the output tensor

# This registers it automatically
BACKENDS.register(
    name="triton",
    kernel_fn=triton_kernel,
    description="My Triton FP4 kernel",
    language="triton",
)
```

### Step 2: Import in smoke_test.py

Already done! Lines 33-41 in `kernel_utils/smoke_test.py`:

```python
try:
    import nvfp4_triton_kernel  # Finds and registers your kernel
except ImportError:
    pass  # Skip if not implemented
```

### Step 3: Test It

```bash
# Test just your kernel
python -m kernel_utils.smoke_test triton

# Test all kernels
python -m kernel_utils.smoke_test

# Deploy to remote GPU
python smoke_deploy.py --ssh root@gpu:22 --backends triton
```

---

## File Structure

```
dev/integration-evaluation/
├── nvfp4_reference_kernel.py     ← Reference (already exists)
├── nvfp4_triton_kernel.py        ← Edit this for Triton
├── nvfp4_cute_kernel.py          ← Edit this for CuTe
│
├── kernel_utils/
│   └── smoke_test.py             ← Imports your kernels (lines 33-41)
│
└── smoke_deploy.py               ← Deploys to remote GPU
```

---

## Required Function Signature

**Your kernel function MUST have this signature:**

```python
def your_kernel(data: input_t) -> output_t:
    """
    Args:
        data: tuple of 7 tensors:
            - a: [m, k, l] float4_e2m1fn_x2
            - b: [1, k, l] float4_e2m1fn_x2
            - scale_a: [m, k, l] float8_e4m3fn (CPU)
            - scale_b: [1, k, l] float8_e4m3fn (CPU)
            - scale_a_permuted: GPU version
            - scale_b_permuted: GPU version
            - c: [m, 1, l] float16 (output buffer)

    Returns:
        c: Modified output tensor [m, 1, l] float16
    """
    a, b, scale_a, scale_b, scale_a_perm, scale_b_perm, c = data

    # Your implementation

    return c
```

**The signature is strict because:**
- All kernels must accept same input format
- Allows fair comparison (same data for all)
- Enables automatic testing

---

## Example: Triton Kernel

```python
# nvfp4_triton_kernel.py
import triton
import triton.language as tl
from kernel_utils.task import input_t, output_t
from kernel_utils.backends import BACKENDS

@triton.jit
def nvfp4_gemv_kernel(
    a_ptr, b_ptr, c_ptr,
    scale_a_ptr, scale_b_ptr,
    M, K, L,
    stride_am, stride_ak, stride_al,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Your Triton kernel."""
    # Get program ID
    pid_m = tl.program_id(0)
    pid_l = tl.program_id(1)

    # Load data
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # Compute
    # ... your implementation ...

    # Store result
    tl.store(c_ptr + offs_m * stride_cm + pid_l * stride_cl, result)


def triton_kernel(data: input_t) -> output_t:
    """Wrapper that launches Triton kernel."""
    a, b, scale_a, scale_b, scale_a_perm, scale_b_perm, c = data
    M, K, L = a.shape

    # Define grid
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        L,
    )

    # Launch kernel
    nvfp4_gemv_kernel[grid](
        a, b, c,
        scale_a_perm, scale_b_perm,
        M, K, L,
        a.stride(0), a.stride(1), a.stride(2),
        BLOCK_M=64,
        BLOCK_K=64,
    )

    return c

# Register automatically on import
BACKENDS.register(
    name="triton",
    kernel_fn=triton_kernel,
    description="Triton FP4 GEMV with 64x64 tiles",
    language="triton",
)
```

---

## Example: CuTe/CUDA Kernel

```python
# nvfp4_cute_kernel.py
import torch
from kernel_utils.task import input_t, output_t
from kernel_utils.backends import BACKENDS

# Assuming you compiled a CUDA extension
try:
    import nvfp4_cuda_extension
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False

def cute_kernel(data: input_t) -> output_t:
    """Wrapper for CuTe CUDA kernel."""
    if not HAS_CUDA_EXT:
        raise RuntimeError("CUDA extension not compiled")

    a, b, scale_a, scale_b, scale_a_perm, scale_b_perm, c = data

    # Call your compiled CUDA extension
    # Note: CuTe expects the permuted scale factors
    nvfp4_cuda_extension.gemv_forward(
        a, b, scale_a_perm, scale_b_perm, c
    )

    return c

# Register
BACKENDS.register(
    name="cute",
    kernel_fn=cute_kernel,
    description="CuTe FP4 tensor core kernel",
    language="cutlass",
)
```

---

## Testing Your Kernel

### Local Testing (if you have GPU locally)

```bash
# Test just your kernel
python -m kernel_utils.smoke_test triton

# Expected output:
# 🔥 Running NVFP4 Smoke Tests
#    Backends: triton
# 🔧 Backend: triton (triton)
#    My Triton FP4 kernel
# 📊 Test: tiny_single
#    ✅ Correctness: PASS
#    ⏱️  Performance: 0.189ms (1.24x)
```

### Remote Testing

```bash
# Deploy and test
python smoke_deploy.py --ssh root@gpu-server:22 --backends triton

# With result download
python smoke_deploy.py --ssh root@gpu-server:22 --backends triton --save
```

### Testing Multiple Kernels

```bash
# Compare all implementations
python -m kernel_utils.smoke_test reference triton cute

# Output shows comparison:
# Backend              Correctness     Speedup
# reference            ✅ 100.0%       1.00x
# triton               ✅ 100.0%       1.24x
# cute                 ✅ 100.0%       2.09x
```

---

## Multiple Versions of Same Kernel

You can register multiple versions:

```python
# nvfp4_triton_experiments.py

# Version 1: 64x64 tiles
def triton_v1(data):
    # ... BLOCK_M=64, BLOCK_K=64
    pass

BACKENDS.register("triton_64", triton_v1, "Triton 64x64 tiles", "triton")

# Version 2: 128x128 tiles
def triton_v2(data):
    # ... BLOCK_M=128, BLOCK_K=128
    pass

BACKENDS.register("triton_128", triton_v2, "Triton 128x128 tiles", "triton")

# Version 3: Optimized with better memory access
def triton_v3(data):
    # ... optimizations
    pass

BACKENDS.register("triton_opt", triton_v3, "Triton optimized", "triton")
```

Then import in smoke_test.py:

```python
try:
    import nvfp4_triton_experiments  # Registers triton_64, triton_128, triton_opt
except ImportError:
    pass
```

Test them:

```bash
python -m kernel_utils.smoke_test triton_64 triton_128 triton_opt
```

---

## Common Mistakes

### ❌ Wrong Signature

```python
# WRONG - different signature
def triton_kernel(a, b, scale_a, scale_b, c):
    return c
```

### ✅ Correct Signature

```python
# CORRECT - takes tuple, returns tensor
def triton_kernel(data: input_t) -> output_t:
    a, b, scale_a, scale_b, scale_a_perm, scale_b_perm, c = data
    return c
```

### ❌ Forgot to Register

```python
# WRONG - kernel exists but not registered
def triton_kernel(data):
    return c
# Missing: BACKENDS.register(...)
```

### ✅ Always Register

```python
def triton_kernel(data):
    return c

# CORRECT - register after defining
BACKENDS.register("triton", triton_kernel, "My kernel", "triton")
```

### ❌ Wrong Import Location

```python
# WRONG - imported in wrong place
# some_random_file.py
import nvfp4_triton_kernel  # Won't be found by smoke_test
```

### ✅ Import in smoke_test.py

```python
# CORRECT - imported in kernel_utils/smoke_test.py (lines 33-41)
try:
    import nvfp4_triton_kernel
except ImportError:
    pass
```

---

## Summary

1. **Write kernel** in `nvfp4_triton_kernel.py` or `nvfp4_cute_kernel.py`
2. **Match signature**: `def kernel(data: input_t) -> output_t:`
3. **Register**: `BACKENDS.register(name, kernel_fn, description, language)`
4. **Import** in `kernel_utils/smoke_test.py` (already done for triton/cute)
5. **Test**: `python -m kernel_utils.smoke_test triton`

The templates (`nvfp4_triton_kernel.py`, `nvfp4_cute_kernel.py`) are ready for you to edit!
