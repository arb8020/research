# Task: Restore WebGPU Backend

The WebGPU backend has been stubbed out. Your task is to restore it.

## What's been removed

1. **`tinygrad/runtime/ops_webgpu.py`** — Device, program, allocator (uses wgpu-native FFI)
2. **`tinygrad/runtime/autogen/webgpu.py`** — ctypes bindings for wgpu-native
3. **`tinygrad/renderer/wgsl.py`** — WGSLRenderer class (generates WGSL shader code)

## What you have

1. **`headers/webgpu.h`** — The C header file for the WebGPU API (wgpu-native)
2. **`WEBGPU_LIB_PATH` env var** — Path to `libwebgpu_dawn.dylib` (the compiled runtime)

You must:
1. Read `headers/webgpu.h` to understand the API
2. Write ctypes FFI bindings in `tinygrad/runtime/autogen/webgpu.py`
3. Implement the runtime in `tinygrad/runtime/ops_webgpu.py`
4. Implement the renderer in `tinygrad/renderer/wgsl.py`

## Success criteria

```bash
# This should work:
PYTHONPATH=. WEBGPU=1 python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"

# Basic tensor operations should work:
PYTHONPATH=. WEBGPU=1 python -c "from tinygrad import Tensor; a=Tensor([1,2,3]); b=Tensor([4,5,6]); print((a+b).numpy())"
```

## Writing ctypes bindings from headers/webgpu.h

The header file contains:
- Struct definitions (e.g., `WGPUBufferDescriptor`, `WGPUShaderModuleDescriptor`)
- Enum definitions (e.g., `WGPUBufferUsage`, `WGPUMapMode`)
- Function declarations (e.g., `wgpuCreateInstance`, `wgpuDeviceCreateBuffer`)

You need to translate these to Python ctypes. Example pattern:

```c
// From webgpu.h
typedef struct WGPUBufferDescriptor {
    WGPUChainedStruct const * nextInChain;
    char const * label;
    WGPUBufferUsageFlags usage;
    uint64_t size;
    WGPUBool mappedAtCreation;
} WGPUBufferDescriptor;

WGPU_EXPORT WGPUBuffer wgpuDeviceCreateBuffer(WGPUDevice device, WGPUBufferDescriptor const * descriptor);
```

```python
# Python ctypes translation
import ctypes

class WGPUBufferDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.c_void_p),
        ("label", ctypes.c_char_p),
        ("usage", ctypes.c_uint64),  # WGPUBufferUsageFlags
        ("size", ctypes.c_uint64),
        ("mappedAtCreation", ctypes.c_uint32),  # WGPUBool
    ]

# Load the library
lib = ctypes.CDLL(os.environ.get("WEBGPU_LIB_PATH", "libwebgpu_dawn.dylib"))

# Declare function
lib.wgpuDeviceCreateBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(WGPUBufferDescriptor)]
lib.wgpuDeviceCreateBuffer.restype = ctypes.c_void_p
```

## Key WebGPU concepts

- **Instance**: Entry point, created with `wgpuCreateInstance`
- **Adapter**: Represents a GPU, requested async from instance
- **Device**: Logical connection to GPU, requested async from adapter
- **Queue**: For submitting commands, obtained from device
- **Buffer**: GPU memory allocation
- **ShaderModule**: Compiled WGSL code
- **ComputePipeline**: Links shader to execution
- **BindGroup**: Binds buffers to shader parameters
- **CommandEncoder**: Records GPU commands
- **ComputePassEncoder**: Records compute dispatch

## Hints

- Look at other backends (CUDA, Metal, OpenCL) for patterns
- WebGPU uses WGSL shader language (similar to HLSL/GLSL)
- WGSLRenderer extends CStyleLanguage — most rendering logic is inherited
- WebGPU has async patterns (adapter request, device request, buffer mapping)
- Use `wgpuInstanceProcessEvents` or callbacks for async operations

## WGSL specifics

- Types: f32, i32, u32, bool, f16 (with `enable f16;`)
- Buffer bindings: `@group(0) @binding(N) var<storage,read_write> buf: array<f32>;`
- Workgroup size: `@workgroup_size(X,Y,Z)`
- Built-ins: `@builtin(workgroup_id)`, `@builtin(local_invocation_id)`
- No native 8/16-bit types except f16 — need packed load/store for char/short

## Do not modify

- `test/` directory
- Core tinygrad infrastructure (tensor.py, device.py, etc.)
