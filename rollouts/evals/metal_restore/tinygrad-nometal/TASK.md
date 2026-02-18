# Task: Restore Metal Backend

The Metal backend has been stubbed out. Your task is to restore it.

## What's been removed

1. **`tinygrad/runtime/ops_metal.py`** — Device, compiler, program, allocator
2. **`tinygrad/runtime/graph/metal.py`** — Graph execution batching
3. **`tinygrad/runtime/autogen/metal.py`** — ObjC bindings for Metal framework
4. **`tinygrad/renderer/cstyle.py`** — MetalRenderer class (generates Metal shader code)

## Success criteria

```bash
# This should work:
PYTHONPATH=. METAL=1 python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"

# Tests should pass:
PYTHONPATH=. METAL=1 python -m pytest test/device/test_metal.py -xvs
```

## Hints

- Look at other backends (CUDA, OpenCL) for patterns
- The original tinygrad repo has working implementations
- MetalRenderer extends CStyleLanguage — most rendering logic is inherited
- You'll need ObjC bindings for Metal framework (see `runtime/support/objc.py`)

## Key classes to implement

| Class | File | Purpose |
|-------|------|---------|
| MetalDevice | ops_metal.py | Manages GPU device and command queues |
| MetalCompiler | ops_metal.py | Compiles Metal source to MTLB binary |
| MetalProgram | ops_metal.py | Loads and executes compiled shaders |
| MetalAllocator | ops_metal.py | GPU buffer allocation |
| MetalBuffer | ops_metal.py | Buffer wrapper |
| MetalRenderer | cstyle.py | Generates Metal shader code |
| MetalGraph | graph/metal.py | Batches kernel executions |

## Do not modify

- `test/` directory
- Core tinygrad infrastructure (tensor.py, device.py, etc.)
