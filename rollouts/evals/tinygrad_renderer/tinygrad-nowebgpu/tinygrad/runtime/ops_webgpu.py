# WebGPU backend - STUBBED for agent restoration exercise
#
# Original implementation had:
# - WebGpuDevice: GPU device management via wgpu-native FFI
# - WebGPUProgram: Loads WGSL shaders and dispatches compute kernels
# - WebGpuAllocator: GPU buffer allocation
#
# The agent must implement these to restore WebGPU support.
# See tinygrad/renderer/wgsl.py for WGSLRenderer (also stubbed).
# See tinygrad/runtime/autogen/webgpu.py for wgpu-native bindings (also stubbed).

from tinygrad.device import Compiled, Allocator, BufferSpec

class WebGpuDevice(Compiled):
    def __init__(self, device: str):
        raise NotImplementedError("TODO: implement WebGpuDevice - manages WebGPU device via wgpu-native")

    def synchronize(self):
        raise NotImplementedError("TODO: implement synchronize - waits for all GPU work to complete")

class WebGPUProgram:
    def __init__(self, dev, name: str, lib: bytes, **kwargs):
        raise NotImplementedError("TODO: implement WebGPUProgram - loads WGSL shader and creates compute pipeline")

    def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
        raise NotImplementedError("TODO: implement __call__ - dispatches compute kernel")

class WebGpuAllocator(Allocator):
    def _alloc(self, size: int, options: BufferSpec):
        raise NotImplementedError("TODO: implement _alloc - allocates GPU buffer")

    def _free(self, opaque, options: BufferSpec):
        raise NotImplementedError("TODO: implement _free - frees GPU buffer")

    def _copyin(self, dest, src: memoryview):
        raise NotImplementedError("TODO: implement _copyin - copies data from CPU to GPU")

    def _copyout(self, dest: memoryview, src):
        raise NotImplementedError("TODO: implement _copyout - copies data from GPU to CPU")
