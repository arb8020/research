# Metal backend - STUBBED for agent restoration exercise
#
# Original implementation had:
# - MetalDevice: GPU device management, command queues
# - MetalCompiler: Compiles Metal shader source to MTLB binary
# - MetalProgram: Loads and executes compiled shaders
# - MetalAllocator: GPU buffer allocation
# - MetalBuffer: Buffer wrapper
#
# The agent must implement these to restore Metal support.
# See tinygrad/renderer/cstyle.py for MetalRenderer (also stubbed).

from tinygrad.device import Compiled, Compiler, LRUAllocator

class MetalDevice(Compiled):
    def __init__(self, device: str):
        raise NotImplementedError("TODO: implement MetalDevice - manages Metal GPU device and command queues")

    def synchronize(self):
        raise NotImplementedError("TODO: implement synchronize - waits for all GPU work to complete")

class MetalCompiler(Compiler):
    def __init__(self):
        raise NotImplementedError("TODO: implement MetalCompiler - compiles Metal source to MTLB binary")

    def compile(self, src: str) -> bytes:
        raise NotImplementedError("TODO: implement compile - takes Metal source, returns MTLB bytes")

class MetalProgram:
    def __init__(self, dev, name: str, lib: bytes, **kwargs):
        raise NotImplementedError("TODO: implement MetalProgram - loads compiled shader and creates pipeline")

    def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
        raise NotImplementedError("TODO: implement __call__ - dispatches compute kernel")

class MetalBuffer:
    def __init__(self, buf, size: int, offset=0):
        raise NotImplementedError("TODO: implement MetalBuffer - wraps MTLBuffer")

class MetalAllocator(LRUAllocator):
    def _alloc(self, size: int, options):
        raise NotImplementedError("TODO: implement _alloc - allocates GPU buffer")

    def _free(self, opaque, options):
        raise NotImplementedError("TODO: implement _free - frees GPU buffer")

    def _as_buffer(self, src) -> memoryview:
        raise NotImplementedError("TODO: implement _as_buffer - returns CPU-accessible view of GPU buffer")

    def _copyin(self, dest, src: memoryview):
        raise NotImplementedError("TODO: implement _copyin - copies data from CPU to GPU")

    def _copyout(self, dest: memoryview, src):
        raise NotImplementedError("TODO: implement _copyout - copies data from GPU to CPU")
