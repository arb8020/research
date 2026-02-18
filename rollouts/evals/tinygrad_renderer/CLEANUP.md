# Cleanup Notes

The vendored tinygrad copies are still ~22MB each. Further reductions possible:

## Already removed
- `extra/` (17MB) - HIP/AMD/NV GPU drivers, assembly
- `docs/` (3.4MB) - PDFs, showcase images
- `.ruff_cache/` - linter cache

## Could still remove

### `tinygrad/runtime/autogen/` NVIDIA/AMD bindings (~5MB)
Not needed for Metal/WebGPU eval:
- `nv_580.py` (1.3MB)
- `nv_570.py` (1.3MB)
- `mesa.py` (914K)
- `amd_gpu.py` (638K)

### `examples/` (~4MB)
Large images that aren't needed:
- `sdxl_seed0.png` (1.5MB)
- Various other demo images

Keep the code examples since agents might reference them for patterns.

### `test/` (~2.7MB)
Task says "do not modify" but tests aren't actually run by the eval. Could remove if we update TASK.md.

## Total potential savings
~10-12MB more per copy if we remove the above.
