# metal_restore eval — workspace setup

The tinygrad task workspace is not stored in this repo. It was previously
a full clone of tinygrad (~17MB) with the Metal backend stubbed out, which
is impractical to vendor.

## Intended approach (not yet implemented)

Store a patch file (`metal_restore.patch`) and a pinned upstream tinygrad
commit SHA. At eval runtime:

```bash
TINYGRAD_REV=<sha>
git clone https://github.com/tinygrad/tinygrad tinygrad-nometal
git -C tinygrad-nometal checkout $TINYGRAD_REV
git -C tinygrad-nometal apply ../metal_restore.patch
```

The patch is small (just stubs out ops_metal.py, graph/metal.py,
autogen/metal.py, and the MetalRenderer in renderer/cstyle.py) and the
full tinygrad tree is fetched on demand, not committed here.

## What the patch stubs out

From TASK.md:
- `tinygrad/runtime/ops_metal.py` — MetalDevice, MetalCompiler, MetalProgram, MetalAllocator
- `tinygrad/runtime/graph/metal.py` — MetalGraph
- `tinygrad/runtime/autogen/metal.py` — ObjC bindings
- `tinygrad/renderer/cstyle.py` — MetalRenderer class

## To regenerate

1. Clone tinygrad at the desired commit
2. Apply your stubs manually
3. Run `git diff HEAD > metal_restore.patch` from inside the tinygrad clone
4. Commit the patch file here alongside eval.py
