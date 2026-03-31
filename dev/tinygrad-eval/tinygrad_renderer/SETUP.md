# tinygrad_renderer eval — workspace setup

The tinygrad task workspaces (`tinygrad-nometal/`, `tinygrad-nowebgpu/`) are
not stored in this repo. They were previously full tinygrad clones (~17MB
each) with GPU backend renderers stubbed out, which is impractical to vendor.

## Intended approach (not yet implemented)

Store patch files and a pinned upstream tinygrad commit SHA. At eval runtime:

```bash
TINYGRAD_REV=<sha>
BACKEND=metal  # or webgpu

git clone https://github.com/tinygrad/tinygrad tinygrad-no${BACKEND}
git -C tinygrad-no${BACKEND} checkout $TINYGRAD_REV
git -C tinygrad-no${BACKEND} apply ../tinygrad-no${BACKEND}.patch
```

Each patch is small (stubs out only the renderer/backend files for that
backend) and tinygrad is fetched on demand, not committed here.

## Backends

- `tinygrad-nometal.patch` — stubs out Metal renderer (macOS GPU backend)
- `tinygrad-nowebgpu.patch` — stubs out WebGPU renderer

## To regenerate

1. Clone tinygrad at the desired commit
2. Apply your stubs manually to the target backend files
3. Run `git diff HEAD > tinygrad-no<backend>.patch` from inside the clone
4. Commit the patch file here alongside eval.py
