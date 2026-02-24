"""Pre-defined Modal images for benchmarking.

These images are defined at module level so Modal can cache them properly.
Import and use directly rather than building dynamically.

Usage:
    from rollouts.inference.benchmark.images import SGLANG_IMAGE, ENGINE_V2_IMAGE

    sandbox = modal.Sandbox.create(image=SGLANG_IMAGE, ...)
"""

import modal

# Pin versions to avoid dependency hell
# SGLang v0.5.9 released 2026-02-20, tested working with torch 2.6.0 + cu126
SGLANG_COMMIT = "bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"  # v0.5.9
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
CUDA_VERSION = "12.6.3"

# SGLang image - pinned to specific commit for reproducibility
# Uses nvidia/cuda devel image because SGLang needs nvcc for JIT compilation
# Use run_commands with uv to get --index-strategy unsafe-best-match
# (needed because flashinfer requires packaging>=24.2 which isn't on pytorch index)
SGLANG_IMAGE = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "curl", "libnuma1")  # libnuma1 needed by sgl_kernel
    .run_commands(
        # Install uv first (not available in nvidia/cuda base)
        "curl -LsSf https://astral.sh/uv/install.sh | sh && "
        "export PATH=$HOME/.local/bin:$PATH && "
        "uv pip install --system --compile-bytecode "
        "--index-strategy unsafe-best-match "
        f"--extra-index-url {TORCH_INDEX} "
        f"torch 'sglang[srt] @ git+https://github.com/sgl-project/sglang.git@{SGLANG_COMMIT}#subdirectory=python'"
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

# engine_v2 image
# Uses run_commands with uv for --index-strategy unsafe-best-match
# (flashinfer requires packaging>=24.2 which isn't on pytorch index)
ENGINE_V2_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "uv pip install --system --compile-bytecode "
        "--index-strategy unsafe-best-match "
        f"--extra-index-url {TORCH_INDEX} "
        "torch==2.6.0 numpy 'transformers>=5.0' httpx safetensors triton "
        "'flashinfer-python>=0.3' fastapi trio trio_asyncio uvicorn"
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)
