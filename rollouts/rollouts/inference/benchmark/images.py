"""Pre-defined Modal images for benchmarking.

These images are defined at module level so Modal can cache them properly.
Import and use directly rather than building dynamically.

Usage:
    from rollouts.inference.benchmark.images import SGLANG_IMAGE, ENGINE_V2_IMAGE

    sandbox = modal.Sandbox.create(image=SGLANG_IMAGE, ...)
"""

import modal

# SGLang image - let sglang pick its torch version
# Using sglang 0.4.6.post5 which worked earlier (needs torch 2.6.0)
SGLANG_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "sglang[srt]==0.4.6.post5",  # Pin to known working version
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
)

# engine_v2 image
ENGINE_V2_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.6.0",
        "numpy",
        "transformers>=5.0",
        "httpx",
        "safetensors",
        "triton",
        "flashinfer-python>=0.3",
        "fastapi",
        "uvicorn",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)
