#!/usr/bin/env python3
"""Test Modal image build with output."""

import modal

modal.enable_output()

# Simple image to test
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "torch>=2.4",
        "transformers>=5.0",
        "accelerate",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .run_commands("echo 'Test build v1'")
)

app = modal.App("test-rollouts-build")


@app.function(image=image, gpu="A10G")
def test() -> str:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    return "OK"


if __name__ == "__main__":
    with app.run():
        result = test.remote()
        print(f"Result: {result}")
