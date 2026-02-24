"""Simple Modal image build test - run with: modal run examples/benchmark/test_modal_simple.py"""

import modal

app = modal.App("test-sglang-image")

# SGLang v0.5.9 pinned to specific commit
SGLANG_COMMIT = "bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"  # v0.5.9
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
CUDA_VERSION = "12.6.3"

# Use nvidia/cuda devel image for nvcc (SGLang JIT needs it)
# Install uv first, then use it with --index-strategy unsafe-best-match
image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "curl")
    .run_commands(
        # Install uv
        "curl -LsSf https://astral.sh/uv/install.sh | sh && "
        "export PATH=$HOME/.local/bin:$PATH && "
        "uv pip install --system --compile-bytecode "
        "--index-strategy unsafe-best-match "
        f"--extra-index-url {TORCH_INDEX} "
        f"torch 'sglang[srt] @ git+https://github.com/sgl-project/sglang.git@{SGLANG_COMMIT}#subdirectory=python'"
    )
)


@app.function(image=image, gpu="A10G", timeout=60)
def test_import() -> str:
    import sglang

    return f"sglang version: {sglang.__version__}"


@app.local_entrypoint()
def main() -> None:
    print("Testing sglang import...")
    result = test_import.remote()
    print(result)
