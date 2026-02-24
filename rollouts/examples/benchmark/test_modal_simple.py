"""Simple Modal image build test - run with: modal run examples/benchmark/test_modal_simple.py"""

import modal

app = modal.App("test-sglang-image")

# SGLang image - pin sglang and let it pick its torch
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "sglang[srt]==0.4.6.post5",
        extra_index_url="https://download.pytorch.org/whl/cu126",
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
