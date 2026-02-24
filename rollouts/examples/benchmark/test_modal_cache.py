"""Test Modal image caching behavior.

Run twice - second run should be much faster if caching works.

Usage:
    time python examples/benchmark/test_modal_cache.py
"""

import time

import modal

# Define image using uv_pip_install for faster builds
SGLANG_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # sglang needs git
    .uv_pip_install(
        "torch==2.6.0",
        "numpy",
        "transformers>=5.0",
        "httpx",
        "sglang[srt]>=0.4",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

app = modal.App.lookup("benchmark-cache-test", create_if_missing=True)


def main() -> None:
    # Force output to show build logs
    import logging

    logging.basicConfig(level=logging.DEBUG)

    print("Creating sandbox with sglang image...", flush=True)
    start = time.time()

    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            app=app,
            image=SGLANG_IMAGE,
            gpu="A10G",  # Cheaper GPU for test
            timeout=60,
        )
        try:
            elapsed = time.time() - start
            print(f"Sandbox created in {elapsed:.1f}s")

            # Quick check
            proc = sandbox.exec("python", "-c", "import sglang; print('sglang imported')")
            print(proc.stdout.read())
        finally:
            sandbox.terminate()


if __name__ == "__main__":
    main()
