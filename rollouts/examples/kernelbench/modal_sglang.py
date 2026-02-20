"""Modal SGLang inference endpoint.

Deploy a model on Modal with SGLang serving, get an OpenAI-compatible API.

Usage:
    # Deploy (creates persistent endpoint)
    modal deploy examples/kernelbench/modal_sglang.py

    # Run ephemerally (for testing)
    modal run examples/kernelbench/modal_sglang.py

    # After deploy, URL will be printed. Something like:
    # https://arb8020--sglang-serve-serve.modal.run

Then point your eval config at it:
    endpoint = Endpoint(
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        base_url="https://arb8020--sglang-serve-serve.modal.run/v1",
        api_format="openai-chat",
    )
"""

import subprocess

import modal

# Model to serve - change this
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
GPU_TYPE = "A100"  # A10G, L40S, A100, H100

app = modal.App("sglang-serve")

# Image with SGLang and dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.4",
        "sglang[all]",
        "transformers>=4.50",
        "accelerate",
        "flashinfer-python",  # For fast attention
        index_url="https://download.pytorch.org/whl/cu124",
        extra_index_url="https://pypi.org/simple",
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    })
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=3600,  # 1 hour
    scaledown_window=300,  # Scale down after 5 min idle
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8000, startup_timeout=180)
def serve() -> None:
    """Start SGLang server and expose OpenAI-compatible API."""
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_ID,
        "--port",
        "8000",
        "--host",
        "0.0.0.0",
        "--mem-fraction-static",
        "0.85",
    ]
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main() -> None:
    """Print deployment info."""
    print(f"Model: {MODEL_ID}")
    print(f"GPU: {GPU_TYPE}")
    print()
    print("Deploy with: modal deploy examples/kernelbench/modal_sglang.py")
    print()
    print("After deploying, your endpoint will be at:")
    print("  https://arb8020--sglang-serve-serve.modal.run/v1")
    print()
    print("Test with:")
    print("  curl -X POST https://arb8020--sglang-serve-serve.modal.run/v1/chat/completions \\")
    print('    -H "Content-Type: application/json" \\')
    print(
        f'    -d \'{{"model": "{MODEL_ID}", "messages": [{{"role": "user", "content": "Hello"}}]}}\''
    )
