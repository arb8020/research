#!/usr/bin/env python3
"""Run reverse text eval on Modal with SGLang.

Usage:
    python examples/eval/reverse_text/modal_eval.py
    python examples/eval/reverse_text/modal_eval.py --model Qwen/Qwen2.5-7B-Instruct --limit 10
"""

import modal

app = modal.App("reverse-text-eval")

# Image with SGLang and dependencies
# Note: SGLang pins transformers==4.57.1 but we need >=5.0 for huggingface_hub compat
# Install SGLang first, then force-upgrade transformers
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "git")
    .pip_install(
        "sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python",
    )
    .pip_install(
        "transformers>=5.0.0",
        "huggingface_hub>=1.4.0",
        "httpx",
        "openai",
        force=True,
    )
    .env({"HF_HOME": "/cache/huggingface"})
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=900,
    volumes={"/cache": modal.Volume.from_name("hf-cache", create_if_missing=True)},
)
def run_sglang_eval(model_name: str, num_samples: int) -> dict:
    """Start SGLang server and run reverse text eval."""
    import re
    import subprocess
    import time
    from difflib import SequenceMatcher

    import httpx

    # Start SGLang server
    print(f"Starting SGLang server with {model_name}...")
    server_proc = subprocess.Popen(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model",
            model_name,
            "--port",
            "30000",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for server to be ready
    base_url = "http://localhost:30000"
    print("Waiting for server to start...")
    for i in range(180):  # 3 minutes timeout
        try:
            resp = httpx.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                print(f"SGLang server ready after {i}s")
                break
        except Exception:
            pass

        # Check if process died
        if server_proc.poll() is not None:
            stdout, _ = server_proc.communicate()
            print(f"Server died with output:\n{stdout[-2000:]}")
            raise RuntimeError("SGLang server crashed during startup")

        time.sleep(1)
    else:
        server_proc.kill()
        raise TimeoutError("SGLang server failed to start within 3 minutes")

    # Tasks
    tasks = [
        {"text": "hello"},
        {"text": "world"},
        {"text": "python"},
        {"text": "machine learning"},
        {"text": "artificial intelligence"},
        {"text": "deep neural network"},
        {"text": "transformer architecture"},
        {"text": "attention mechanism"},
        {"text": "gradient descent"},
        {"text": "backpropagation"},
        {"text": "natural language"},
        {"text": "computer vision"},
        {"text": "reinforcement learning"},
        {"text": "neural network"},
        {"text": "deep learning"},
    ][:num_samples]

    # Run eval using OpenAI client
    from openai import OpenAI

    client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy")

    results = []
    for i, task in enumerate(tasks):
        text = task["text"]
        expected = text[::-1]

        prompt = (
            f"Reverse the following text character-by-character. "
            f"Put your answer in <reversed_text> tags.\n\n"
            f"Text to reverse: {text}"
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )

            content = response.choices[0].message.content or ""

            # Parse response
            match = re.search(r"<reversed_text>\s*(.*?)\s*</reversed_text>", content, re.DOTALL)
            parsed = match.group(1).strip() if match else content.strip()

            exact_match = parsed == expected
            similarity = SequenceMatcher(None, parsed, expected).ratio()

            results.append({
                "text": text,
                "expected": expected,
                "parsed": parsed,
                "exact_match": exact_match,
                "similarity": similarity,
            })

            status = "✓" if exact_match else "✗"
            print(
                f"[{i + 1}/{len(tasks)}] {status} '{text}' -> '{parsed}' (expected: '{expected}')"
            )

        except Exception as e:
            print(f"[{i + 1}/{len(tasks)}] Error: {e}")
            results.append({
                "text": text,
                "error": str(e),
                "exact_match": False,
                "similarity": 0.0,
            })

    # Cleanup
    print("\nShutting down server...")
    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()

    # Compute summary
    exact_matches = sum(1 for r in results if r.get("exact_match", False))
    mean_similarity = sum(r.get("similarity", 0) for r in results) / len(results)

    return {
        "model": model_name,
        "total_samples": len(results),
        "exact_matches": exact_matches,
        "accuracy": exact_matches / len(results),
        "mean_similarity": mean_similarity,
        "results": results,
    }


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-3B-Instruct", limit: int = 5) -> None:
    """Run the eval."""
    print("Running reverse text eval on Modal")
    print(f"  Model: {model}")
    print(f"  Samples: {limit}")
    print()

    result = run_sglang_eval.remote(model, limit)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model: {result['model']}")
    print(f"  Samples: {result['total_samples']}")
    print(f"  Exact matches: {result['exact_matches']}")
    print(f"  Accuracy: {result['accuracy']:.1%}")
    print(f"  Mean similarity: {result['mean_similarity']:.3f}")
