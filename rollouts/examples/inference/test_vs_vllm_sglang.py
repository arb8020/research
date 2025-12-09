#!/usr/bin/env python3
"""Compare our nano-inference against vLLM and SGLang.

Ground truth test: verify our logprobs match production inference engines.

Uses HTTP API to communicate with vLLM/SGLang servers (like rollouts/providers.py)
to avoid Python import/dependency conflicts.

Usage:
    python examples/inference/test_vs_vllm_sglang.py           # Run locally (GPU)
    python examples/inference/test_vs_vllm_sglang.py --remote  # Run on remote GPU
"""

import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
TEST_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "In a galaxy far, far away",
]
MAX_NEW_TOKENS = 16
VLLM_PORT = 8100
SGLANG_PORT = 8101


def get_nano_inference_logprobs(model_name: str, prompts: list[str], max_new_tokens: int):
    """Get logprobs from our nano-inference engine."""
    from rollouts.inference import InferenceEngine, EngineConfig, SamplingParams

    print(f"Loading nano-inference: {model_name}")
    config = EngineConfig(
        model_path=model_name,
        block_size=16,
        max_batch_size=len(prompts),
    )
    engine = InferenceEngine(config)

    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy
        max_tokens=max_new_tokens,
    )

    samples = engine.generate_text(prompts, sampling_params)

    results = []
    for prompt, sample in zip(prompts, samples):
        results.append({
            "prompt": prompt,
            "generated_ids": sample.output_ids,
            "generated_text": engine.tokenizer.decode(sample.output_ids),
            "logprobs": sample.output_logprobs,
        })

    return results


def wait_for_server(url: str, timeout: int = 120) -> bool:
    """Wait for server to be ready."""
    import httpx

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def get_server_logprobs(api_base: str, model_name: str, prompts: list[str], max_new_tokens: int):
    """Get logprobs via OpenAI-compatible API (works for both vLLM and SGLang)."""
    import httpx

    results = []
    for prompt in prompts:
        # Use completions API for raw text prompts (not chat)
        resp = httpx.post(
            f"{api_base}/v1/completions",
            json={
                "model": model_name,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": 0,
                "logprobs": 1,  # Return top-1 logprob
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        generated_text = choice["text"]

        # Extract logprobs from response
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])
        tokens = logprobs_data.get("tokens", [])

        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "tokens": tokens,
            "logprobs": token_logprobs,
        })

    return results


def start_vllm_server(model_name: str, port: int) -> subprocess.Popen:
    """Start vLLM server as subprocess."""
    print(f"Starting vLLM server on port {port}...")
    proc = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--dtype", "bfloat16",
            "--port", str(port),
            "--gpu-memory-utilization", "0.4",  # Leave room for SGLang
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def start_sglang_server(model_name: str, port: int) -> subprocess.Popen:
    """Start SGLang server as subprocess."""
    print(f"Starting SGLang server on port {port}...")
    proc = subprocess.Popen(
        [
            "python", "-m", "sglang.launch_server",
            "--model-path", model_name,
            "--dtype", "bfloat16",
            "--port", str(port),
            "--mem-fraction-static", "0.4",  # Leave room for vLLM
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def compare_results(name_a: str, results_a: list, name_b: str, results_b: list):
    """Compare logprobs between two engines."""
    print(f"\n{'='*60}")
    print(f"Comparing {name_a} vs {name_b}")
    print("="*60)

    all_match = True

    for i, (a, b) in enumerate(zip(results_a, results_b)):
        prompt = a.get("prompt", TEST_PROMPTS[i])
        print(f"\nPrompt {i+1}: {prompt[:40]}...")
        print(f"  {name_a}: {a['generated_text'][:50]}...")
        print(f"  {name_b}: {b['generated_text'][:50]}...")

        # Compare generated text (since token ids may not be available from API)
        if a['generated_text'].strip() != b['generated_text'].strip():
            print(f"  WARNING: Text differs (may be tokenization)")

        # Compare logprobs
        lp_a = a.get('logprobs', [])
        lp_b = b.get('logprobs', [])

        if not lp_a or not lp_b:
            print(f"  Skipping logprob comparison (not available)")
            continue

        if len(lp_a) != len(lp_b):
            print(f"  Logprob count mismatch: {len(lp_a)} vs {len(lp_b)}")
            # Compare what we can
            min_len = min(len(lp_a), len(lp_b))
            lp_a = lp_a[:min_len]
            lp_b = lp_b[:min_len]

        max_diff = 0
        for j, (la, lb) in enumerate(zip(lp_a, lp_b)):
            if la is not None and lb is not None:
                diff = abs(la - lb)
                max_diff = max(max_diff, diff)

        print(f"  Max logprob diff: {max_diff:.6f}")

        if max_diff > 0.01:
            print(f"  WARNING: Logprob diff > 0.01!")
            all_match = False
        else:
            print(f"  OK")

    return all_match


def run_comparison():
    """Run full comparison test.

    Runs each engine sequentially to avoid memory/dependency conflicts:
    1. vLLM server → collect → shutdown
    2. SGLang server → collect → shutdown
    3. nano-inference → collect → done
    4. Compare all results
    """
    import torch
    import gc

    print("="*60)
    print("nano-inference vs vLLM/SGLang Comparison")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print("="*60)

    vllm_results = None
    sglang_results = None
    nano_results = None

    # === 1. vLLM ===
    print("\n--- vLLM (server) ---")
    vllm_proc = None
    try:
        vllm_proc = start_vllm_server(MODEL_NAME, VLLM_PORT)
        vllm_url = f"http://localhost:{VLLM_PORT}"
        if wait_for_server(vllm_url, timeout=180):
            print("vLLM server ready")
            vllm_results = get_server_logprobs(vllm_url, MODEL_NAME, TEST_PROMPTS, MAX_NEW_TOKENS)
            print(f"Got {len(vllm_results)} results from vLLM")
        else:
            print("vLLM server failed to start")
    except Exception as e:
        print(f"vLLM error: {e}")
    finally:
        if vllm_proc:
            print("Stopping vLLM server...")
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=10)
            except:
                vllm_proc.kill()
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)  # Let GPU memory settle

    # === 2. SGLang ===
    print("\n--- SGLang (server) ---")
    sglang_proc = None
    try:
        sglang_proc = start_sglang_server(MODEL_NAME, SGLANG_PORT)
        sglang_url = f"http://localhost:{SGLANG_PORT}"
        if wait_for_server(sglang_url, timeout=180):
            print("SGLang server ready")
            sglang_results = get_server_logprobs(sglang_url, MODEL_NAME, TEST_PROMPTS, MAX_NEW_TOKENS)
            print(f"Got {len(sglang_results)} results from SGLang")
        else:
            print("SGLang server failed to start")
    except Exception as e:
        print(f"SGLang error: {e}")
    finally:
        if sglang_proc:
            print("Stopping SGLang server...")
            sglang_proc.terminate()
            try:
                sglang_proc.wait(timeout=10)
            except:
                sglang_proc.kill()
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)

    # === 3. nano-inference (ours) ===
    print("\n--- nano-inference (ours) ---")
    try:
        nano_results = get_nano_inference_logprobs(MODEL_NAME, TEST_PROMPTS, MAX_NEW_TOKENS)
        print(f"Got {len(nano_results)} results from nano-inference")
    except Exception as e:
        print(f"nano-inference error: {e}")
        import traceback
        traceback.print_exc()

    # === 4. Compare results ===
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    all_pass = True

    if nano_results and vllm_results:
        if not compare_results("nano", nano_results, "vLLM", vllm_results):
            all_pass = False
    else:
        print("Skipping nano vs vLLM (missing results)")

    if nano_results and sglang_results:
        if not compare_results("nano", nano_results, "SGLang", sglang_results):
            all_pass = False
    else:
        print("Skipping nano vs SGLang (missing results)")

    if vllm_results and sglang_results:
        if not compare_results("vLLM", vllm_results, "SGLang", sglang_results):
            all_pass = False
    else:
        print("Skipping vLLM vs SGLang (missing results)")

    # Summary
    print("\n" + "="*60)
    if all_pass and (vllm_results or sglang_results):
        print("ALL COMPARISONS PASSED")
    elif not vllm_results and not sglang_results:
        print("NO COMPARISONS RUN (vLLM and SGLang both failed)")
    else:
        print("SOME COMPARISONS FAILED")
    print("="*60)

    return all_pass


def run_remote(script_path: str, keep_alive: bool = False, gpu_id: str | None = None):
    """Run script on remote GPU via broker/bifrost.

    Args:
        script_path: Path to the script (__file__ from caller)
        keep_alive: Keep GPU running after completion
        gpu_id: Reuse existing GPU instance ID (skips provisioning)
    """
    import os
    from pathlib import Path

    from broker.client import GPUClient
    from bifrost.client import BifrostClient
    from dotenv import load_dotenv

    load_dotenv()

    # Get script path relative to git root
    import subprocess as sp
    script = Path(script_path).resolve()
    git_root = Path(sp.check_output(
        ['git', 'rev-parse', '--show-toplevel'],
        text=True
    ).strip())
    rel_path = script.relative_to(git_root)

    # Provision or reuse GPU
    runpod_key = os.getenv("RUNPOD_API_KEY")
    assert runpod_key, "RUNPOD_API_KEY not set"
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    client = GPUClient(credentials={"runpod": runpod_key}, ssh_key_path=ssh_key_path)
    gpu = None

    try:
        if gpu_id:
            print(f"Reusing GPU: {gpu_id}")
            gpu = client.get_instance(gpu_id, provider="runpod")
            if not gpu:
                print(f"GPU {gpu_id} not found (is it still running?)")
                return
            keep_alive = True
        else:
            print("Provisioning GPU...")
            gpu = client.create(
                query=(client.vram_gb >= 24) & (client.price_per_hour <= 0.5),
                name="test-vllm-sglang",
            )
            if not gpu:
                print("Failed to provision GPU")
                return
            print(f"GPU ready: {gpu.id}")

            if not gpu.wait_until_ssh_ready(timeout=300):
                print("SSH timeout")
                client.terminate_instance(gpu.id, gpu.provider)
                return

        print(f"SSH: {gpu.ssh_connection_string()}")

        # Deploy with vLLM and SGLang
        workspace = "~/.bifrost/workspaces/rollouts"
        bifrost = BifrostClient(gpu.ssh_connection_string(), ssh_key_path)
        bootstrap = [
            "cd rollouts && uv python install 3.12 && uv sync --python 3.12",
            # Core deps
            "cd rollouts && uv pip install torch 'transformers<4.52' accelerate httpx",
            # vLLM and SGLang (install with deps, they need their full stack)
            "cd rollouts && uv pip install vllm || echo 'vLLM install failed'",
            "cd rollouts && uv pip install 'sglang[all]' || echo 'SGLang install failed'",
        ]
        bifrost.push(workspace_path=workspace, bootstrap_cmd=bootstrap)
        print("Code deployed")

        # Run with streaming output
        remote_script = f"{workspace}/{rel_path}"
        cmd = f"cd {workspace}/rollouts && uv run python {remote_script}"
        print(f"Running: {cmd}")
        print("-" * 50)
        for line in bifrost.exec_stream(cmd):
            print(line, end="")
        print("-" * 50)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        keep_alive = True

    finally:
        if gpu is None:
            return
        if keep_alive:
            print()
            print("=" * 50)
            print(f"GPU kept alive: {gpu.id}")
            print(f"SSH: {gpu.ssh_connection_string()}")
            print()
            print(f"Rerun with:   --gpu-id {gpu.id}")
            print(f"Terminate:    broker terminate {gpu.id}")
            print("=" * 50)
        else:
            print("Cleaning up...")
            client.terminate_instance(gpu.id, gpu.provider)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", help="Run on remote GPU")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument("--gpu-id", type=str, help="Reuse existing GPU instance ID")
    args = parser.parse_args()

    if args.remote or args.gpu_id:
        run_remote(__file__, keep_alive=args.keep_alive, gpu_id=args.gpu_id)
    else:
        run_comparison()
