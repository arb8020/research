#!/usr/bin/env python3
"""Test engine_v2 on GPU.

Tests:
1. Basic generation (smoke test)
2. Logprobs (greedy and sampling)
3. Batched generation
4. FlashAttention backend (if available)
5. CUDA graphs (if available)

Run locally (CPU fallback):
    python examples/inference/test_engine_v2.py

Run on RunPod:
    python examples/inference/test_engine_v2.py --provision --provider runpod

Reuse existing pod:
    python examples/inference/test_engine_v2.py --node-id runpod:<id>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Wide event logging pattern from logging_sucks.md
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    max_batch_size: int = 8
    max_seq_len: int = 512
    num_tokens: int = 20


def emit_event(event: str, **data: Any) -> None:
    """Emit structured log event (wide event pattern)."""
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event": event,
        **data,
    }
    print(json.dumps(entry), flush=True)


def test_basic_generation(engine, config: TestConfig) -> dict:
    """Test 1: Basic generation smoke test."""
    from rollouts.inference.core import SamplingParams

    emit_event("test_start", test="basic_generation")
    start = time.perf_counter()

    params = SamplingParams(max_tokens=config.num_tokens, temperature=0.7)
    uid = engine.add_request("Hello, my name is", params)
    finished = engine.run_to_completion()

    duration_ms = (time.perf_counter() - start) * 1000
    req = finished[0]
    output_tokens = req.input_ids.tolist()
    num_generated = len(output_tokens) - 4  # "Hello, my name is" is ~4 tokens

    result = {
        "test": "basic_generation",
        "success": len(finished) == 1 and num_generated > 0,
        "duration_ms": duration_ms,
        "num_generated": num_generated,
        "tokens_per_sec": num_generated / (duration_ms / 1000) if duration_ms > 0 else 0,
    }
    emit_event("test_done", **result)
    return result


def test_logprobs_greedy(engine, config: TestConfig) -> dict:
    """Test 2a: Logprobs with greedy decoding."""
    from rollouts.inference.core import SamplingParams

    emit_event("test_start", test="logprobs_greedy")
    start = time.perf_counter()

    params = SamplingParams(max_tokens=config.num_tokens, temperature=0.0, return_logprobs=True)
    engine.add_request("The capital of France is", params)
    finished = engine.run_to_completion()

    duration_ms = (time.perf_counter() - start) * 1000
    req = finished[0]

    has_logprobs = req.logprobs is not None
    logprobs_valid = False
    if has_logprobs:
        lps = req.logprobs.tolist()
        logprobs_valid = all(lp < 0 for lp in lps)  # All log probs should be negative

    result = {
        "test": "logprobs_greedy",
        "success": has_logprobs and logprobs_valid,
        "duration_ms": duration_ms,
        "has_logprobs": has_logprobs,
        "num_logprobs": len(req.logprobs) if has_logprobs else 0,
        "logprobs_sample": req.logprobs[:5].tolist() if has_logprobs else [],
    }
    emit_event("test_done", **result)
    return result


def test_logprobs_sampling(engine, config: TestConfig) -> dict:
    """Test 2b: Logprobs with temperature sampling."""
    from rollouts.inference.core import SamplingParams

    emit_event("test_start", test="logprobs_sampling")
    start = time.perf_counter()

    params = SamplingParams(max_tokens=config.num_tokens, temperature=0.8, return_logprobs=True)
    engine.add_request("Once upon a time", params)
    finished = engine.run_to_completion()

    duration_ms = (time.perf_counter() - start) * 1000
    req = finished[0]

    has_logprobs = req.logprobs is not None
    logprobs_valid = False
    if has_logprobs:
        lps = req.logprobs.tolist()
        logprobs_valid = all(lp < 0 for lp in lps)

    result = {
        "test": "logprobs_sampling",
        "success": has_logprobs and logprobs_valid,
        "duration_ms": duration_ms,
        "has_logprobs": has_logprobs,
        "num_logprobs": len(req.logprobs) if has_logprobs else 0,
        "logprobs_sample": req.logprobs[:5].tolist() if has_logprobs else [],
    }
    emit_event("test_done", **result)
    return result


def test_token_input(engine, config: TestConfig) -> dict:
    """Test 3: Token-level input (RL use case)."""
    from rollouts.inference.core import SamplingParams

    emit_event("test_start", test="token_input")
    start = time.perf_counter()

    # RL passes token IDs directly, not strings
    # Use tokenizer to get some real token IDs
    prompt_tokens = engine.tokenizer.encode("The answer is", add_special_tokens=True)

    params = SamplingParams(max_tokens=config.num_tokens, temperature=0.7, return_logprobs=True)
    engine.add_request(prompt_tokens, params)  # Pass token list, not string
    finished = engine.run_to_completion()

    duration_ms = (time.perf_counter() - start) * 1000
    req = finished[0]

    # Verify prompt tokens are preserved
    output_tokens = req.input_ids.tolist()
    prompt_preserved = output_tokens[: len(prompt_tokens)] == prompt_tokens

    result = {
        "test": "token_input",
        "success": len(finished) == 1 and prompt_preserved and req.logprobs is not None,
        "duration_ms": duration_ms,
        "prompt_tokens": len(prompt_tokens),
        "output_tokens": len(output_tokens),
        "prompt_preserved": prompt_preserved,
        "has_logprobs": req.logprobs is not None,
    }
    emit_event("test_done", **result)
    return result


def test_multi_sample_per_prompt(engine, config: TestConfig) -> dict:
    """Test 4: Multiple samples per prompt (RL generates N samples per prompt)."""
    from rollouts.inference.core import SamplingParams

    emit_event("test_start", test="multi_sample_per_prompt")
    start = time.perf_counter()

    # RL typically generates 4-8 samples per prompt
    n_samples = 4
    prompt = "What is 2 + 2?"

    params = SamplingParams(max_tokens=config.num_tokens, temperature=0.8, return_logprobs=True)
    for _ in range(n_samples):
        engine.add_request(prompt, params)

    finished = engine.run_to_completion()
    duration_ms = (time.perf_counter() - start) * 1000

    # All should complete with logprobs
    all_have_logprobs = all(req.logprobs is not None for req in finished)

    # With temperature > 0, outputs should differ (not all identical)
    outputs = [req.input_ids.tolist() for req in finished]
    unique_outputs = len(set(tuple(o) for o in outputs))

    result = {
        "test": "multi_sample_per_prompt",
        "success": len(finished) == n_samples and all_have_logprobs,
        "duration_ms": duration_ms,
        "n_samples": n_samples,
        "n_finished": len(finished),
        "all_have_logprobs": all_have_logprobs,
        "unique_outputs": unique_outputs,
        "diversity": unique_outputs / n_samples,
    }
    emit_event("test_done", **result)
    return result


def test_batched_generation(engine, config: TestConfig) -> dict:
    """Test 5: Batched generation with different prompts."""
    from rollouts.inference.core import SamplingParams

    emit_event("test_start", test="batched_generation")
    start = time.perf_counter()

    prompts = [
        "The meaning of life is",
        "In the beginning",
        "To be or not to be",
        "All happy families",
    ]

    params = SamplingParams(max_tokens=config.num_tokens, temperature=0.7)
    for p in prompts:
        engine.add_request(p, params)

    finished = engine.run_to_completion()
    duration_ms = (time.perf_counter() - start) * 1000

    all_generated = all(len(req.input_ids) > 5 for req in finished)
    total_tokens = sum(len(req.input_ids) for req in finished)

    result = {
        "test": "batched_generation",
        "success": len(finished) == len(prompts) and all_generated,
        "duration_ms": duration_ms,
        "num_prompts": len(prompts),
        "num_finished": len(finished),
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / (duration_ms / 1000) if duration_ms > 0 else 0,
    }
    emit_event("test_done", **result)
    return result


def test_flash_attention(engine, config: TestConfig) -> dict:
    """Test 6: FlashAttention backend."""
    emit_event("test_start", test="flash_attention")

    backend_name = type(engine.attn_backend).__name__
    is_flash = "Flash" in backend_name

    result = {
        "test": "flash_attention",
        "success": True,  # Not a failure if unavailable
        "backend": backend_name,
        "flash_enabled": is_flash,
    }
    emit_event("test_done", **result)
    return result


def test_cuda_graphs(engine, config: TestConfig) -> dict:
    """Test 7: CUDA graphs status."""
    emit_event("test_start", test="cuda_graphs")

    cuda_graphs_enabled = getattr(engine, "_use_cuda_graphs", False)

    result = {
        "test": "cuda_graphs",
        "success": True,  # Not a failure if unavailable
        "cuda_graphs_enabled": cuda_graphs_enabled,
    }
    emit_event("test_done", **result)
    return result


def run_tests(config: TestConfig, require_gpu: bool = True) -> list[dict]:
    """Run all tests and return results."""
    import torch

    from rollouts.inference.engine_v2 import EngineConfig, InferenceEngineV2

    cuda_available = torch.cuda.is_available()

    if require_gpu and not cuda_available:
        emit_event("run_skipped", reason="CUDA not available", require_gpu=require_gpu)
        print("\n⚠️  Skipping tests: CUDA not available (use --cpu to run on CPU anyway)")
        return []

    emit_event(
        "run_start",
        config=asdict(config),
        cuda_available=cuda_available,
        cuda_device=torch.cuda.get_device_name() if cuda_available else None,
    )

    # Create engine
    emit_event("engine_init_start")
    engine_config = EngineConfig(
        model_path=config.model_name,
        max_batch_size=config.max_batch_size,
        max_seq_len=config.max_seq_len,
    )
    engine = InferenceEngineV2(engine_config)
    emit_event(
        "engine_init_done",
        attention_backend=type(engine.attn_backend).__name__,
        cuda_graphs=getattr(engine, "_use_cuda_graphs", False),
        overlap=getattr(engine, "_use_overlap", False),
        radix_cache=getattr(engine, "_use_radix_cache", False),
    )

    # Run tests
    results = []
    tests = [
        test_basic_generation,
        test_logprobs_greedy,
        test_logprobs_sampling,
        test_token_input,
        test_multi_sample_per_prompt,
        test_batched_generation,
        test_flash_attention,
        test_cuda_graphs,
    ]

    for test_fn in tests:
        try:
            result = test_fn(engine, config)
            results.append(result)
        except Exception as e:
            emit_event("test_error", test=test_fn.__name__, error=str(e))
            results.append({"test": test_fn.__name__, "success": False, "error": str(e)})

    # Summary
    passed = sum(1 for r in results if r.get("success", False))
    emit_event(
        "run_done",
        total_tests=len(results),
        passed=passed,
        failed=len(results) - passed,
    )

    return results


def run_remote(node_id: str | None = None, provider: str = "runpod") -> None:
    """Deploy and run on remote GPU."""
    import os

    from dotenv import load_dotenv

    from bifrost.client import BifrostClient
    from broker.client import GPUClient

    load_dotenv()

    script = Path(__file__).resolve()
    import subprocess

    git_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
    rel_path = script.relative_to(git_root)

    runpod_key = os.getenv("RUNPOD_API_KEY")
    assert runpod_key, "RUNPOD_API_KEY not set"
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    client = GPUClient(credentials={"runpod": runpod_key}, ssh_key_path=ssh_key_path)
    gpu = None

    try:
        if node_id:
            print(f"Reusing instance: {node_id}")
            gpu = client.get_instance(node_id, provider=provider)
            if not gpu:
                print(f"GPU {node_id} not found")
                return
        else:
            print(f"Provisioning GPU on {provider}...")
            gpu = client.create(
                query=(client.vram_gb >= 24) & (client.price_per_hour <= 0.5),
                name="inference-test-v2",
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

        # Deploy
        workspace = "~/.bifrost/workspaces/rollouts"
        bifrost = BifrostClient(gpu.ssh_connection_string(), ssh_key_path)
        bootstrap = [
            "cd rollouts && uv python install 3.12 && uv sync --python 3.12",
            "uv pip install torch 'transformers<4.52' accelerate flash-attn sgl-kernel",
        ]
        bifrost.push(workspace_path=workspace, bootstrap_cmd=bootstrap)
        print("Code deployed")

        # Run with streaming output
        remote_script = f"{workspace}/{rel_path}"
        cmd = f"cd {workspace}/rollouts && uv run python {remote_script}"
        print(f"Running: {cmd}")
        print("-" * 60)
        for line in bifrost.exec_stream(cmd):
            print(line, end="")
        print("-" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")

    finally:
        if gpu is not None:
            print()
            print("=" * 60)
            print(f"Instance: {gpu.id}")
            print(f"SSH: {gpu.ssh_connection_string()}")
            print()
            print(f"Rerun with:   --node-id {gpu.id}")
            print(f"Terminate:    broker terminate {gpu.id}")
            print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test engine_v2 on GPU")
    parser.add_argument("--provision", action="store_true", help="Provision remote GPU")
    parser.add_argument("--provider", default="runpod", help="GPU provider (runpod, modal)")
    parser.add_argument("--node-id", help="Reuse existing instance")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Model to test")
    parser.add_argument("--cpu", action="store_true", help="Allow running on CPU (for local testing)")
    args = parser.parse_args()

    if args.provision or args.node_id:
        run_remote(node_id=args.node_id, provider=args.provider)
    else:
        # Run locally
        from rollouts._logging import setup_logging

        setup_logging(level="INFO", use_color=True)

        config = TestConfig(model_name=args.model)
        results = run_tests(config, require_gpu=not args.cpu)

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for r in results:
            status = "✓" if r.get("success") else "✗"
            print(f"  {status} {r['test']}")
        passed = sum(1 for r in results if r.get("success"))
        print(f"\n{passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
