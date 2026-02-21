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


def test_basic_generation(engine: Any, config: TestConfig) -> dict:
    """Test 1: Basic generation smoke test."""
    from rollouts.inference.core import SamplingParams

    emit_event("test_start", test="basic_generation")
    start = time.perf_counter()

    params = SamplingParams(max_tokens=config.num_tokens, temperature=0.7)
    engine.add_request("Hello, my name is", params)
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


def test_logprobs_greedy(engine: Any, config: TestConfig) -> dict:
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


def test_logprobs_sampling(engine: Any, config: TestConfig) -> dict:
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


def test_token_input(engine: Any, config: TestConfig) -> dict:
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


def test_multi_sample_per_prompt(engine: Any, config: TestConfig) -> dict:
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


def test_batched_generation(engine: Any, config: TestConfig) -> dict:
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


def test_flash_attention(engine: Any, _config: TestConfig) -> dict:
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


def test_cuda_graphs(engine: Any, _config: TestConfig) -> dict:
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


def test_logprob_alignment(engine: Any, config: TestConfig) -> dict:
    """Test 8: Verify logprobs align with tokens (critical for RL).

    For RL training, logprobs[i] must correspond to token[prompt_len + i].
    Off-by-one errors here break importance ratio computation.
    """
    from rollouts.inference.core import SamplingParams

    emit_event("test_start", test="logprob_alignment")
    start = time.perf_counter()

    # Use greedy decoding for determinism
    prompt = "The answer is"
    prompt_tokens = engine.tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_tokens)

    params = SamplingParams(max_tokens=10, temperature=0.0, return_logprobs=True)
    engine.add_request(prompt, params)
    finished = engine.run_to_completion()

    req = finished[0]
    all_tokens = req.input_ids.tolist()
    generated_tokens = all_tokens[prompt_len:]
    logprobs = req.logprobs.tolist() if req.logprobs is not None else []

    # Critical check: num_logprobs == num_generated_tokens
    alignment_correct = len(logprobs) == len(generated_tokens)

    # Additional check: logprobs are reasonable (not all zeros, not NaN)
    logprobs_valid = (
        len(logprobs) > 0
        and all(lp < 0 for lp in logprobs)  # All negative
        and all(lp > -100 for lp in logprobs)  # Not extreme
    )

    duration_ms = (time.perf_counter() - start) * 1000

    result = {
        "test": "logprob_alignment",
        "success": alignment_correct and logprobs_valid,
        "duration_ms": duration_ms,
        "prompt_len": prompt_len,
        "num_generated": len(generated_tokens),
        "num_logprobs": len(logprobs),
        "alignment_correct": alignment_correct,
        "logprobs_valid": logprobs_valid,
        "sample_logprobs": logprobs[:5] if logprobs else [],
    }
    emit_event("test_done", **result)
    return result


def test_determinism(engine: Any, config: TestConfig) -> dict:
    """Test 9: Verify greedy decoding is deterministic.

    Same prompt + temperature=0 should always produce same output.
    We call shutdown() between runs to clear radix cache and ensure
    each run starts from identical state.
    """
    from rollouts.inference.core import SamplingParams

    emit_event("test_start", test="determinism")
    start = time.perf_counter()

    prompt = "Once upon a time in a land far away"
    params = SamplingParams(max_tokens=15, temperature=0.0, return_logprobs=True)

    # Generate 3 times, clearing cache between runs
    outputs = []
    logprobs_list = []
    for _ in range(3):
        engine.shutdown()  # Clear radix cache for determinism
        engine.add_request(prompt, params)
        finished = engine.run_to_completion()
        outputs.append(finished[0].input_ids.tolist())
        if finished[0].logprobs is not None:
            logprobs_list.append(finished[0].logprobs.tolist())

    # All outputs should be identical
    all_same = all(o == outputs[0] for o in outputs)

    # All logprobs should be identical (within tolerance)
    logprobs_same = True
    if len(logprobs_list) == 3:
        for i in range(len(logprobs_list[0])):
            vals = [lp[i] for lp in logprobs_list]
            if max(vals) - min(vals) > 1e-5:
                logprobs_same = False
                break

    duration_ms = (time.perf_counter() - start) * 1000

    result = {
        "test": "determinism",
        "success": all_same and logprobs_same,
        "duration_ms": duration_ms,
        "outputs_identical": all_same,
        "logprobs_identical": logprobs_same,
        "output_len": len(outputs[0]),
    }
    emit_event("test_done", **result)
    return result


def test_single_token_prompt(engine: Any, _config: TestConfig) -> dict:
    """Test 11: Edge case - single token prompt."""
    from rollouts.inference.core import SamplingParams

    emit_event("test_start", test="single_token_prompt")
    start = time.perf_counter()

    engine.shutdown()

    # Single token prompt (just the BOS or a single word)
    prompt_ids = [engine.tokenizer.bos_token_id or 1]
    params = SamplingParams(max_tokens=5, temperature=0.0)

    engine.add_request(prompt_ids, params)
    finished = engine.run_to_completion()

    success = len(finished) == 1 and len(finished[0].input_ids) > len(prompt_ids)

    duration_ms = (time.perf_counter() - start) * 1000

    result = {
        "test": "single_token_prompt",
        "success": success,
        "duration_ms": duration_ms,
        "prompt_len": len(prompt_ids),
        "output_len": len(finished[0].input_ids) if finished else 0,
    }
    emit_event("test_done", **result)
    return result


def test_max_length_generation(engine: Any, config: TestConfig) -> dict:
    """Test 12: Edge case - generate until max length."""
    from rollouts.inference.core import SamplingParams

    emit_event("test_start", test="max_length_generation")
    start = time.perf_counter()

    engine.shutdown()

    prompt = "Count: 1, 2, 3,"
    # Request more tokens than we'll actually generate (should hit max_tokens)
    max_tokens = 50
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)

    engine.add_request(prompt, params)
    finished = engine.run_to_completion()

    prompt_len = len(engine.tokenizer.encode(prompt))
    output_len = len(finished[0].input_ids) - prompt_len

    # Should generate exactly max_tokens (or close to it)
    success = output_len == max_tokens

    duration_ms = (time.perf_counter() - start) * 1000

    result = {
        "test": "max_length_generation",
        "success": success,
        "duration_ms": duration_ms,
        "expected_tokens": max_tokens,
        "actual_tokens": output_len,
    }
    emit_event("test_done", **result)
    return result


def test_hf_reference(engine: Any, config: TestConfig) -> dict:
    """Test 11: Compare outputs against HuggingFace reference.

    This is the ground truth correctness test. We generate with both
    engine_v2 and HuggingFace, and verify token-for-token match.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from rollouts.inference.core import SamplingParams

    emit_event("test_start", test="hf_reference")
    start = time.perf_counter()

    # Clear engine state for determinism
    engine.shutdown()

    prompt = "The quick brown fox"
    max_tokens = 10

    # 1. Generate with engine_v2
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine.add_request(prompt, params)
    finished = engine.run_to_completion()
    engine_tokens = finished[0].input_ids.tolist()
    engine_output = engine_tokens[len(engine.tokenizer.encode(prompt)) :]

    # 2. Generate with HuggingFace
    hf_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=engine.config.dtype,
        device_map=engine.device,
    )
    hf_model.eval()

    input_ids = engine.tokenizer.encode(prompt, return_tensors="pt").to(engine.device)
    with torch.no_grad():
        hf_output_ids = hf_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=engine.tokenizer.pad_token_id or engine.tokenizer.eos_token_id,
        )
    hf_output = hf_output_ids[0, input_ids.shape[1] :].tolist()

    # Free HF model memory
    del hf_model
    torch.cuda.empty_cache()

    # Compare
    tokens_match = engine_output == hf_output
    match_count = sum(1 for a, b in zip(engine_output, hf_output, strict=False) if a == b)

    duration_ms = (time.perf_counter() - start) * 1000

    result = {
        "test": "hf_reference",
        "success": tokens_match,
        "duration_ms": duration_ms,
        "tokens_match": tokens_match,
        "match_count": match_count,
        "total_tokens": len(engine_output),
        "engine_output": engine_output[:5],  # First 5 for debugging
        "hf_output": hf_output[:5],
    }
    emit_event("test_done", **result)
    return result


def test_weight_reload(engine: Any, config: TestConfig) -> dict:
    """Test 10: Weight hot-reload (RL use case)."""
    from rollouts.inference.core import SamplingParams
    from rollouts.inference.models.weight import load_weights

    emit_event("test_start", test="weight_reload")
    start = time.perf_counter()

    # 1. Generate with original weights (clear cache first for clean state)
    engine.shutdown()
    params = SamplingParams(max_tokens=5, temperature=0.0, return_logprobs=True)
    engine.add_request("The capital of France is", params)
    finished_before = engine.run_to_completion()
    tokens_before = finished_before[0].input_ids.tolist()

    # 2. Reload the same weights (tests the mechanism, not weight changes)
    # In practice, this would be new weights from training
    # Note: reload_weights() clears radix cache internally
    state_dict = load_weights(config.model_name, engine.device, engine.config.dtype)
    engine.reload_weights(state_dict)

    # 3. Generate again - should work after reload
    engine.add_request("The capital of France is", params)
    finished_after = engine.run_to_completion()
    tokens_after = finished_after[0].input_ids.tolist()

    duration_ms = (time.perf_counter() - start) * 1000

    # With same weights and temperature=0, outputs should be identical
    outputs_match = tokens_before == tokens_after

    result = {
        "test": "weight_reload",
        "success": outputs_match,
        "duration_ms": duration_ms,
        "outputs_match": outputs_match,
        "tokens_before": len(tokens_before),
        "tokens_after": len(tokens_after),
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
        test_logprob_alignment,
        test_determinism,
        test_single_token_prompt,  # Edge case: single token input
        test_max_length_generation,  # Edge case: hit max_tokens limit
        test_hf_reference,  # Compare to HuggingFace ground truth
        test_weight_reload,
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


async def run_remote_async(node_id: str | None = None, keep_alive: bool = True) -> None:
    """Deploy and run on remote GPU via broker/bifrost."""
    import os
    import subprocess

    from dotenv import load_dotenv

    from bifrost.client import BifrostClient
    from broker.client import GPUClient

    # Load .env from git root (workspace root)
    script = Path(__file__).resolve()
    git_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
    load_dotenv(git_root / ".env")

    # Get script path relative to git root (already computed above)
    rel_path = script.relative_to(git_root)

    # Provision or reuse GPU
    runpod_key = os.getenv("RUNPOD_API_KEY")
    assert runpod_key, "RUNPOD_API_KEY not set"
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    client = GPUClient(credentials={"runpod": runpod_key}, ssh_key_path=ssh_key_path)
    gpu = None

    try:
        if node_id:
            print(f"Reusing instance: {node_id}")
            gpu = await client.get_instance(node_id, provider="runpod")
            if not gpu:
                print(f"GPU {node_id} not found (is it still running?)")
                return
            keep_alive = True
        else:
            print("Provisioning GPU...")
            gpu = await client.create(
                query=(client.vram_gb >= 24) & (client.price_per_hour <= 0.5),
                name=f"inference-{script.stem}",
            )
            if not gpu:
                print("Failed to provision GPU")
                return
            print(f"GPU ready: {gpu.id}")

            if not await gpu.wait_until_ssh_ready(timeout=300):
                print("SSH timeout")
                await client.terminate_instance(gpu.id, gpu.provider)
                return

        print(f"SSH: {gpu.ssh_connection_string()}")

        # Deploy (bifrost methods are sync, only broker is async)
        workspace = "~/.bifrost/workspaces/rollouts"
        bifrost = BifrostClient(gpu.ssh_connection_string(), ssh_key_path)
        bifrost.push(workspace_path=workspace, allow_dirty=True)
        print("Code synced")

        # Bootstrap steps
        bootstrap_steps = [
            ("Installing uv", "curl -LsSf https://astral.sh/uv/install.sh | sh"),
            (
                "Syncing deps",
                f"~/.local/bin/uv python install 3.12 && ~/.local/bin/uv sync --project {workspace}/rollouts --python 3.12",
            ),
            (
                "Installing torch",
                "~/.local/bin/uv pip install torch 'transformers<4.52' accelerate",
            ),
        ]
        for label, cmd in bootstrap_steps:
            print(f"  {label}...")
            result = bifrost.exec(cmd, working_dir=workspace)
            if result.exit_code != 0:
                raise RuntimeError(f"Bootstrap '{label}' failed: {result.stderr or result.stdout}")
        print("Bootstrap done")

        # Run with streaming output
        remote_script = f"{workspace}/{rel_path}"
        cmd = f"cd {workspace}/rollouts && ~/.local/bin/uv run python {remote_script}"
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
            print(f"Instance kept alive: {gpu.id}")
            print(f"SSH: {gpu.ssh_connection_string()}")
            print()
            print(f"Rerun with:   --node-id {gpu.id}")
            print(f"Terminate:    broker terminate {gpu.id}")
            print("=" * 50)
        else:
            print("Cleaning up...")
            await client.terminate_instance(gpu.id, gpu.provider)


def run_remote(node_id: str | None = None, keep_alive: bool = True) -> None:
    """Sync wrapper for run_remote_async."""
    import trio

    trio.run(run_remote_async, node_id, keep_alive)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test engine_v2 on GPU")
    parser.add_argument("--provision", action="store_true", help="Provision remote GPU")
    parser.add_argument("--provider", default="runpod", help="GPU provider (runpod, modal)")
    parser.add_argument("--node-id", help="Reuse existing instance")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Model to test")
    parser.add_argument(
        "--cpu", action="store_true", help="Allow running on CPU (for local testing)"
    )
    args = parser.parse_args()

    if args.provision or args.node_id:
        run_remote(node_id=args.node_id)
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
