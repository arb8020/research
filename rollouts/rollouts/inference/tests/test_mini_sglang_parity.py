"""Numerical equivalence tests against mini-sglang.

These tests verify that our inference engine produces identical outputs
to mini-sglang for the same inputs.

Test categories:
1. Logits comparison - exact logit values for same input
2. Generation comparison - same output tokens for greedy decoding
3. Sampling comparison - same distribution statistics

Requirements:
- mini-sglang installed: pip install minisgl
- CUDA GPU available
- Test model downloaded (SmolLM2-135M or Qwen3-0.6B)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import traceback

import torch

from rollouts._logging import setup_logging

# Setup logging - use color for console, respects LOG_LEVEL env var
# Disable queue_handler for Python 3.11 compatibility (Modal sandbox)
setup_logging(
    use_color=True,
    logger_levels={"httpx": "WARNING"},
    use_queue_handler=(sys.version_info >= (3, 12)),
)
logger = logging.getLogger(__name__)


def _run_mini_sglang_in_subprocess(
    model_name: str,
    prompts: list[str],
    max_tokens: int,
    ignore_eos: bool = False,
) -> list[list[int]]:
    script = """
import json
import os
import traceback
import torch
from minisgl.core import SamplingParams as MiniSGLSamplingParams
from minisgl.llm import LLM as MiniSGLLLM

model_name = os.environ["MINISGL_MODEL_NAME"]
prompts = json.loads(os.environ["MINISGL_PROMPTS"])
max_tokens = int(os.environ["MINISGL_MAX_TOKENS"])
ignore_eos = os.environ.get("MINISGL_IGNORE_EOS", "0") == "1"

try:
    llm = MiniSGLLLM(model_name, dtype=torch.bfloat16)
    params = MiniSGLSamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )
    results = llm.generate(prompts, params)
    tokens = [r["token_ids"] for r in results]
    print(json.dumps({"ok": True, "tokens": tokens}))
except Exception as e:
    traceback.print_exc()
    print(json.dumps({"ok": False, "type": type(e).__name__, "repr": repr(e)}))
"""

    env = {
        **os.environ,
        "MINISGL_MODEL_NAME": model_name,
        "MINISGL_PROMPTS": json.dumps(prompts),
        "MINISGL_MAX_TOKENS": str(max_tokens),
        "MINISGL_IGNORE_EOS": "1" if ignore_eos else "0",
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            timeout=180,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"mini-sglang subprocess timed out after 180s stdout={e.stdout!r} stderr={e.stderr!r}"
        ) from e

    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    payload = None
    if stdout_lines:
        try:
            payload = json.loads(stdout_lines[-1])
        except json.JSONDecodeError:
            payload = None

    if proc.returncode != 0 or not payload or not payload.get("ok", False):
        details = payload or {"type": "UnknownError", "repr": "No JSON payload from subprocess"}
        raise RuntimeError(
            f"mini-sglang subprocess failed: type={details.get('type')} "
            f"repr={details.get('repr')} stderr={proc.stderr!r}"
        )

    return payload["tokens"]


def check_mini_sglang_available() -> bool:
    """Check if mini-sglang is installed."""
    try:
        import minisgl

        return True
    except ImportError:
        return False


def test_logits_vs_mini_sglang():
    """Compare logits between our engine and mini-sglang.

    This is the most precise test - compares raw logits before sampling.
    """
    if not torch.cuda.is_available():
        logger.info("Skipping (no CUDA)")
        return None

    if not check_mini_sglang_available():
        logger.info("Skipping (mini-sglang not installed)")
        return None

    logger.info("Testing logits equivalence...")

    # TODO: Implement when we have a way to extract logits from mini-sglang
    # mini-sglang's LLM.generate() doesn't expose logits directly
    logger.info("Skipping logits test (mini-sglang doesn't expose logits)")
    return None


def test_greedy_generation_vs_mini_sglang():
    """Compare greedy generation outputs.

    For greedy decoding (temperature=0), both engines should produce
    identical token sequences.
    """
    if not torch.cuda.is_available():
        logger.info("Skipping (no CUDA)")
        return None

    if not check_mini_sglang_available():
        logger.info("Skipping (mini-sglang not installed)")
        return None

    logger.info("Testing greedy generation vs mini-sglang...")

    from ..core import SamplingParams
    from ..engine_v2 import EngineConfig, InferenceEngineV2

    model_name = "HuggingFaceTB/SmolLM2-135M"
    prompts = [
        "Write me a very long fantasy story about a dragon and a lighthouse.",
        "Tell a long science fiction story about a spaceship crew lost in time.",
    ]
    max_tokens = 16

    # mini-sglang
    logger.info("Running mini-sglang...")
    try:
        mini_tokens = _run_mini_sglang_in_subprocess(
            model_name=model_name,
            prompts=prompts,
            max_tokens=max_tokens,
            ignore_eos=False,
        )
        logger.info(f"mini-sglang outputs: {mini_tokens}")
    except Exception as e:
        logger.error("mini-sglang runtime failed")
        logger.error(f"mini-sglang exception type: {type(e).__name__}")
        logger.error(f"mini-sglang exception repr: {e!r}")
        traceback.print_exc()
        return False

    # Our engine
    logger.info("Running our engine...")
    try:
        # mini-sglang's scheduler advances Req.device_len during prefill before
        # host-side token append, so effective returned completion length is
        # one token shorter than SamplingParams.max_tokens.
        # Match mini-sglang's observed API behavior for parity comparison.
        our_max_tokens = max(1, max_tokens - 1)
        engine = InferenceEngineV2(
            EngineConfig(
                model_path=model_name,
                max_batch_size=8,
                max_tokens_per_batch=512,
                max_seq_len=512,
                model_impl="functional",
            )
        )
        params = SamplingParams(temperature=0.0, max_tokens=our_max_tokens, ignore_eos=False)
        our_results = engine.generate(prompts, params)

        # Extract generated tokens (not including prompt)
        our_tokens = []
        by_uid = {req.uid: req for req in our_results}
        for uid, prompt in enumerate(prompts):
            req = by_uid[uid]
            # req.input_ids includes prompt, get only generated
            prompt_len = len(engine.tokenizer.encode(prompt, add_special_tokens=True))
            generated = req.input_ids[prompt_len:].tolist()
            # mini-sglang LLM API excludes terminal EOS token from returned token_ids.
            if generated and generated[-1] == engine.eos_token_id:
                generated = generated[:-1]
            our_tokens.append(generated)

        logger.info(f"Our outputs: {our_tokens}")
        engine.shutdown()
    except Exception as e:
        logger.error(f"Our engine failed: {e}")
        traceback.print_exc()
        return False

    # Compare
    match = True
    for i, (mini, ours) in enumerate(zip(mini_tokens, our_tokens, strict=False)):
        if mini != ours:
            logger.error(f"Mismatch for prompt {i}:")
            logger.error(f"  mini-sglang: {mini}")
            logger.error(f"  ours: {ours}")
            match = False

    if match:
        logger.info("PASS: Greedy generation matches mini-sglang")
    else:
        logger.error("FAIL: Generation differs from mini-sglang")

    return match


def test_kv_cache_correctness():
    """Test that KV cache produces correct results.

    Verifies that using cached K,V gives same results as recomputing.
    """
    if not torch.cuda.is_available():
        logger.info("Skipping (no CUDA)")
        return None

    logger.info("Testing KV cache correctness...")

    from ..attention.backend import build_attention_metadata
    from ..attention.reference import ReferenceAttentionBackend
    from ..kv_cache import CacheConfig, KVCachePool

    device = torch.device("cuda")
    dtype = torch.float32  # Higher precision for comparison

    # Setup
    num_layers = 2
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 64
    num_slots = 200

    # Create cache
    cache_config = CacheConfig(
        num_layers=num_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        num_slots=num_slots,
        dtype=dtype,
    )
    kv_pool = KVCachePool(cache_config, device)

    backend = ReferenceAttentionBackend(
        k_cache=kv_pool.k_cache,
        v_cache=kv_pool.v_cache,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    # Test: Process sequence in two steps, compare to single step
    seq_len = 10
    torch.manual_seed(42)
    full_q = torch.randn(seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    full_k = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    full_v = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Single step: process all at once (baseline)
    page_table = torch.arange(num_slots, dtype=torch.int32, device=device).view(1, -1)
    out_loc_full = torch.arange(seq_len, dtype=torch.int32, device=device)

    metadata_full = build_attention_metadata(
        cached_lens=[0],
        extend_lens=[seq_len],
        page_table=page_table,
        device=device,
    )

    out_single = backend.forward(full_q, full_k, full_v, 0, metadata_full, out_loc_full)

    # Reset cache
    kv_pool.reset()

    # Two steps: process first 5, then last 5
    split = 5

    # Step 1: process first 5 tokens
    out_loc_1 = torch.arange(split, dtype=torch.int32, device=device)
    metadata_1 = build_attention_metadata(
        cached_lens=[0],
        extend_lens=[split],
        page_table=page_table,
        device=device,
    )
    out_1 = backend.forward(
        full_q[:split], full_k[:split], full_v[:split], 0, metadata_1, out_loc_1
    )

    # Step 2: process last 5 tokens (using cached K,V from step 1)
    out_loc_2 = torch.arange(split, seq_len, dtype=torch.int32, device=device)
    metadata_2 = build_attention_metadata(
        cached_lens=[split],
        extend_lens=[seq_len - split],
        page_table=page_table,
        device=device,
    )
    out_2 = backend.forward(
        full_q[split:], full_k[split:], full_v[split:], 0, metadata_2, out_loc_2
    )

    # Combine outputs from two steps
    out_two_step = torch.cat([out_1, out_2], dim=0)

    # Compare
    max_diff = (out_single - out_two_step).abs().max().item()
    logger.info(f"Max diff between single-step and two-step: {max_diff:.2e}")

    if max_diff < 1e-5:
        logger.info("PASS: KV cache produces correct results")
        return True
    else:
        logger.error(f"FAIL: Max diff {max_diff} exceeds threshold")
        return False


def test_radix_cache_operations():
    """Test radix cache insert/match/evict operations."""
    logger.info("Testing radix cache operations...")

    from ..radix import init_radix_state, insert_prefix, lock, match_prefix, unlock

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # State dict pattern (nmoe style)
    state: dict = {}
    init_radix_state(state, device)

    # Insert some sequences
    seq1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=device)
    slots1 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)

    seq2 = torch.tensor([1, 2, 3, 6, 7], dtype=torch.int32, device=device)  # Shares prefix
    slots2 = torch.tensor([0, 1, 2, 5, 6], dtype=torch.int32, device=device)

    # Insert first sequence
    inserted1 = insert_prefix(state, seq1, slots1)
    assert inserted1 == 5, f"Expected 5 inserted, got {inserted1}"

    # Insert second sequence (should reuse prefix)
    inserted2 = insert_prefix(state, seq2, slots2)
    assert inserted2 == 2, f"Expected 2 inserted (shared prefix), got {inserted2}"

    # Match prefix
    query = torch.tensor([1, 2, 3, 4, 5, 8, 9], dtype=torch.int32, device=device)
    handle, matched_slots = match_prefix(state, query)

    assert handle.cached_len == 5, f"Expected 5 matched, got {handle.cached_len}"
    assert len(matched_slots) == 5

    # Lock/unlock
    lock(state, handle)
    assert state["protected_tokens"] > 0

    unlock(state, handle)
    assert state["protected_tokens"] == 0

    logger.info("PASS: Radix cache operations work correctly")
    return True


def run_all_tests():
    """Run all parity tests."""
    results = []

    # CPU-only tests
    results.append(("radix_cache", test_radix_cache_operations()))

    # GPU tests
    if torch.cuda.is_available():
        results.append(("kv_cache_correctness", test_kv_cache_correctness()))
        results.append(("greedy_vs_minisglang", test_greedy_generation_vs_mini_sglang()))
    else:
        logger.info("Skipping GPU tests (no CUDA)")

    # Print summary
    print("\n" + "=" * 60)
    print("mini-sglang Parity Test Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results:
        if result is None:
            status = "SKIP"
            skipped += 1
        elif result:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
