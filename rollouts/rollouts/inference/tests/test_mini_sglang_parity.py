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

import logging
import sys

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

    from minisgl.core import SamplingParams as MiniSGLSamplingParams
    from minisgl.llm import LLM as MiniSGLLLM

    from ..core import SamplingParams
    from ..engine_v2 import EngineConfig, InferenceEngineV2

    model_name = "HuggingFaceTB/SmolLM2-135M"
    prompts = [
        "The capital of France is",
        "Hello, my name is",
    ]
    max_tokens = 10

    # mini-sglang
    logger.info("Running mini-sglang...")
    try:
        mini_llm = MiniSGLLLM(model_name, dtype=torch.bfloat16)
        mini_params = MiniSGLSamplingParams(temperature=0.0, max_tokens=max_tokens)
        mini_results = mini_llm.generate(prompts, mini_params)
        mini_tokens = [r["token_ids"] for r in mini_results]
        logger.info(f"mini-sglang outputs: {mini_tokens}")
    except Exception as e:
        logger.warning(f"mini-sglang failed: {e}")
        return None

    # Our engine
    logger.info("Running our engine...")
    try:
        engine = InferenceEngineV2(
            EngineConfig(
                model_path=model_name,
                max_batch_size=8,
                max_tokens_per_batch=512,
                max_seq_len=512,
                attention_backend="reference",
            )
        )
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        our_results = engine.generate(prompts, params)

        # Extract generated tokens (not including prompt)
        our_tokens = []
        for req in our_results:
            # req.input_ids includes prompt, get only generated
            prompt_len = len(engine.tokenizer.encode(prompts[req.uid], add_special_tokens=True))
            generated = req.input_ids[prompt_len:].tolist()
            our_tokens.append(generated)

        logger.info(f"Our outputs: {our_tokens}")
        engine.shutdown()
    except Exception as e:
        logger.error(f"Our engine failed: {e}")
        import traceback

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
