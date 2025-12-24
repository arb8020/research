#!/usr/bin/env python3
"""Integration test for KV cache flow.

Tests the full pipeline:
1. PagedKVCache allocates blocks
2. Context builder creates InferenceContext
3. FlexAttentionBackend stores/retrieves K/V

Usage:
    python examples/inference/test_kv_cache.py           # Run locally (GPU)
    python examples/inference/test_kv_cache.py --remote  # Run on remote GPU
"""

import sys
from pathlib import Path

# Add parent to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_paged_kv_cache():
    """Test block allocation logic (CPU-only)."""
    from rollouts.inference.cache import PagedKVCache

    print("Testing PagedKVCache...")

    # Test allocate
    cache = PagedKVCache(num_blocks=16, block_size=4)
    token_ids = [1, 2, 3, 4, 5, 6, 7]  # 7 tokens = 2 blocks
    block_ids = cache.allocate(seq_id=0, token_ids=token_ids)
    assert len(block_ids) == 2, f"Expected 2 blocks, got {len(block_ids)}"
    assert cache.num_free_blocks() == 14
    print("  allocate: OK")

    # Test deallocate
    cache.deallocate(seq_id=0)
    assert cache.num_free_blocks() == 16
    print("  deallocate: OK")

    # Test prefix caching
    cache = PagedKVCache(num_blocks=16, block_size=4)
    token_ids = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 complete blocks
    cache.allocate(seq_id=0, token_ids=token_ids)
    cache.allocate(seq_id=1, token_ids=token_ids + [9, 10])
    cached = cache.get_cached_tokens(seq_id=1)
    assert cached == 8, f"Expected 8 cached tokens, got {cached}"
    print("  prefix_caching: OK")

    print("PagedKVCache: PASSED")


def test_context_builder():
    """Test context building (CPU-only)."""
    from rollouts.inference.cache import PagedKVCache
    from rollouts.inference.context import (
        allocate_and_build_context,
        extend_and_build_context,
    )

    print("Testing ContextBuilder...")

    # Test prefill context
    cache = PagedKVCache(num_blocks=16, block_size=4)
    token_ids = [1, 2, 3, 4, 5]  # 5 tokens
    ctx, cached = allocate_and_build_context(cache, seq_id=0, token_ids=token_ids)
    assert ctx.is_prefill
    assert len(ctx.slot_mapping) == 5
    print("  prefill_context: OK")

    # Test decode context
    cache = PagedKVCache(num_blocks=16, block_size=4)
    cache.allocate(seq_id=0, token_ids=[1, 2, 3])
    ctx = extend_and_build_context(cache, seq_id=0, new_token=4, current_len=3)
    assert not ctx.is_prefill
    assert len(ctx.slot_mapping) == 1
    print("  decode_context: OK")

    print("ContextBuilder: PASSED")


def test_flex_attention_backend():
    """Test attention backend with GPU."""
    import torch
    from rollouts.inference.attention import CacheConfig, FlexAttentionBackend

    print("Testing FlexAttentionBackend...")

    config = CacheConfig(
        num_layers=2,
        num_kv_heads=4,
        head_dim=32,
        num_blocks=16,
        block_size=4,
        device="cuda",
    )
    backend = FlexAttentionBackend(config)
    backend.set_num_heads(8)

    # Create K/V tensors
    batch, seq_len = 1, 4
    k = torch.randn(batch, seq_len, 4, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seq_len, 4, 32, device="cuda", dtype=torch.bfloat16)

    # Store into layer 0, slots 0-3
    slot_mapping = tuple(range(4))
    backend._store_kv(k, v, layer_idx=0, slot_mapping=slot_mapping)

    # Verify stored
    assert torch.allclose(backend.k_cache[0, 0], k[0, 0])
    assert torch.allclose(backend.v_cache[0, 3], v[0, 3])
    print("  store_kv: OK")

    print("FlexAttentionBackend: PASSED")


def test_end_to_end():
    """End-to-end integration test."""
    import torch
    from rollouts.inference.attention import CacheConfig, FlexAttentionBackend
    from rollouts.inference.cache import PagedKVCache
    from rollouts.inference.context import (
        allocate_and_build_context,
        extend_and_build_context,
    )

    print("Testing End-to-End...")

    # Setup
    num_layers = 2
    num_kv_heads = 4
    num_heads = 8
    head_dim = 32
    block_size = 4

    cache = PagedKVCache(num_blocks=16, block_size=block_size)
    config = CacheConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=16,
        block_size=block_size,
        device="cuda",
    )
    backend = FlexAttentionBackend(config)
    backend.set_num_heads(num_heads)

    # === PREFILL ===
    prompt_tokens = [1, 2, 3, 4, 5]
    ctx, cached = allocate_and_build_context(cache, seq_id=0, token_ids=prompt_tokens)

    batch, seq_len = 1, len(prompt_tokens)
    q = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)

    out = backend.forward(q, k, v, layer_idx=0, ctx=ctx)
    assert out.shape == (batch, seq_len, num_heads, head_dim)
    print("  prefill: OK")

    # === DECODE ===
    ctx = extend_and_build_context(cache, seq_id=0, new_token=6, current_len=5)

    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 1, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, 1, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)

    out = backend.forward(q, k, v, layer_idx=0, ctx=ctx)
    assert out.shape == (batch, 1, num_heads, head_dim)
    print("  decode: OK")

    # Cleanup
    cache.deallocate(seq_id=0)
    assert cache.num_free_blocks() == 16
    print("  cleanup: OK")

    print("End-to-End: PASSED")


def run_tests():
    """Run all tests."""
    import torch

    print("=" * 50)
    print("KV Cache Integration Tests")
    print("=" * 50)

    # CPU tests (always run)
    test_paged_kv_cache()
    test_context_builder()

    # GPU tests
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name()}")
        test_flex_attention_backend()
        test_attention_correctness()
        test_end_to_end()
    else:
        print("\nSkipping GPU tests (CUDA not available)")

    print()
    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


def run_remote():
    """Run tests on remote GPU."""
    import os
    from pathlib import Path

    from bifrost.client import BifrostClient
    from broker.client import GPUClient
    from dotenv import load_dotenv

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
        print("Provisioning GPU...")
        gpu = client.create(
            query=(client.vram_gb >= 24) & (client.price_per_hour <= 0.5),
            name="test-kv-cache",
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

        workspace = "~/.bifrost/workspaces/rollouts"
        bifrost = BifrostClient(gpu.ssh_connection_string(), ssh_key_path)
        bootstrap = [
            "cd rollouts && uv python install 3.12 && uv sync --python 3.12",
            "uv pip install torch 'transformers<4.52' datasets accelerate",
        ]
        bifrost.push(workspace_path=workspace, bootstrap_cmd=bootstrap)
        print("Code deployed")

        remote_script = f"{workspace}/{rel_path}"
        cmd = f"cd {workspace}/rollouts && uv run python {remote_script}"
        print(f"Running: {cmd}")
        print("-" * 50)
        for line in bifrost.exec_stream(cmd):
            print(line, end="")
        print("-" * 50)

    finally:
        if gpu:
            print("Cleaning up...")
            client.terminate_instance(gpu.id, gpu.provider)


def test_attention_correctness():
    """Verify our attention matches PyTorch reference implementation."""
    import torch
    import torch.nn.functional as F
    from rollouts.inference.attention import CacheConfig, FlexAttentionBackend
    from rollouts.inference.types import InferenceContext

    print("Testing Attention Correctness...")

    # Setup
    batch = 2
    seq_len = 8
    num_heads = 4
    num_kv_heads = 2  # GQA: 4 query heads, 2 kv heads
    head_dim = 32

    config = CacheConfig(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=16,
        block_size=16,
        device="cuda",
    )
    backend = FlexAttentionBackend(config)
    backend.set_num_heads(num_heads)

    # Random Q/K/V
    torch.manual_seed(42)
    q = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)

    # Create context for prefill (no caching, just compute attention)
    ctx = InferenceContext(
        is_prefill=True,
        slot_mapping=None,  # Skip cache storage for this test
        seq_lens=tuple([seq_len] * batch),
        max_seq_len=seq_len,
    )

    # Our implementation
    our_out = backend.forward(q, k, v, layer_idx=0, ctx=ctx)

    # Reference: PyTorch SDPA with GQA expansion
    q_ref = q.transpose(1, 2)  # [batch, heads, seq, dim]
    k_ref = k.transpose(1, 2)
    v_ref = v.transpose(1, 2)

    # Expand KV heads for GQA
    k_ref = k_ref.repeat_interleave(num_heads // num_kv_heads, dim=1)
    v_ref = v_ref.repeat_interleave(num_heads // num_kv_heads, dim=1)

    ref_out = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    ref_out = ref_out.transpose(1, 2)  # [batch, seq, heads, dim]

    # Compare
    max_diff = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()

    print(f"  max_diff: {max_diff:.6f}")
    print(f"  mean_diff: {mean_diff:.6f}")

    # bfloat16 has ~3 decimal digits of precision
    assert max_diff < 0.01, f"Max diff {max_diff} too large"
    assert mean_diff < 0.001, f"Mean diff {mean_diff} too large"

    print("  correctness: OK")
    print("Attention Correctness: PASSED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", help="Run on remote GPU")
    args = parser.parse_args()

    if args.remote:
        run_remote()
    else:
        run_tests()
