#!/usr/bin/env python3
"""Debug/verify sliding window attention implementation.

Tests that our FlexAttention sliding window mask produces correct results
by comparing against a naive reference implementation.

Usage:
    # Local (with GPU)
    uv run python -m rollouts.tools.functional_extractor.debug_swa --local

    # Remote GPU
    uv run python -m rollouts.tools.functional_extractor.debug_swa --gpu-id <id>
"""

from __future__ import annotations

import os


def debug_swa():
    """Test sliding window attention mask correctness."""
    import torch
    import torch.nn.functional as F

    print("=" * 60)
    print("Sliding Window Attention Debug")
    print("=" * 60)

    # Test parameters
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 32
    window_size = 4  # Small window for visualization

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Use float32 for numerical accuracy comparison

    print("\nTest config:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  seq_len: {seq_len}")
    print(f"  head_dim: {head_dim}")
    print(f"  window_size: {window_size}")
    print(f"  device: {device}")

    # Create random Q, K, V
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # =====================================================
    # Method 1: Reference implementation (explicit mask)
    # =====================================================
    print("\n--- Reference (explicit mask) ---")

    # Build causal sliding window mask manually
    # True where attention is allowed
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    # Sliding window: only attend to positions within window_size
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    window_mask = (row_idx - col_idx) < window_size

    # Combined: causal AND in window
    combined_mask = causal_mask & window_mask

    print(f"Reference mask shape: {combined_mask.shape}")
    print("Reference mask (first 8x8):")
    print(combined_mask[:8, :8].int())

    # Convert to attention mask format (-inf for masked positions)
    attn_mask = torch.where(
        combined_mask,
        torch.zeros_like(combined_mask, dtype=dtype),
        torch.full_like(combined_mask, float("-inf"), dtype=dtype),
    )
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]

    # Compute attention with explicit mask
    with torch.no_grad():
        ref_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
    print(f"Reference output shape: {ref_output.shape}")

    # =====================================================
    # Method 2: FlexAttention with our mask
    # =====================================================
    print("\n--- FlexAttention (create_sliding_window_causal_mask) ---")

    from rollouts.inference.attention.mask import create_sliding_window_causal_mask

    block_mask = create_sliding_window_causal_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        window_size=window_size,
        block_size=8,  # Small block size for test
        device=device,
    )
    print(f"BlockMask created: {type(block_mask)}")

    from torch.nn.attention.flex_attention import flex_attention

    with torch.no_grad():
        flex_output = flex_attention(q, k, v, block_mask=block_mask)
    print(f"FlexAttention output shape: {flex_output.shape}")

    # =====================================================
    # Compare results
    # =====================================================
    print("\n--- Comparison ---")

    diff = (ref_output - flex_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max absolute diff: {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    # Check if they match
    matches = torch.allclose(ref_output, flex_output, rtol=1e-4, atol=1e-4)
    print(f"Outputs match (rtol=1e-4, atol=1e-4): {matches}")

    if not matches:
        print("\nDifference by position (batch=0, head=0):")
        pos_diff = diff[0, 0].max(dim=-1).values
        for i, d in enumerate(pos_diff):
            if d > 1e-4:
                print(f"  pos {i}: {d:.2e}")

    # =====================================================
    # Test with larger window (full causal = no sliding)
    # =====================================================
    print("\n--- Full causal (window_size >= seq_len) ---")

    block_mask_full = create_sliding_window_causal_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        window_size=seq_len + 1,  # Larger than sequence = full causal
        block_size=8,
        device=device,
    )

    with torch.no_grad():
        flex_output_full = flex_attention(q, k, v, block_mask=block_mask_full)
        sdpa_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    full_diff = (flex_output_full - sdpa_causal).abs().max().item()
    print(f"Max diff (full SWA vs is_causal=True): {full_diff:.2e}")
    print(f"Matches: {full_diff < 1e-4}")

    # =====================================================
    # Test factory function
    # =====================================================
    print("\n--- Factory function (create_attention_mask) ---")

    from rollouts.inference.attention.mask import create_attention_mask

    # With sliding window
    mask_swa = create_attention_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        sliding_window=window_size,
        device=device,
    )
    print(f"With sliding_window={window_size}: {type(mask_swa)}")

    # Without sliding window (returns None, use is_causal=True)
    mask_causal = create_attention_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        sliding_window=None,
        device=device,
    )
    print(f"With sliding_window=None: {mask_causal}")

    # =====================================================
    # Summary
    # =====================================================
    print("\n" + "=" * 60)
    if matches and full_diff < 1e-4:
        print("SUCCESS: Sliding window attention implementation is correct!")
    else:
        print("FAILURE: Results don't match reference")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str, help="Reuse existing GPU instance")
    parser.add_argument("--local", action="store_true", help="Run locally (requires GPU)")
    args = parser.parse_args()

    # Check if we can run locally
    run_local = args.local
    if not run_local:
        try:
            import torch

            run_local = torch.cuda.is_available()
        except ImportError:
            run_local = False

    if run_local:
        debug_swa()
    else:
        print("No local GPU. Running on remote GPU...")

        # Inline GPU provisioning to avoid import issues
        from dotenv import load_dotenv

        load_dotenv()

        runpod_key = os.environ.get("RUNPOD_API_KEY")
        assert runpod_key, "RUNPOD_API_KEY not set"


        # TODO: Migrate to bifrost v2 API - kerbal has been deleted
        # See bifrost.ProcessSpec + client.submit() for the new pattern
        raise NotImplementedError("kerbal.python_env has been removed - migrate to bifrost v2 API")
