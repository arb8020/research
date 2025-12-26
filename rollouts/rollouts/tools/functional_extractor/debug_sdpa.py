#!/usr/bin/env python3
"""Debug SDPA calls directly - are HF and my code calling SDPA identically?"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def debug_sdpa() -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    print("=" * 60)
    print("SDPA Direct Comparison")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    weights = dict(model.state_dict())

    # Simple test: same Q, K, V, same mask -> same output?
    batch, heads, seq, head_dim = 1, 14, 8, 64

    q = torch.randn(batch, heads, seq, head_dim, device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn(batch, heads, seq, head_dim, device="cuda:0", dtype=torch.bfloat16)
    v = torch.randn(batch, heads, seq, head_dim, device="cuda:0", dtype=torch.bfloat16)

    # Create mask with padding
    attention_mask_2d = torch.ones(batch, seq, device="cuda:0", dtype=torch.long)
    attention_mask_2d[0, :3] = 0  # First 3 are padding

    # Build 4D mask (same way both HF and I do it)
    from qwen_functional import create_causal_mask

    mask_4d = create_causal_mask(seq, q.device, q.dtype, attention_mask_2d)

    print(f"\nMask shape: {mask_4d.shape}")
    print(f"Mask at pos 3 (first real): {mask_4d[0, 0, 3, :].tolist()}")

    # Call SDPA twice with identical inputs
    with torch.no_grad():
        out1 = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_4d, is_causal=False)
        out2 = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_4d, is_causal=False)

    diff_same_call = (out1 - out2).abs().max().item()
    print(f"\nSame SDPA call twice: max_diff = {diff_same_call:.2e}")

    # Now compare is_causal=True (no mask) vs explicit mask (for non-padded case)
    # This shows the kernel difference
    with torch.no_grad():
        out_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Build full causal mask (no padding)
        full_mask = torch.tril(torch.ones(seq, seq, device="cuda:0", dtype=torch.bool))
        full_mask = full_mask.unsqueeze(0).unsqueeze(0)
        min_dtype = torch.finfo(torch.bfloat16).min
        full_mask_float = torch.where(full_mask, 0.0, min_dtype).to(torch.bfloat16)

        out_explicit = F.scaled_dot_product_attention(
            q, k, v, attn_mask=full_mask_float, is_causal=False
        )

    diff_kernel = (out_causal - out_explicit).abs().max().item()
    print(f"is_causal=True vs explicit causal mask: max_diff = {diff_kernel:.2e}")
    print("  ^ This shows kernel difference for mathematically identical ops")

    # Now the real test: Does HF's forward match my forward with same inputs?
    print("\n" + "=" * 60)
    print("HF vs Functional with attention mask")
    print("=" * 60)

    input_ids = torch.randint(1, 1000, (2, 16), device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, :4] = 0
    attention_mask[1, :2] = 0

    print(f"attention_mask[0]: {attention_mask[0].tolist()}")
    print(f"attention_mask[1]: {attention_mask[1].tolist()}")

    with torch.no_grad():
        hf_out = model(input_ids, attention_mask=attention_mask).logits

        from qwen_functional import qwen_forward

        my_out = qwen_forward(input_ids, weights, attention_mask=attention_mask)

    diff = (hf_out - my_out).abs()

    # Check diff at each position
    print("\nPer-position max diff:")
    for b in range(2):
        for pos in range(16):
            if attention_mask[b, pos] == 1:  # Only real positions
                pos_diff = diff[b, pos].max().item()
                marker = "***" if pos_diff > 0.1 else ""
                print(f"  batch {b}, pos {pos}: {pos_diff:.4f} {marker}")


if __name__ == "__main__":
    import torch

    if torch.cuda.is_available():
        debug_sdpa()
    else:
        print("No GPU available. Run on remote GPU.")
        import argparse

        from verify import run_on_gpu
        from config import DeploymentConfig

        parser = argparse.ArgumentParser()
        parser.add_argument("--gpu-id", type=str)
        args = parser.parse_args()
        run_on_gpu(__file__, deployment=DeploymentConfig(vram_gb=16), gpu_id=args.gpu_id, keep_alive=True)
