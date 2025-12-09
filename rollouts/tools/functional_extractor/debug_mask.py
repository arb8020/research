#!/usr/bin/env python3
"""Debug attention mask differences between HF and functional implementation."""

from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def debug_on_gpu():
    """Compare attention mask handling."""
    import torch
    from transformers import AutoModelForCausalLM

    from qwen_functional import (
        qwen_forward,
        create_causal_mask,
    )

    print("=" * 60)
    print("Attention Mask Debug")
    print("=" * 60)

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    weights = {k: v for k, v in model.state_dict().items()}

    # Test case: left padding
    print("\n### Left Padding Test ###")
    input_ids = torch.randint(1, 1000, (2, 8), device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, :3] = 0  # First 3 tokens are padding for batch item 0
    attention_mask[1, :1] = 0  # First 1 token is padding for batch item 1

    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask:\n{attention_mask}")

    # Compute position IDs like HF does
    positions = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    print(f"Computed positions:\n{positions}")

    # Check what HF model actually uses for position_ids
    with torch.no_grad():
        # Run HF model with hooks to capture position_ids
        hf_out = model(input_ids, attention_mask=attention_mask, output_attentions=True)
        print(f"\nHF output shape: {hf_out.logits.shape}")

        # Check my implementation
        my_out = qwen_forward(input_ids, weights, attention_mask=attention_mask)
        print(f"My output shape: {my_out.shape}")

        # Compare logits at non-padded positions
        diff = (hf_out.logits - my_out).abs()
        print(f"\nMax diff overall: {diff.max().item():.2e}")

        # Look at diff per position
        for b in range(2):
            for s in range(8):
                pos_diff = diff[b, s].max().item()
                is_pad = attention_mask[b, s] == 0
                marker = "(PAD)" if is_pad else ""
                print(f"  batch={b}, pos={s}: max_diff={pos_diff:.2e} {marker}")

    # Let's check how HF constructs its 4D attention mask
    # Check what attention implementation HF is using
    print("\n### HF Attention Implementation ###")
    print(f"Model attn_implementation: {model.config._attn_implementation}")
    print(f"Layer 0 self_attn type: {type(model.model.layers[0].self_attn)}")

    # Run HF without output_attentions to use SDPA
    print("\n### Test: HF with SDPA (no output_attentions) ###")
    with torch.no_grad():
        hf_out_sdpa = model(input_ids, attention_mask=attention_mask, output_attentions=False)
        diff_sdpa = (hf_out_sdpa.logits - my_out).abs()
        print(f"Max diff (HF SDPA vs my impl): {diff_sdpa.max().item():.2e}")
        for b in range(2):
            for s in range(8):
                pos_diff = diff_sdpa[b, s].max().item()
                is_pad = attention_mask[b, s] == 0
                marker = "(PAD)" if is_pad else ""
                print(f"  batch={b}, pos={s}: max_diff={pos_diff:.2e} {marker}")

    print("\n### Mask Construction Debug ###")

    # Compare my mask vs HF's mask approach
    seq_len = 8
    device = input_ids.device
    dtype = torch.bfloat16

    # My mask construction
    my_mask = create_causal_mask(seq_len, device, dtype, attention_mask)
    print(f"\nMy mask shape: {my_mask.shape}")
    print(f"My mask [0, 0, :, :] (batch 0, head 0):")
    print(my_mask[0, 0])

    # What HF expects: let's look at what _prepare_4d_causal_attention_mask does
    # From looking at HF code, they use a different approach for SDPA:
    # - They return None when there's no padding (is_causal=True handles it)
    # - When there's padding, they create a min_dtype mask

    # Key insight: HF's SDPA path may actually NOT pass a mask at all
    # Let me check if the issue is that SDPA doesn't use my mask correctly
    # when is_causal=False

    # Test: what happens if we just use is_causal=True and ignore padding entirely?
    # Test with expanded mask (batch, num_heads, seq_len, seq_len)
    print("\n### Test: expanded mask to num_heads ###")
    from qwen_functional import NUM_HEADS, attention, rms_norm
    expanded_mask = my_mask.expand(-1, NUM_HEADS, -1, -1)
    print(f"Expanded mask shape: {expanded_mask.shape}")

    # Let's also look at the specific attention outputs
    print("\n### Layer 0 Attention Comparison ###")
    with torch.no_grad():
        # Get embedding
        hidden = torch.nn.functional.embedding(input_ids, weights["model.embed_tokens.weight"])

        # Compute positions
        if attention_mask is not None:
            positions = (attention_mask.cumsum(-1) - 1).clamp(min=0).long()
        else:
            positions = torch.arange(8, device=device).unsqueeze(0).expand(2, -1)

        from qwen_functional import compute_rope_embeddings
        cos, sin = compute_rope_embeddings(positions, dtype=hidden.dtype)

        # HF layer 0 attention
        hf_norm = model.model.layers[0].input_layernorm(hidden)

        # Run HF attention
        hf_attn_out = model.model.layers[0].self_attn(
            hf_norm,
            attention_mask=attention_mask,  # HF 2D mask
            position_embeddings=(cos, sin),
        )[0]

        # Run my attention with expanded mask
        my_norm = rms_norm(hidden, weights["model.layers.0.input_layernorm.weight"])

        # Check norm matches
        norm_diff = (hf_norm - my_norm).abs().max().item()
        print(f"RMSNorm diff: {norm_diff:.2e}")

        # My attention with expanded mask
        my_attn_out = attention(
            my_norm,
            q_weight=weights["model.layers.0.self_attn.q_proj.weight"],
            q_bias=weights["model.layers.0.self_attn.q_proj.bias"],
            k_weight=weights["model.layers.0.self_attn.k_proj.weight"],
            k_bias=weights["model.layers.0.self_attn.k_proj.bias"],
            v_weight=weights["model.layers.0.self_attn.v_proj.weight"],
            v_bias=weights["model.layers.0.self_attn.v_proj.bias"],
            o_weight=weights["model.layers.0.self_attn.o_proj.weight"],
            cos=cos,
            sin=sin,
            attention_mask=expanded_mask,
        )

        attn_diff = (hf_attn_out - my_attn_out).abs()
        print(f"Attention output max diff: {attn_diff.max().item():.2e}")
        for b in range(2):
            for s in range(8):
                pos_diff = attn_diff[b, s].max().item()
                is_pad = attention_mask[b, s] == 0
                marker = "(PAD)" if is_pad else ""
                print(f"  batch={b}, pos={s}: max_diff={pos_diff:.2e} {marker}")

    print("\n### Test: ignoring padding mask ###")
    with torch.no_grad():
        # My implementation without any mask (is_causal=True path)
        my_out_no_mask = qwen_forward(input_ids, weights, attention_mask=None)
        diff_no_mask = (hf_out.logits - my_out_no_mask).abs()
        print(f"Max diff with NO mask: {diff_no_mask.max().item():.2e}")

        for b in range(2):
            for s in range(8):
                pos_diff = diff_no_mask[b, s].max().item()
                is_pad = attention_mask[b, s] == 0
                marker = "(PAD)" if is_pad else ""
                print(f"  batch={b}, pos={s}: max_diff={pos_diff:.2e} {marker}")

    print("\nDone debugging!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--gpu-id", type=str)
    args = parser.parse_args()

    if args.remote or args.gpu_id:
        from tools.functional_extractor.verify import run_on_gpu
        run_on_gpu(__file__, gpu_id=args.gpu_id, keep_alive=True, vram_gb=16)
    else:
        import torch
        if torch.cuda.is_available():
            debug_on_gpu()
        else:
            print("No GPU available. Use --remote to run on remote GPU.")
