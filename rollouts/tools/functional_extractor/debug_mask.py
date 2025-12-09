#!/usr/bin/env python3
"""Debug attention mask construction - compare HF vs functional."""

from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def debug_mask():
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    from qwen_functional import create_causal_mask

    print("=" * 60)
    print("Attention Mask Debug")
    print("=" * 60)

    # Load model to get config
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    # Test case: left padding
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(1, 1000, (batch_size, seq_len), device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, :3] = 0  # First 3 tokens are padding for batch item 0
    attention_mask[1, :1] = 0  # First 1 token is padding for batch item 1

    print(f"\nTest case: batch={batch_size}, seq={seq_len}")
    print(f"attention_mask[0] = {attention_mask[0].tolist()}")
    print(f"attention_mask[1] = {attention_mask[1].tolist()}")

    # Get HF's mask construction
    print("\n### HuggingFace Mask ###")

    # Get the actual mask from HF model forward
    # We need to trace through how HF constructs the mask
    dtype = torch.bfloat16

    # Check what HF model does with attention_mask
    # Look at Qwen2Model._update_causal_mask

    # Direct HF mask construction
    converter = AttentionMaskConverter(is_causal=True, sliding_window=None)

    # Use to_4d to convert 2D -> 4D
    # HF signature: to_4d(attention_mask_2d, query_length, key_value_length=None, dtype=None, ...)
    hf_mask_4d = converter.to_4d(
        attention_mask,  # 2D mask
        seq_len,  # query_length
        key_value_length=seq_len,  # key_value_length (same for prefill)
        dtype=dtype,
    )

    print(f"HF mask shape: {hf_mask_4d.shape}")
    print(f"HF mask dtype: {hf_mask_4d.dtype}")

    # Now apply _unmask_unattended (HF does this in _prepare_4d_causal_attention_mask_for_sdpa)
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter as AMC
    hf_mask_4d_unmasked = AMC._unmask_unattended(hf_mask_4d, min_dtype=torch.finfo(dtype).min)

    print(f"\nHF mask (after _unmask_unattended) sample values:")
    print(f"  batch 0, pos 0 (padding): {hf_mask_4d_unmasked[0, 0, 0, :].tolist()}")
    print(f"  batch 0, pos 3 (first real): {hf_mask_4d_unmasked[0, 0, 3, :].tolist()}")
    print(f"  batch 0, pos 7 (last): {hf_mask_4d_unmasked[0, 0, 7, :].tolist()}")
    print(f"  batch 1, pos 0 (padding): {hf_mask_4d_unmasked[1, 0, 0, :].tolist()}")
    print(f"  batch 1, pos 1 (first real): {hf_mask_4d_unmasked[1, 0, 1, :].tolist()}")

    # Get my mask construction
    print("\n### My Mask ###")
    my_mask = create_causal_mask(seq_len, input_ids.device, dtype, attention_mask)

    if my_mask is None:
        print("My mask is None (using is_causal=True)")
    else:
        print(f"My mask shape: {my_mask.shape}")
        print(f"My mask dtype: {my_mask.dtype}")
        print(f"\nMy mask sample values:")
        print(f"  batch 0, pos 0 (padding): {my_mask[0, 0, 0, :].tolist()}")
        print(f"  batch 0, pos 3 (first real): {my_mask[0, 0, 3, :].tolist()}")
        print(f"  batch 0, pos 7 (last): {my_mask[0, 0, 7, :].tolist()}")
        print(f"  batch 1, pos 0 (padding): {my_mask[1, 0, 0, :].tolist()}")
        print(f"  batch 1, pos 1 (first real): {my_mask[1, 0, 1, :].tolist()}")

    # Compare
    print("\n### Comparison ###")
    if my_mask is not None and hf_mask_4d_unmasked is not None:
        match = torch.allclose(my_mask, hf_mask_4d_unmasked)
        max_diff = (my_mask - hf_mask_4d_unmasked).abs().max().item()
        print(f"Masks match: {match}")
        print(f"Max diff: {max_diff}")

        if not match:
            diff = (my_mask - hf_mask_4d_unmasked).abs()
            print(f"\nDifference locations (non-zero):")
            for b in range(batch_size):
                for i in range(seq_len):
                    row_diff = diff[b, 0, i, :]
                    if row_diff.max() > 0:
                        print(f"  batch {b}, pos {i}: max_diff={row_diff.max().item():.2e}")
                        print(f"    HF:   {hf_mask_4d_unmasked[b, 0, i, :].tolist()}")
                        print(f"    Mine: {my_mask[b, 0, i, :].tolist()}")

    # Now let's also compare what actually happens during forward
    print("\n### Actual Forward Comparison ###")

    # Run HF model with mask and capture intermediate attention output
    weights = dict(model.state_dict())

    with torch.no_grad():
        # Get embeddings
        embeds = model.model.embed_tokens(input_ids)

        # Get position embeddings (HF way)
        # HF computes position_ids from attention_mask
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0).long()
        print(f"\nPosition IDs from mask:")
        print(f"  batch 0: {position_ids[0].tolist()}")
        print(f"  batch 1: {position_ids[1].tolist()}")

        # Get cos/sin from HF
        hf_cos, hf_sin = model.model.rotary_emb(embeds, position_ids)

        # Get what HF passes to attention
        # In Qwen2Model._update_causal_mask, it decides whether to use mask or is_causal
        # Let's check if HF uses is_causal when possible

        # Actually run HF attention layer 0 with explicit mask
        layer0_out_with_mask, _, _ = model.model.layers[0](
            embeds,
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(1).to(dtype) * torch.finfo(dtype).min * (1 - attention_mask.unsqueeze(1).unsqueeze(1).to(dtype)),
            position_ids=position_ids,
            use_cache=False,
        )

        # This is getting complicated. Let me just run the full forward with both
        hf_logits = model(input_ids, attention_mask=attention_mask).logits

        from qwen_functional import qwen_forward
        my_logits = qwen_forward(input_ids, weights, attention_mask=attention_mask)

        print(f"\nFull forward comparison:")
        print(f"  HF logits shape: {hf_logits.shape}")
        print(f"  My logits shape: {my_logits.shape}")

        # Compare at real positions only
        for b in range(batch_size):
            first_real = (attention_mask[b] == 1).nonzero()[0].item()
            for pos in range(first_real, seq_len):
                diff = (hf_logits[b, pos] - my_logits[b, pos]).abs().max().item()
                if pos == first_real or pos == seq_len - 1 or diff > 0.1:
                    print(f"    batch {b}, pos {pos}: diff={diff:.4f}")


if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():
        debug_mask()
    else:
        print("No GPU available. Run on remote GPU.")
        from verify import run_on_gpu
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--gpu-id", type=str)
        args = parser.parse_args()
        run_on_gpu(__file__, gpu_id=args.gpu_id, keep_alive=True, vram_gb=16)
