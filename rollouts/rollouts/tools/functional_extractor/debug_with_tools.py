#!/usr/bin/env python3
"""Debug attention mask issue using analysis tools."""

from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def debug_with_tools():
    import torch
    from analysis_tools import (
        capture_intermediates,
    )
    from transformers import AutoModelForCausalLM

    print("=" * 60)
    print("Debug with Analysis Tools")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    weights = dict(model.state_dict())

    # Test case with padding
    input_ids = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, :3] = 0  # First 3 are padding

    print(f"\nInput IDs: {input_ids.shape}")
    print(f"Attention mask: {attention_mask[0].tolist()}")

    # Capture HF intermediates
    print("\n### Capturing HF intermediates ###")
    hf_captures = capture_intermediates(
        model,
        (input_ids,),
        {"attention_mask": attention_mask},
        layer_pattern="layers.0",  # Just first layer for now
    )
    print(f"Captured {len(hf_captures)} intermediate values")
    for c in hf_captures[:10]:
        print(f"  {c.name}: {c.shape}")

    # Now we need to capture from our functional implementation
    # But our functional code doesn't have hooks - we need to compare at specific points
    # Let's compare layer by layer manually

    print("\n### Manual layer-by-layer comparison ###")
    from qwen_functional import (
        attention,
        compute_rope_embeddings,
        create_causal_mask,
        mlp,
        rms_norm,
    )

    # Step 1: Embeddings
    with torch.no_grad():
        hf_embed = model.model.embed_tokens(input_ids)
        my_embed = torch.nn.functional.embedding(input_ids, weights["model.embed_tokens.weight"])

    embed_diff = (hf_embed - my_embed).abs().max().item()
    print(f"Embedding: diff={embed_diff:.2e} {'✓' if embed_diff < 1e-5 else '✗'}")

    # Step 2: Position IDs and RoPE
    # HF computes position_ids from attention_mask
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0).long()
    print(f"Position IDs: {position_ids[0].tolist()}")

    with torch.no_grad():
        hf_cos, hf_sin = model.model.rotary_emb(hf_embed, position_ids)
        my_cos, my_sin = compute_rope_embeddings(position_ids, dtype=hf_embed.dtype)

    cos_diff = (hf_cos - my_cos).abs().max().item()
    sin_diff = (hf_sin - my_sin).abs().max().item()
    print(f"RoPE cos: diff={cos_diff:.2e} {'✓' if cos_diff < 1e-5 else '✗'}")
    print(f"RoPE sin: diff={sin_diff:.2e} {'✓' if sin_diff < 1e-5 else '✗'}")

    # Step 3: Layer 0 input norm
    with torch.no_grad():
        hf_norm = model.model.layers[0].input_layernorm(hf_embed)
        my_norm = rms_norm(my_embed, weights["model.layers.0.input_layernorm.weight"])

    norm_diff = (hf_norm - my_norm).abs().max().item()
    print(f"Layer 0 input norm: diff={norm_diff:.2e} {'✓' if norm_diff < 1e-5 else '✗'}")

    # Step 4: Create attention mask
    seq_len = input_ids.shape[1]
    my_attn_mask = create_causal_mask(seq_len, input_ids.device, hf_embed.dtype, attention_mask)

    # Get HF's mask via _update_causal_mask
    cache_position = torch.arange(seq_len, device="cuda:0")
    hf_attn_mask = model.model._update_causal_mask(
        attention_mask=attention_mask,
        input_tensor=hf_embed,
        cache_position=cache_position,
        past_key_values=None,
        output_attentions=False,
    )

    if hf_attn_mask is not None and my_attn_mask is not None:
        mask_diff = (hf_attn_mask - my_attn_mask).abs().max().item()
        print(f"Attention mask: diff={mask_diff:.2e} {'✓' if mask_diff < 1e-5 else '✗'}")
    else:
        print(f"Attention mask: HF={hf_attn_mask is not None}, Mine={my_attn_mask is not None}")

    # Step 5: Layer 0 attention output
    with torch.no_grad():
        # HF attention
        hf_attn_out = model.model.layers[0].self_attn(
            hidden_states=hf_norm,
            attention_mask=hf_attn_mask,
            position_embeddings=(hf_cos, hf_sin),
        )[0]

        # My attention
        my_attn_out = attention(
            my_norm,
            q_weight=weights["model.layers.0.self_attn.q_proj.weight"],
            q_bias=weights["model.layers.0.self_attn.q_proj.bias"],
            k_weight=weights["model.layers.0.self_attn.k_proj.weight"],
            k_bias=weights["model.layers.0.self_attn.k_proj.bias"],
            v_weight=weights["model.layers.0.self_attn.v_proj.weight"],
            v_bias=weights["model.layers.0.self_attn.v_proj.bias"],
            o_weight=weights["model.layers.0.self_attn.o_proj.weight"],
            cos=my_cos,
            sin=my_sin,
            attention_mask=my_attn_mask,
        )

    attn_diff = (hf_attn_out - my_attn_out).abs().max().item()
    print(f"Layer 0 attention: diff={attn_diff:.2e} {'✓' if attn_diff < 1e-5 else '✗'}")

    # Per-position diff
    print("\n  Per-position attention diff:")
    for pos in range(seq_len):
        pos_diff = (hf_attn_out[0, pos] - my_attn_out[0, pos]).abs().max().item()
        is_real = attention_mask[0, pos].item() == 1
        marker = "  (padding)" if not is_real else ""
        print(f"    pos {pos}: {pos_diff:.4f}{marker}")

    # If attention differs, dig deeper into Q, K, V
    if attn_diff > 1e-4:
        print("\n### Debugging attention internals ###")
        import torch.nn.functional as F

        # Q, K, V projections
        hf_q = model.model.layers[0].self_attn.q_proj(hf_norm)
        hf_k = model.model.layers[0].self_attn.k_proj(hf_norm)
        hf_v = model.model.layers[0].self_attn.v_proj(hf_norm)

        my_q = F.linear(
            my_norm,
            weights["model.layers.0.self_attn.q_proj.weight"],
            weights["model.layers.0.self_attn.q_proj.bias"],
        )
        my_k = F.linear(
            my_norm,
            weights["model.layers.0.self_attn.k_proj.weight"],
            weights["model.layers.0.self_attn.k_proj.bias"],
        )
        my_v = F.linear(
            my_norm,
            weights["model.layers.0.self_attn.v_proj.weight"],
            weights["model.layers.0.self_attn.v_proj.bias"],
        )

        q_diff = (hf_q - my_q).abs().max().item()
        k_diff = (hf_k - my_k).abs().max().item()
        v_diff = (hf_v - my_v).abs().max().item()

        print(f"Q proj: diff={q_diff:.2e}")
        print(f"K proj: diff={k_diff:.2e}")
        print(f"V proj: diff={v_diff:.2e}")

    # Run full forward and check where it diverges
    print("\n### Full forward - per-layer diff ###")
    from qwen_functional import qwen_forward

    # Run HF full forward
    with torch.no_grad():
        hf_logits = model(input_ids, attention_mask=attention_mask).logits

    # Run my forward
    with torch.no_grad():
        my_logits = qwen_forward(input_ids, weights, attention_mask=attention_mask)

    logits_diff = (hf_logits - my_logits).abs().max().item()
    print(f"Full forward logits: diff={logits_diff:.2e}")

    # Per-position logits diff
    print("\n  Per-position logits diff:")
    for pos in range(seq_len):
        pos_diff = (hf_logits[0, pos] - my_logits[0, pos]).abs().max().item()
        is_real = attention_mask[0, pos].item() == 1
        marker = "  (padding)" if not is_real else ""
        status = "✗" if pos_diff > 0.01 else "✓"
        print(f"    pos {pos}: {pos_diff:.4f} {status}{marker}")

    # Check each layer's output by running incrementally
    print("\n### Per-layer accumulation ###")
    # This requires modifying qwen_forward to output intermediates, or running HF layer by layer
    # For now, check if layer 0's output feeds correctly to layer 1

    # Layer 0 full output
    with torch.no_grad():
        hf_h = hf_embed
        for i in range(1):  # Just layer 0
            hf_h = model.model.layers[i](
                hf_h,
                attention_mask=hf_attn_mask,
                position_embeddings=(hf_cos, hf_sin),
            )[0]

        my_h = my_embed
        residual = my_h
        h = rms_norm(my_h, weights["model.layers.0.input_layernorm.weight"])
        h = attention(
            h,
            q_weight=weights["model.layers.0.self_attn.q_proj.weight"],
            q_bias=weights["model.layers.0.self_attn.q_proj.bias"],
            k_weight=weights["model.layers.0.self_attn.k_proj.weight"],
            k_bias=weights["model.layers.0.self_attn.k_proj.bias"],
            v_weight=weights["model.layers.0.self_attn.v_proj.weight"],
            v_bias=weights["model.layers.0.self_attn.v_proj.bias"],
            o_weight=weights["model.layers.0.self_attn.o_proj.weight"],
            cos=my_cos,
            sin=my_sin,
            attention_mask=my_attn_mask,
        )
        my_h = residual + h
        residual = my_h
        h = rms_norm(my_h, weights["model.layers.0.post_attention_layernorm.weight"])
        h = mlp(
            h,
            gate_weight=weights["model.layers.0.mlp.gate_proj.weight"],
            up_weight=weights["model.layers.0.mlp.up_proj.weight"],
            down_weight=weights["model.layers.0.mlp.down_proj.weight"],
        )
        my_h = residual + h

    layer0_diff = (hf_h - my_h).abs().max().item()
    print(f"After layer 0: diff={layer0_diff:.2e} {'✓' if layer0_diff < 1e-5 else '✗'}")

    # Check ALL layers one by one
    print("\n### Per-layer diff accumulation ###")
    from qwen_functional import transformer_layer

    hf_h = hf_embed
    my_h = my_embed

    for layer_idx in range(24):
        with torch.no_grad():
            hf_h = model.model.layers[layer_idx](
                hf_h,
                attention_mask=hf_attn_mask,
                position_embeddings=(hf_cos, hf_sin),
            )[0]

            my_h = transformer_layer(my_h, weights, layer_idx, my_cos, my_sin, my_attn_mask)

        layer_diff = (hf_h - my_h).abs().max().item()
        # Only print if diff > threshold or every 4 layers
        if layer_diff > 1e-5 or layer_idx % 4 == 0:
            status = "✓" if layer_diff < 1e-5 else "✗"
            print(f"  Layer {layer_idx:2d}: diff={layer_diff:.2e} {status}")

    # After all layers, check final norm and lm_head
    print("\n### After all layers ###")
    print(f"Hidden state after layer 23: diff={(hf_h - my_h).abs().max().item():.2e}")

    # Final RMSNorm
    hf_normed = model.model.norm(hf_h)
    my_normed = rms_norm(my_h, weights["model.norm.weight"])
    norm_diff = (hf_normed - my_normed).abs().max().item()
    print(f"After final norm: diff={norm_diff:.2e}")

    # LM Head
    # HF ties weights: model.lm_head.weight === model.model.embed_tokens.weight
    hf_logits_check = model.lm_head(hf_normed)
    import torch.nn.functional as F

    my_logits_check = F.linear(my_normed, weights["model.embed_tokens.weight"])
    lm_head_diff = (hf_logits_check - my_logits_check).abs().max().item()
    print(f"After LM head: diff={lm_head_diff:.2e}")

    # Sanity check: compare HF full forward vs this step-by-step
    hf_full = model(input_ids, attention_mask=attention_mask).logits
    step_by_step_diff = (hf_full - hf_logits_check).abs().max().item()
    print(f"HF full vs step-by-step: diff={step_by_step_diff:.2e}")

    # Check sliding window config
    print("\n### Sliding Window Analysis ###")
    print(f"Config sliding_window: {model.config.sliding_window}")
    print(f"Config max_window_layers: {getattr(model.config, 'max_window_layers', 'N/A')}")

    # Check each layer's attention_type
    for i, layer in enumerate(model.model.layers):
        attn_type = getattr(layer.self_attn, "attention_type", "unknown")
        if i < 4 or i > 20 or attn_type != "unknown":
            print(f"Layer {i}: attention_type={attn_type}")

    # Check what mask model.model._update_causal_mask returns for full forward
    # vs what create_causal_mask returns
    print("\n### Mask from model.forward() ###")

    # We need to see what mask is actually used inside model.forward()
    # Let's hook into model.model to see
    masks_captured = []

    def capture_mask_hook(module, args, kwargs):
        if "attention_mask" in kwargs:
            mask = kwargs["attention_mask"]
            if isinstance(mask, dict):
                masks_captured.append(("dict", list(mask.keys())))
            elif mask is not None:
                masks_captured.append(("tensor", mask.shape, mask.dtype, mask.clone()))
            else:
                masks_captured.append(("None",))
        return None

    # Hook the first layer to see what mask it receives
    hook = model.model.layers[0].register_forward_pre_hook(capture_mask_hook, with_kwargs=True)

    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)

    hook.remove()

    # Capture position_embeddings too
    position_embeddings_captured = []

    def capture_pos_hook(module, args, kwargs):
        if "position_embeddings" in kwargs:
            pos_emb = kwargs["position_embeddings"]
            if pos_emb is not None:
                position_embeddings_captured.append((pos_emb[0].clone(), pos_emb[1].clone()))
        return None

    hook2 = model.model.layers[0].register_forward_pre_hook(capture_pos_hook, with_kwargs=True)

    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)

    hook2.remove()

    if position_embeddings_captured:
        captured_cos, captured_sin = position_embeddings_captured[0]
        print("\nPosition embeddings from full forward:")
        print(f"  cos shape: {captured_cos.shape}, sin shape: {captured_sin.shape}")

        cos_diff = (captured_cos - my_cos).abs().max().item()
        sin_diff = (captured_sin - my_sin).abs().max().item()
        print(f"  cos diff: {cos_diff:.2e}")
        print(f"  sin diff: {sin_diff:.2e}")

        if cos_diff > 1e-5:
            print(f"\n  Captured cos[0, :5, :4]: {captured_cos[0, :5, :4].tolist()}")
            print(f"  My cos[0, :5, :4]:       {my_cos[0, :5, :4].tolist()}")

    if masks_captured and len(masks_captured[0]) > 3:
        captured_mask = masks_captured[0][3]
        print(f"Masks received by layer 0 during full forward: shape={captured_mask.shape}")

        # Compare with our mask
        mask_diff = (captured_mask - my_attn_mask).abs().max().item()
        print(f"Captured mask vs my_attn_mask: diff={mask_diff:.2e}")

        # Show values
        print(f"\nCaptured mask at pos 3 (first real): {captured_mask[0, 0, 3, :].tolist()}")
        print(f"My mask at pos 3 (first real):       {my_attn_mask[0, 0, 3, :].tolist()}")
        print(f"\nCaptured mask at pos 5: {captured_mask[0, 0, 5, :].tolist()}")
        print(f"My mask at pos 5:       {my_attn_mask[0, 0, 5, :].tolist()}")
    else:
        print(f"Masks received by layer 0 during full forward: {masks_captured}")


if __name__ == "__main__":
    import torch

    if torch.cuda.is_available():
        debug_with_tools()
    else:
        print("No GPU available. Run on remote GPU.")
        import argparse

        from verify import run_on_gpu

        parser = argparse.ArgumentParser()
        parser.add_argument("--gpu-id", type=str)
        args = parser.parse_args()
        run_on_gpu(__file__, gpu_id=args.gpu_id, keep_alive=True, vram_gb=16)
