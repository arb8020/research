#!/usr/bin/env python3
"""Debug Qwen functional implementation layer by layer."""

from __future__ import annotations

import os
import sys

# Add script dir to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def debug_on_gpu():
    """Compare each component against HF implementation."""
    import torch
    from transformers import AutoModelForCausalLM

    from qwen_functional import (
        rms_norm,
        compute_rope_embeddings,
        attention,
        mlp,
        qwen_forward,
        HIDDEN_SIZE,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
    )

    print("=" * 60)
    print("Qwen2.5-0.5B Debug - Layer by Layer Comparison")
    print("=" * 60)

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    weights = {k: v for k, v in model.state_dict().items()}
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda:0")

    # ========== Test 1: Embedding ==========
    print("\n### Test 1: Embedding ###")
    with torch.no_grad():
        hf_embed = model.model.embed_tokens(input_ids)
        my_embed = torch.nn.functional.embedding(input_ids, weights["model.embed_tokens.weight"])

    embed_match = torch.allclose(hf_embed, my_embed, rtol=1e-5, atol=1e-5)
    embed_diff = (hf_embed - my_embed).abs().max().item()
    print(f"Embedding match: {embed_match}, max_diff: {embed_diff:.2e}")
    print(f"HF shape: {hf_embed.shape}, dtype: {hf_embed.dtype}")

    # ========== Test 2: RMSNorm ==========
    print("\n### Test 2: RMSNorm (input_layernorm layer 0) ###")
    with torch.no_grad():
        hf_norm = model.model.layers[0].input_layernorm(hf_embed)
        my_norm = rms_norm(hf_embed, weights["model.layers.0.input_layernorm.weight"])

    norm_match = torch.allclose(hf_norm, my_norm, rtol=1e-5, atol=1e-5)
    norm_diff = (hf_norm - my_norm).abs().max().item()
    print(f"RMSNorm match: {norm_match}, max_diff: {norm_diff:.2e}")

    # ========== Test 3: RoPE embeddings ==========
    print("\n### Test 3: RoPE Embeddings ###")
    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device="cuda:0").unsqueeze(0).expand(batch_size, -1)

    with torch.no_grad():
        # HF computes RoPE in the rotary embedding module
        hf_cos, hf_sin = model.model.rotary_emb(hf_embed, positions)
        my_cos, my_sin = compute_rope_embeddings(positions, dtype=hf_embed.dtype)

    cos_match = torch.allclose(hf_cos, my_cos, rtol=1e-5, atol=1e-5)
    sin_match = torch.allclose(hf_sin, my_sin, rtol=1e-5, atol=1e-5)
    cos_diff = (hf_cos - my_cos).abs().max().item()
    sin_diff = (hf_sin - my_sin).abs().max().item()
    print(f"Cos match: {cos_match}, max_diff: {cos_diff:.2e}")
    print(f"Sin match: {sin_match}, max_diff: {sin_diff:.2e}")
    print(f"HF cos shape: {hf_cos.shape}, my cos shape: {my_cos.shape}")

    # ========== Test 4: Attention (layer 0) ==========
    print("\n### Test 4: Attention (layer 0) ###")
    with torch.no_grad():
        # Run HF attention
        hf_attn_out, _ = model.model.layers[0].self_attn(
            hidden_states=hf_norm,
            position_embeddings=(hf_cos, hf_sin),
            attention_mask=None,
        )

        # Run my attention
        my_attn_out = attention(
            hf_norm,
            q_weight=weights["model.layers.0.self_attn.q_proj.weight"],
            q_bias=weights["model.layers.0.self_attn.q_proj.bias"],
            k_weight=weights["model.layers.0.self_attn.k_proj.weight"],
            k_bias=weights["model.layers.0.self_attn.k_proj.bias"],
            v_weight=weights["model.layers.0.self_attn.v_proj.weight"],
            v_bias=weights["model.layers.0.self_attn.v_proj.bias"],
            o_weight=weights["model.layers.0.self_attn.o_proj.weight"],
            cos=my_cos,
            sin=my_sin,
        )

    attn_match = torch.allclose(hf_attn_out, my_attn_out, rtol=1e-5, atol=1e-5)
    attn_diff = (hf_attn_out - my_attn_out).abs().max().item()
    print(f"Attention match: {attn_match}, max_diff: {attn_diff:.2e}")

    # ========== Test 5: MLP (layer 0) ==========
    print("\n### Test 5: MLP (layer 0) ###")
    # Use the output after attention + residual
    hidden_after_attn = hf_embed + hf_attn_out
    hidden_for_mlp = model.model.layers[0].post_attention_layernorm(hidden_after_attn)

    with torch.no_grad():
        hf_mlp_out = model.model.layers[0].mlp(hidden_for_mlp)
        my_mlp_out = mlp(
            hidden_for_mlp,
            gate_weight=weights["model.layers.0.mlp.gate_proj.weight"],
            up_weight=weights["model.layers.0.mlp.up_proj.weight"],
            down_weight=weights["model.layers.0.mlp.down_proj.weight"],
        )

    mlp_match = torch.allclose(hf_mlp_out, my_mlp_out, rtol=1e-5, atol=1e-5)
    mlp_diff = (hf_mlp_out - my_mlp_out).abs().max().item()
    print(f"MLP match: {mlp_match}, max_diff: {mlp_diff:.2e}")

    # ========== Test 6: Full layer 0 ==========
    print("\n### Test 6: Full Layer 0 ###")
    with torch.no_grad():
        # HF layer 0
        hf_layer_out = model.model.layers[0](
            hf_embed,
            position_embeddings=(hf_cos, hf_sin),
        )[0]

        # My layer 0 (manual)
        residual = hf_embed
        h = rms_norm(hf_embed, weights["model.layers.0.input_layernorm.weight"])
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
        )
        h = residual + h
        residual = h
        h = rms_norm(h, weights["model.layers.0.post_attention_layernorm.weight"])
        h = mlp(
            h,
            gate_weight=weights["model.layers.0.mlp.gate_proj.weight"],
            up_weight=weights["model.layers.0.mlp.up_proj.weight"],
            down_weight=weights["model.layers.0.mlp.down_proj.weight"],
        )
        my_layer_out = residual + h

    layer_match = torch.allclose(hf_layer_out, my_layer_out, rtol=1e-5, atol=1e-5)
    layer_diff = (hf_layer_out - my_layer_out).abs().max().item()
    print(f"Layer 0 match: {layer_match}, max_diff: {layer_diff:.2e}")

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Embedding:  {'PASS' if embed_match else 'FAIL'} (diff: {embed_diff:.2e})")
    print(f"RMSNorm:    {'PASS' if norm_match else 'FAIL'} (diff: {norm_diff:.2e})")
    print(f"RoPE cos:   {'PASS' if cos_match else 'FAIL'} (diff: {cos_diff:.2e})")
    print(f"RoPE sin:   {'PASS' if sin_match else 'FAIL'} (diff: {sin_diff:.2e})")
    print(f"Attention:  {'PASS' if attn_match else 'FAIL'} (diff: {attn_diff:.2e})")
    print(f"MLP:        {'PASS' if mlp_match else 'FAIL'} (diff: {mlp_diff:.2e})")
    print(f"Layer 0:    {'PASS' if layer_match else 'FAIL'} (diff: {layer_diff:.2e})")


if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():
        debug_on_gpu()
    else:
        print("No GPU available. Run on remote GPU.")
        from verify import run_on_gpu
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--gpu-id", type=str)
        args = parser.parse_args()
        run_on_gpu(__file__, gpu_id=args.gpu_id, keep_alive=True, vram_gb=16)
