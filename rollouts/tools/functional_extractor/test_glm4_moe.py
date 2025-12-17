#!/usr/bin/env python3
"""Test functional GLM-4.5 MoE implementation against HuggingFace model.

Downloads only the required shards for testing partial layers.

Run locally (requires GPU with ~40GB VRAM for 5 layers):
    python -m tools.functional_extractor.test_glm4_moe

Run on remote GPU:
    python -m tools.functional_extractor.test_glm4_moe --remote
    python -m tools.functional_extractor.test_glm4_moe --remote --gpu-id <id>

Options:
    --num-layers N    Number of layers to test (default: 5)
    --layer-by-layer  Test each layer individually to find divergence
"""

from __future__ import annotations

import argparse
import sys


def test_on_gpu(num_layers: int = 5, layer_by_layer: bool = False) -> None:  # noqa: PLR0915
    """Run verification test on GPU."""
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import torch
    from glm4_moe_functional import (
        FIRST_K_DENSE_REPLACE,
        glm4_moe_forward,
    )
    from transformers import AutoConfig, AutoModelForCausalLM

    print("=" * 60)
    print(f"GLM-4.5 MoE Functional Implementation Test ({num_layers} layers)")
    print("=" * 60)

    # Determine which shards we need
    # Shard 1: embed + layer 0, Shard 2: layer 1, Shard 3: layer 2, Shard 4+: MoE layers
    shards_needed = min(num_layers + 1, 93)  # +1 for embeddings in shard 1
    print(f"\nWill download shards 1-{shards_needed} (~{estimate_download_size(shards_needed)} GB)")

    # Load model with only required layers
    print("\nLoading model configuration...")
    config = AutoConfig.from_pretrained("zai-org/GLM-4.5", trust_remote_code=True)

    # Modify config to only load num_layers
    original_num_layers = config.num_hidden_layers
    config.num_hidden_layers = num_layers

    print(f"Loading model with {num_layers} layers (original: {original_num_layers})...")
    model = AutoModelForCausalLM.from_pretrained(
        "zai-org/GLM-4.5",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    weights = dict(model.state_dict())
    print(f"Loaded {len(weights)} weight tensors")

    # Test inputs
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda:0")

    if layer_by_layer:
        test_layer_by_layer(model, weights, input_ids, num_layers)
    else:
        test_full_forward(model, weights, input_ids, num_layers)


def estimate_download_size(shards: int) -> float:
    """Estimate download size in GB."""
    # Shard 1: 3.75 GB, Shards 2-3: 0.65 GB each, Shards 4+: 7.87 GB each
    size = 3.75
    if shards >= 2:
        size += 0.65
    if shards >= 3:
        size += 0.65
    if shards >= 4:
        size += (shards - 3) * 7.87
    return round(size, 1)


def test_full_forward(
    model: "AutoModelForCausalLM",
    weights: dict,
    input_ids: "torch.Tensor",
    num_layers: int,
) -> None:
    """Test full forward pass."""
    import torch
    from glm4_moe_functional import glm4_moe_forward

    print("\n### Full Forward Pass Test ###")

    with torch.no_grad():
        # HuggingFace forward
        hf_output = model(input_ids)
        hf_logits = hf_output.logits

        # Functional forward
        func_logits = glm4_moe_forward(input_ids, weights, num_layers=num_layers)

    # Compare
    matches = torch.allclose(hf_logits, func_logits, rtol=1e-4, atol=1e-4)
    max_diff = (hf_logits - func_logits).abs().max().item()

    status = "PASS" if matches else "FAIL"
    print(f"  Full forward: max_diff={max_diff:.2e} [{status}]")

    if not matches:
        print(f"\n  HF logits shape: {hf_logits.shape}")
        print(f"  Func logits shape: {func_logits.shape}")
        print(f"  HF logits sample: {hf_logits[0, 0, :5]}")
        print(f"  Func logits sample: {func_logits[0, 0, :5]}")

        # Find where the max diff is
        diff = (hf_logits - func_logits).abs()
        max_idx = diff.argmax()
        batch_idx = max_idx // (diff.shape[1] * diff.shape[2])
        seq_idx = (max_idx % (diff.shape[1] * diff.shape[2])) // diff.shape[2]
        vocab_idx = max_idx % diff.shape[2]
        print(f"  Max diff at: batch={batch_idx}, seq={seq_idx}, vocab={vocab_idx}")

        sys.exit(1)


def test_layer_by_layer(
    model: "AutoModelForCausalLM",
    weights: dict,
    input_ids: "torch.Tensor",
    num_layers: int,
) -> None:
    """Test each layer individually to find divergence point."""
    import torch
    import torch.nn.functional as F
    from glm4_moe_functional import (
        FIRST_K_DENSE_REPLACE,
        compute_rope_embeddings,
        rms_norm,
        transformer_layer,
    )

    print("\n### Layer-by-Layer Test ###")

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Test embeddings
    with torch.no_grad():
        hf_embed = model.model.embed_tokens(input_ids)
        func_embed = F.embedding(input_ids, weights["model.embed_tokens.weight"])

    embed_diff = (hf_embed - func_embed).abs().max().item()
    embed_match = embed_diff < 1e-5
    print(f"  Embeddings: max_diff={embed_diff:.2e} [{'PASS' if embed_match else 'FAIL'}]")

    if not embed_match:
        print("  Stopping at embeddings divergence")
        return

    # Compute RoPE
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    cos, sin = compute_rope_embeddings(positions, dtype=hf_embed.dtype)

    # Test each layer
    hf_hidden = hf_embed
    func_hidden = func_embed.clone()

    for layer_idx in range(num_layers):
        layer_type = "dense" if layer_idx < FIRST_K_DENSE_REPLACE else "MoE"

        with torch.no_grad():
            # HF layer forward
            hf_layer = model.model.layers[layer_idx]
            hf_out = hf_layer(hf_hidden, position_embeddings=(cos, sin))
            if isinstance(hf_out, tuple):
                hf_out = hf_out[0]

            # Functional layer forward
            func_out = transformer_layer(func_hidden, weights, layer_idx, cos, sin)

        layer_diff = (hf_out - func_out).abs().max().item()
        layer_match = layer_diff < 1e-4

        status = "PASS" if layer_match else "FAIL"
        print(f"  Layer {layer_idx:2d} ({layer_type:5s}): max_diff={layer_diff:.2e} [{status}]")

        if not layer_match:
            print(f"\n  ### DIVERGENCE AT LAYER {layer_idx} ({layer_type}) ###")
            debug_layer(model, weights, layer_idx, hf_hidden, cos, sin)
            return

        # Update hidden states for next layer
        hf_hidden = hf_out
        func_hidden = func_out

    # Test final norm
    with torch.no_grad():
        hf_normed = model.model.norm(hf_hidden)
        func_normed = rms_norm(func_hidden, weights["model.norm.weight"])

    norm_diff = (hf_normed - func_normed).abs().max().item()
    norm_match = norm_diff < 1e-4
    print(f"  Final norm: max_diff={norm_diff:.2e} [{'PASS' if norm_match else 'FAIL'}]")

    # Test LM head
    with torch.no_grad():
        hf_logits = model.lm_head(hf_normed)
        func_logits = F.linear(func_normed, weights["model.embed_tokens.weight"])

    logits_diff = (hf_logits - func_logits).abs().max().item()
    logits_match = logits_diff < 1e-4
    print(f"  LM head: max_diff={logits_diff:.2e} [{'PASS' if logits_match else 'FAIL'}]")

    if embed_match and norm_match and logits_match:
        print(f"\n  All {num_layers} layers PASSED!")


def debug_layer(
    model: "AutoModelForCausalLM",
    weights: dict,
    layer_idx: int,
    hidden_states: "torch.Tensor",
    cos: "torch.Tensor",
    sin: "torch.Tensor",
) -> None:
    """Debug a specific layer to find component divergence."""
    import torch
    from glm4_moe_functional import (
        FIRST_K_DENSE_REPLACE,
        attention,
        dense_mlp,
        moe_block,
        rms_norm,
    )

    prefix = f"model.layers.{layer_idx}"
    hf_layer = model.model.layers[layer_idx]

    print("\n  Debugging layer components...")

    with torch.no_grad():
        # Input layernorm
        hf_normed = hf_layer.input_layernorm(hidden_states)
        func_normed = rms_norm(hidden_states, weights[f"{prefix}.input_layernorm.weight"])
        norm_diff = (hf_normed - func_normed).abs().max().item()
        print(f"    input_layernorm: max_diff={norm_diff:.2e}")

        # Attention
        hf_attn_out = None

        def capture_attn(module, input, output):
            nonlocal hf_attn_out
            hf_attn_out = output[0] if isinstance(output, tuple) else output

        hook = hf_layer.self_attn.register_forward_hook(capture_attn)
        _ = hf_layer(hidden_states, position_embeddings=(cos, sin))
        hook.remove()

        func_attn_out = attention(
            func_normed,
            q_weight=weights[f"{prefix}.self_attn.q_proj.weight"],
            q_bias=weights[f"{prefix}.self_attn.q_proj.bias"],
            k_weight=weights[f"{prefix}.self_attn.k_proj.weight"],
            k_bias=weights[f"{prefix}.self_attn.k_proj.bias"],
            v_weight=weights[f"{prefix}.self_attn.v_proj.weight"],
            v_bias=weights[f"{prefix}.self_attn.v_proj.bias"],
            o_weight=weights[f"{prefix}.self_attn.o_proj.weight"],
            q_norm_weight=weights[f"{prefix}.self_attn.q_norm.weight"],
            k_norm_weight=weights[f"{prefix}.self_attn.k_norm.weight"],
            cos=cos,
            sin=sin,
        )

        if hf_attn_out is not None:
            attn_diff = (hf_attn_out - func_attn_out).abs().max().item()
            print(f"    attention: max_diff={attn_diff:.2e}")

        # Post-attention residual
        hf_post_attn = hidden_states + hf_attn_out
        func_post_attn = hidden_states + func_attn_out

        # Post-attention layernorm
        hf_post_normed = hf_layer.post_attention_layernorm(hf_post_attn)
        func_post_normed = rms_norm(func_post_attn, weights[f"{prefix}.post_attention_layernorm.weight"])
        post_norm_diff = (hf_post_normed - func_post_normed).abs().max().item()
        print(f"    post_attention_layernorm: max_diff={post_norm_diff:.2e}")

        # MLP/MoE
        if layer_idx < FIRST_K_DENSE_REPLACE:
            func_mlp_out = dense_mlp(
                func_post_normed,
                gate_weight=weights[f"{prefix}.mlp.gate_proj.weight"],
                up_weight=weights[f"{prefix}.mlp.up_proj.weight"],
                down_weight=weights[f"{prefix}.mlp.down_proj.weight"],
            )
            print("    MLP type: dense")
        else:
            func_mlp_out = moe_block(func_post_normed, weights, layer_idx)
            print("    MLP type: MoE")

        # Capture HF MLP output
        hf_mlp_out = None

        def capture_mlp(module, input, output):
            nonlocal hf_mlp_out
            hf_mlp_out = output

        hook = hf_layer.mlp.register_forward_hook(capture_mlp)
        _ = hf_layer(hidden_states, position_embeddings=(cos, sin))
        hook.remove()

        if hf_mlp_out is not None:
            mlp_diff = (hf_mlp_out - func_mlp_out).abs().max().item()
            print(f"    mlp: max_diff={mlp_diff:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test functional GLM-4.5 MoE implementation")
    parser.add_argument("--remote", action="store_true", help="Run on remote GPU")
    parser.add_argument("--gpu-id", type=str, help="Reuse existing GPU instance")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU alive after test")
    parser.add_argument("--num-layers", type=int, default=5, help="Number of layers to test")
    parser.add_argument("--layer-by-layer", action="store_true", help="Test each layer individually")
    args = parser.parse_args()

    if args.remote:
        from tools.functional_extractor.config import DeploymentConfig
        from tools.functional_extractor.verify import run_on_gpu

        # GLM-4.5 needs more VRAM - estimate based on layers
        vram_needed = 24 if args.num_layers <= 3 else 48 if args.num_layers <= 5 else 80

        run_on_gpu(
            script_path=__file__,
            deployment=DeploymentConfig(vram_gb=vram_needed),
            gpu_id=args.gpu_id,
            keep_alive=args.keep_alive or bool(args.gpu_id),
        )
    else:
        import torch

        if not torch.cuda.is_available():
            print("No GPU available locally. Use --remote to run on remote GPU.")
            sys.exit(1)

        test_on_gpu(num_layers=args.num_layers, layer_by_layer=args.layer_by_layer)


if __name__ == "__main__":
    main()
