#!/usr/bin/env python3
"""Explore Qwen3-Next model architecture.

This script is designed to run directly on a GPU instance via SSH.
It handles its own environment setup to avoid venv caching issues.
"""

import subprocess
import sys


def setup_environment():
    """Install required packages with correct versions."""
    print("Setting up environment...")

    # Install packages using uv pip (pip not available in uv-managed venvs)
    subprocess.run([
        "uv", "pip", "install", "-q",
        "huggingface_hub>=0.26.0"
    ], check=True)

    subprocess.run([
        "uv", "pip", "install", "-q",
        "git+https://github.com/huggingface/transformers.git"
    ], check=True)

    subprocess.run([
        "uv", "pip", "install", "-q",
        "accelerate", "safetensors"
    ], check=True)

    print("Environment setup complete!")


def explore():
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    # Use non-FP8 version (bfloat16) - compatible with all GPUs
    MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    print("=" * 60)
    print(f"Exploring {MODEL_NAME} Architecture")
    print("=" * 60)

    # First just get config
    print("\n### Config ###")
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"hidden_size: {config.hidden_size}")
    print(f"intermediate_size: {config.intermediate_size}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_key_value_heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
    print(f"head_dim: {getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)}")
    print(f"vocab_size: {config.vocab_size}")
    print(f"rope_theta: {getattr(config, 'rope_theta', 'N/A')}")
    print(f"rms_norm_eps: {getattr(config, 'rms_norm_eps', 'N/A')}")

    # MoE-specific attributes
    print(f"\n### MoE Config ###")
    for attr in ['num_experts', 'num_experts_per_tok', 'moe_intermediate_size',
                 'shared_expert_intermediate_size', 'router_aux_loss_coef',
                 'norm_topk_prob', 'first_k_dense_replace']:
        val = getattr(config, attr, 'N/A')
        if val != 'N/A':
            print(f"{attr}: {val}")

    # Model-specific attributes
    print(f"\n### Model-specific ###")
    for attr in ['attn_logit_softcapping', 'final_logit_softcapping', 'query_pre_attn_scalar',
                 'sliding_window', 'attention_bias', 'mlp_bias', 'original_max_position_embeddings',
                 'max_position_embeddings', 'rope_scaling', 'use_qkv_bias', 'partial_rotary_factor',
                 'linear_attention_config']:
        val = getattr(config, attr, 'N/A')
        if val != 'N/A':
            print(f"{attr}: {val}")

    # Load the actual model for structure exploration
    print("\n### Loading model... ###")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="balanced",  # Use both GPUs
        trust_remote_code=True,
    )
    model.eval()

    print("\n### Model structure ###")
    print(f"Model type: {type(model).__name__}")
    print(f"Base model type: {type(model.model).__name__}")

    # Check attention module (may be self_attn or linear_attn)
    layer0 = model.model.layers[0]
    if hasattr(layer0, 'self_attn'):
        attn = layer0.self_attn
        attn_name = "self_attn"
    elif hasattr(layer0, 'linear_attn'):
        attn = layer0.linear_attn
        attn_name = "linear_attn"
    else:
        attn = None
        attn_name = "unknown"
    print(f"\n### Attention module (layer 0): {attn_name} ###")
    print(f"Attention type: {type(attn).__name__}" if attn else "No attention module found")

    # List all parameters in attention
    if attn:
        print("\nAttention parameters:")
        for name, param in attn.named_parameters():
            print(f"  {name}: {param.shape}")

        # Check for special attributes
        print("\nSpecial attributes:")
        for attr in ['scaling', 'softcap', 'is_causal', 'attention_dropout', 'q_norm', 'k_norm',
                     'A_log', 'D', 'dt_bias', 'conv1d']:
            if hasattr(attn, attr):
                val = getattr(attn, attr)
                if hasattr(val, 'weight'):
                    print(f"  {attr}: {type(val).__name__} (has weight)")
                elif hasattr(val, 'shape'):
                    print(f"  {attr}: Tensor{list(val.shape)}")
                else:
                    print(f"  {attr}: {val}")

    # Check MLP / MoE block
    mlp = model.model.layers[0].mlp
    print(f"\n### MLP/MoE module (layer 0) ###")
    print(f"MLP type: {type(mlp).__name__}")

    # List submodules of MLP
    print("\nMLP submodules:")
    for name, module in mlp.named_children():
        print(f"  {name}: {type(module).__name__}")

    # List MLP parameters
    print("\nMLP parameters (first 20):")
    mlp_params = list(mlp.named_parameters())
    for name, param in mlp_params[:20]:
        print(f"  {name}: {param.shape}")
    if len(mlp_params) > 20:
        print(f"  ... and {len(mlp_params) - 20} more parameters")

    # Check norms
    print(f"\n### Norms ###")
    print(f"input_layernorm type: {type(model.model.layers[0].input_layernorm).__name__}")
    print(f"pre_feedforward_layernorm: {hasattr(model.model.layers[0], 'pre_feedforward_layernorm')}")
    print(f"post_feedforward_layernorm: {hasattr(model.model.layers[0], 'post_feedforward_layernorm')}")
    print(f"post_attention_layernorm: {hasattr(model.model.layers[0], 'post_attention_layernorm')}")

    # List all layer 0 submodules
    print("\n### Layer 0 submodules ###")
    for name, module in model.model.layers[0].named_children():
        print(f"  {name}: {type(module).__name__}")

    # Weight keys
    print("\n### Weight keys (sample) ###")
    weights = dict(model.state_dict())
    layer0_keys = [k for k in weights.keys() if 'layers.0.' in k]
    for k in sorted(layer0_keys)[:30]:
        print(f"  {k}: {weights[k].shape}")
    if len(layer0_keys) > 30:
        print(f"  ... and {len(layer0_keys) - 30} more keys")

    # Check if there's q_norm/k_norm
    print("\n### Q/K Norm check ###")
    q_norm_keys = [k for k in weights.keys() if 'q_norm' in k.lower() or 'k_norm' in k.lower()]
    print(f"Q/K norm keys found: {q_norm_keys[:10]}")

    # Check for MoE-specific keys
    print("\n### MoE-specific keys ###")
    moe_keys = [k for k in weights.keys() if 'expert' in k.lower() or 'gate' in k.lower() or 'router' in k.lower()]
    print(f"Found {len(moe_keys)} MoE-related keys")
    for k in sorted(moe_keys)[:15]:
        print(f"  {k}: {weights[k].shape}")
    if len(moe_keys) > 15:
        print(f"  ... and {len(moe_keys) - 15} more MoE keys")

    # Test basic forward
    print("\n### Basic forward test ###")
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda:0")
    with torch.no_grad():
        out = model(input_ids)
    print(f"Output shape: {out.logits.shape}")
    print(f"Output dtype: {out.logits.dtype}")

    print("\n" + "=" * 60)
    print("Exploration complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-only", action="store_true", help="Only setup environment")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    args = parser.parse_args()

    if not args.skip_setup:
        setup_environment()

    if not args.setup_only:
        explore()
