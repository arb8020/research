#!/usr/bin/env python3
"""Explore model architectures to understand differences."""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def explore():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    # Model to explore - change this to explore different models
    MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"  # MoE model: 80B total, 3B active

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
    print(
        f"head_dim: {getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)}"
    )
    print(f"vocab_size: {config.vocab_size}")
    print(f"rope_theta: {getattr(config, 'rope_theta', 'N/A')}")
    print(f"rms_norm_eps: {getattr(config, 'rms_norm_eps', 'N/A')}")

    # Model-specific attributes
    print("\n### Model-specific ###")
    for attr in [
        "attn_logit_softcapping",
        "final_logit_softcapping",
        "query_pre_attn_scalar",
        "sliding_window",
        "attention_bias",
        "mlp_bias",
        "original_max_position_embeddings",
        "max_position_embeddings",
        "rope_scaling",
        "use_qkv_bias",
    ]:
        val = getattr(config, attr, "N/A")
        if val != "N/A":
            print(f"{attr}: {val}")

    # Load the actual model for structure exploration
    print("\n### Loading model... ###")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    print("\n### Model structure ###")
    print(f"Model type: {type(model).__name__}")
    print(f"Base model type: {type(model.model).__name__}")

    # Check attention module
    attn = model.model.layers[0].self_attn
    print("\n### Attention module (layer 0) ###")
    print(f"Attention type: {type(attn).__name__}")

    # List all parameters in attention
    print("\nAttention parameters:")
    for name, param in attn.named_parameters():
        print(f"  {name}: {param.shape}")

    # Check for special attributes
    print("\nSpecial attributes:")
    for attr in ["scaling", "softcap", "is_causal", "attention_dropout", "q_norm", "k_norm"]:
        if hasattr(attn, attr):
            val = getattr(attn, attr)
            if hasattr(val, "weight"):
                print(f"  {attr}: {type(val).__name__} (has weight)")
            else:
                print(f"  {attr}: {val}")

    # Check MLP
    mlp = model.model.layers[0].mlp
    print("\n### MLP module (layer 0) ###")
    print(f"MLP type: {type(mlp).__name__}")
    for name, param in mlp.named_parameters():
        print(f"  {name}: {param.shape}")

    # Check norms
    print("\n### Norms ###")
    print(f"input_layernorm type: {type(model.model.layers[0].input_layernorm).__name__}")
    print(
        f"pre_feedforward_layernorm: {hasattr(model.model.layers[0], 'pre_feedforward_layernorm')}"
    )
    print(
        f"post_feedforward_layernorm: {hasattr(model.model.layers[0], 'post_feedforward_layernorm')}"
    )
    print(f"post_attention_layernorm: {hasattr(model.model.layers[0], 'post_attention_layernorm')}")

    # List all layer 0 submodules
    print("\n### Layer 0 submodules ###")
    for name, module in model.model.layers[0].named_children():
        print(f"  {name}: {type(module).__name__}")

    # Weight keys
    print("\n### Weight keys (sample) ###")
    weights = dict(model.state_dict())
    layer0_keys = [k for k in weights.keys() if "layers.0." in k]
    for k in sorted(layer0_keys)[:25]:
        print(f"  {k}: {weights[k].shape}")

    # Check if there's q_norm/k_norm
    print("\n### Q/K Norm check ###")
    q_norm_keys = [k for k in weights.keys() if "q_norm" in k.lower() or "k_norm" in k.lower()]
    print(f"Q/K norm keys found: {q_norm_keys[:10]}")

    # Test basic forward
    print("\n### Basic forward test ###")
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda:0")
    with torch.no_grad():
        out = model(input_ids)
    print(f"Output shape: {out.logits.shape}")
    print(f"Output dtype: {out.logits.dtype}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str, help="Reuse existing GPU instance")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU alive after run")
    parser.add_argument("--max-price", type=float, default=3.0, help="Max price per hour")
    parser.add_argument("--remote", action="store_true", help="Run on remote GPU")
    args = parser.parse_args()

    # Check if we have local GPU
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    if has_gpu and not args.remote:
        explore()
    else:
        print("No local GPU. Running on remote GPU...")
        from verify import run_on_gpu

        run_on_gpu(
            __file__,
            gpu_id=args.gpu_id,
            keep_alive=args.keep_alive or bool(args.gpu_id),
            vram_gb=80,  # Need large GPU for 80B model
            max_price=args.max_price,
        )
