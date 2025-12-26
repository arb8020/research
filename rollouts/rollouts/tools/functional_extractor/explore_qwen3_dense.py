#!/usr/bin/env python3
"""Explore Qwen3 dense model architecture for functional extraction.

Qwen3-0.6B is small enough to run on any GPU and shares core architecture
with larger Qwen3 models.
"""

import subprocess
import sys


def setup_environment() -> None:
    """Install required packages."""
    print("Setting up environment...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "transformers>=4.51",
            "accelerate",
            "safetensors",
            "torch",
        ],
        check=True,
    )
    print("Environment ready!")


def explore() -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    MODEL_NAME = "Qwen/Qwen3-0.6B"

    print("=" * 60)
    print(f"Exploring {MODEL_NAME} Architecture")
    print("=" * 60)

    # Get config
    print("\n### Config ###")
    config = AutoConfig.from_pretrained(MODEL_NAME)
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
    print(f"tie_word_embeddings: {getattr(config, 'tie_word_embeddings', 'N/A')}")

    # Model-specific attributes
    print("\n### Model-specific ###")
    for attr in [
        "attention_bias",
        "mlp_bias",
        "max_position_embeddings",
        "rope_scaling",
        "use_sliding_window",
        "sliding_window",
        "attention_dropout",
        "use_qkv_bias",
    ]:
        val = getattr(config, attr, "N/A")
        if val != "N/A":
            print(f"{attr}: {val}")

    # Load model
    print("\n### Loading model... ###")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    print("\n### Model structure ###")
    print(f"Model type: {type(model).__name__}")
    print(f"Base model type: {type(model.model).__name__}")

    # Attention module
    attn = model.model.layers[0].self_attn
    print("\n### Attention module (layer 0) ###")
    print(f"Attention type: {type(attn).__name__}")

    print("\nAttention parameters:")
    for name, param in attn.named_parameters():
        print(f"  {name}: {param.shape}")

    print("\nAttention attributes:")
    for attr in [
        "scaling",
        "is_causal",
        "attention_dropout",
        "q_norm",
        "k_norm",
        "num_heads",
        "num_key_value_heads",
        "head_dim",
        "num_key_value_groups",
    ]:
        if hasattr(attn, attr):
            val = getattr(attn, attr)
            if hasattr(val, "weight"):
                print(f"  {attr}: {type(val).__name__} with weight {val.weight.shape}")
            else:
                print(f"  {attr}: {val}")

    # MLP module
    mlp = model.model.layers[0].mlp
    print("\n### MLP module (layer 0) ###")
    print(f"MLP type: {type(mlp).__name__}")

    print("\nMLP parameters:")
    for name, param in mlp.named_parameters():
        print(f"  {name}: {param.shape}")

    # Norms
    print("\n### Norms ###")
    layer0 = model.model.layers[0]
    print(f"input_layernorm: {type(layer0.input_layernorm).__name__}")
    if hasattr(layer0, "post_attention_layernorm"):
        print(f"post_attention_layernorm: {type(layer0.post_attention_layernorm).__name__}")
    if hasattr(layer0, "pre_feedforward_layernorm"):
        print(f"pre_feedforward_layernorm: {type(layer0.pre_feedforward_layernorm).__name__}")

    # Layer submodules
    print("\n### Layer 0 submodules ###")
    for name, module in layer0.named_children():
        print(f"  {name}: {type(module).__name__}")

    # Weight keys sample
    print("\n### Weight keys (layer 0) ###")
    weights = dict(model.state_dict())
    layer0_keys = sorted([k for k in weights.keys() if "layers.0." in k])
    for k in layer0_keys:
        print(f"  {k}: {weights[k].shape}")

    # Check embeddings
    print("\n### Embeddings ###")
    print(f"embed_tokens: {weights['model.embed_tokens.weight'].shape}")
    if "lm_head.weight" in weights:
        print(f"lm_head: {weights['lm_head.weight'].shape}")
        # Check if tied
        tied = torch.equal(weights["model.embed_tokens.weight"], weights["lm_head.weight"])
        print(f"tied_embeddings: {tied}")
    else:
        print("lm_head: NOT PRESENT (tied to embed_tokens)")

    # Forward test
    print("\n### Forward test ###")
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda:0")
    with torch.no_grad():
        out = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {out.logits.shape}")
    print(f"Output dtype: {out.logits.dtype}")

    # Print key differences from Qwen2.5
    print("\n### Key Architecture Notes for Functional Implementation ###")
    print("1. Check if Q/K norms are used")
    print("2. Check bias in attention projections")
    print("3. Check if embeddings are tied")
    print("4. RoPE configuration")

    print("\n" + "=" * 60)
    print("Exploration complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-setup", action="store_true")
    args = parser.parse_args()

    if not args.skip_setup:
        setup_environment()

    explore()
