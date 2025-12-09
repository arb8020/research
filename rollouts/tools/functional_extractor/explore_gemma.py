#!/usr/bin/env python3
"""Explore Phi-3 architecture to understand what's different."""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def explore():
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    print("=" * 60)
    print("Exploring Phi-3-mini Architecture")
    print("=" * 60)

    # First just get config
    print("\n### Config ###")
    config = AutoConfig.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
    print(f"hidden_size: {config.hidden_size}")
    print(f"intermediate_size: {config.intermediate_size}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_key_value_heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
    print(f"head_dim: {getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)}")
    print(f"vocab_size: {config.vocab_size}")
    print(f"rope_theta: {getattr(config, 'rope_theta', 'N/A')}")
    print(f"rms_norm_eps: {getattr(config, 'rms_norm_eps', 'N/A')}")

    # Model-specific attributes
    print(f"\n### Model-specific ###")
    for attr in ['attn_logit_softcapping', 'final_logit_softcapping', 'query_pre_attn_scalar',
                 'sliding_window', 'attention_bias', 'mlp_bias', 'original_max_position_embeddings',
                 'max_position_embeddings', 'rope_scaling', 'use_qkv_bias']:
        val = getattr(config, attr, 'N/A')
        if val != 'N/A':
            print(f"{attr}: {val}")

    # Load model to see structure
    print("\n### Loading model... ###")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
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
    print(f"\n### Attention module (layer 0) ###")
    print(f"Attention type: {type(attn).__name__}")

    # List all parameters in attention
    print("\nAttention parameters:")
    for name, param in attn.named_parameters():
        print(f"  {name}: {param.shape}")

    # Check for special attributes
    print("\nSpecial attributes:")
    for attr in ['scaling', 'softcap', 'is_causal', 'attention_dropout']:
        if hasattr(attn, attr):
            print(f"  {attr}: {getattr(attn, attr)}")

    # Check MLP
    mlp = model.model.layers[0].mlp
    print(f"\n### MLP module (layer 0) ###")
    print(f"MLP type: {type(mlp).__name__}")
    for name, param in mlp.named_parameters():
        print(f"  {name}: {param.shape}")

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
    for k in sorted(layer0_keys)[:20]:
        print(f"  {k}: {weights[k].shape}")

    # Check if there's q_norm/k_norm
    print("\n### Q/K Norm check ###")
    q_norm_keys = [k for k in weights.keys() if 'q_norm' in k.lower() or 'k_norm' in k.lower()]
    print(f"Q/K norm keys found: {q_norm_keys[:5]}")

    # Test basic forward
    print("\n### Basic forward test ###")
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda:0")
    with torch.no_grad():
        out = model(input_ids)
    print(f"Output shape: {out.logits.shape}")
    print(f"Output dtype: {out.logits.dtype}")


if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():
        explore()
    else:
        print("No GPU available. Run on remote GPU.")
        from verify import run_on_gpu
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--gpu-id", type=str)
        args = parser.parse_args()
        run_on_gpu(__file__, gpu_id=args.gpu_id, keep_alive=True, vram_gb=24)
