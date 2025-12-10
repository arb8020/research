#!/usr/bin/env python3
"""Explore Qwen2.5-0.5B architecture to understand what we need to implement.

Run locally (no GPU needed for structure inspection):
    python -m tools.functional_extractor.explore_qwen --local

Run on GPU (for actual forward pass):
    python -m tools.functional_extractor.explore_qwen
    python -m tools.functional_extractor.explore_qwen --gpu-id <id>  # reuse
"""

from __future__ import annotations

import argparse
import sys


def explore_local():
    """Explore model structure without loading weights (fast, no GPU)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    print("=" * 60)
    print("Qwen2.5-0.5B Architecture Exploration (local)")
    print("=" * 60)

    # Load config only (no weights)
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    print("\n### Config ###")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")
    print(f"Num KV heads: {config.num_key_value_heads}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Max position embeddings: {config.max_position_embeddings}")
    print(f"RoPE theta: {config.rope_theta}")
    print(f"RMS norm eps: {config.rms_norm_eps}")
    print(f"Tie word embeddings: {config.tie_word_embeddings}")

    # Load model structure (meta device = no actual weights)
    print("\n### Module Structure ###")
    import torch
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # Print module tree
    def print_modules(module, prefix="", max_depth=4, depth=0):
        if depth >= max_depth:
            return
        for name, child in module.named_children():
            child_class = child.__class__.__name__
            print(f"{prefix}{name}: {child_class}")
            print_modules(child, prefix + "  ", max_depth, depth + 1)

    print_modules(model)

    # Print parameter shapes from config
    print("\n### Expected Weight Shapes ###")
    h = config.hidden_size
    i = config.intermediate_size
    v = config.vocab_size
    n_heads = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head_dim = h // n_heads

    print(f"embed_tokens.weight: ({v}, {h})")
    print(f"lm_head.weight: ({v}, {h})")
    print(f"norm.weight: ({h},)")
    print()
    print("Per layer:")
    print(f"  input_layernorm.weight: ({h},)")
    print(f"  post_attention_layernorm.weight: ({h},)")
    print(f"  self_attn.q_proj.weight: ({n_heads * head_dim}, {h})")
    print(f"  self_attn.k_proj.weight: ({n_kv * head_dim}, {h})")
    print(f"  self_attn.v_proj.weight: ({n_kv * head_dim}, {h})")
    print(f"  self_attn.o_proj.weight: ({h}, {n_heads * head_dim})")
    print(f"  mlp.gate_proj.weight: ({i}, {h})")
    print(f"  mlp.up_proj.weight: ({i}, {h})")
    print(f"  mlp.down_proj.weight: ({h}, {i})")


def explore_gpu():
    """Full exploration with loaded model on GPU."""
    import torch
    from transformers import AutoModelForCausalLM

    # Add parent to path for imports
    sys.path.insert(0, str(__file__).rsplit("/tools/", 1)[0])
    from tools.functional_extractor.tools import (
        capture_intermediate,
        get_weight_info,
        list_modules,
        read_module_source,
    )

    print("=" * 60)
    print("Qwen2.5-0.5B Architecture Exploration (GPU)")
    print("=" * 60)

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    # List all modules
    print("\n### All Modules ###")
    for path in list_modules(model, max_depth=4):
        print(f"  {path}")

    # Get weight info
    print("\n### All Weights ###")
    for w in get_weight_info(model):
        print(f"  {w.name}: {w.shape} ({w.dtype})")

    # Read attention source
    print("\n### Attention Module Source ###")
    attn_source = read_module_source(model, "model.layers.0.self_attn")
    print(f"Class: {attn_source.class_name}")
    print(f"File: {attn_source.file_path}")
    print("Source (first 2000 chars):")
    print(attn_source.source[:2000])

    # Read MLP source
    print("\n### MLP Module Source ###")
    mlp_source = read_module_source(model, "model.layers.0.mlp")
    print(f"Class: {mlp_source.class_name}")
    print(f"File: {mlp_source.file_path}")
    print("Source (first 1500 chars):")
    print(mlp_source.source[:1500])

    # Capture intermediates
    print("\n### Intermediate Captures ###")
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda:0")

    captures = [
        "model.embed_tokens",
        "model.layers.0",
        "model.layers.0.self_attn",
        "model.layers.0.mlp",
        "model.norm",
    ]

    for path in captures:
        try:
            cap = capture_intermediate(model, path, input_ids)
            print(f"  {path}: {cap.output_shape}")
        except Exception as e:
            print(f"  {path}: ERROR - {e}")

    # Full forward pass output
    print("\n### Full Forward Pass ###")
    with torch.no_grad():
        output = model(input_ids)
    print(f"  logits shape: {output.logits.shape}")
    print(f"  logits dtype: {output.logits.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Explore Qwen2.5-0.5B architecture")
    parser.add_argument("--local", action="store_true", help="Run locally without GPU")
    parser.add_argument("--gpu-id", type=str, help="Reuse existing GPU instance")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU alive after completion")
    args = parser.parse_args()

    if args.local:
        explore_local()
    else:
        # Check if we're already on GPU
        import torch
        if torch.cuda.is_available():
            explore_gpu()
        else:
            # Run on remote GPU
            from tools.functional_extractor.verify import run_on_gpu
            run_on_gpu(
                script_path=__file__,
                gpu_id=args.gpu_id,
                keep_alive=args.keep_alive or bool(args.gpu_id),
                vram_gb=16,
            )


if __name__ == "__main__":
    main()
