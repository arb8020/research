#!/usr/bin/env python3
"""Debug script for Qwen3-Next functional implementation.

Uses DebugSession to find layer-by-layer divergence between HF and functional.

Usage:
    # Run on GPU
    python tools/functional_extractor/debug_qwen3_next.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add rollouts to path for imports
_script_dir = Path(__file__).resolve().parent
_rollouts_dir = _script_dir.parent.parent
if str(_rollouts_dir) not in sys.path:
    sys.path.insert(0, str(_rollouts_dir))


def setup_environment():
    """Install required packages with correct versions."""
    import subprocess
    print("Setting up environment...")

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


def debug():
    import torch
    from transformers import AutoModelForCausalLM

    # Import debug toolkit and functional implementation
    from tools.functional_extractor.debug_toolkit import (
        DebugSession,
        capture_hf_internals,
        compare_tensors,
        print_comparison_report,
    )
    from tools.functional_extractor.qwen3_next_functional import (
        qwen3_next_forward,
        transformer_layer,
        rms_norm,
        compute_rope_embeddings,
        self_attention,
        gated_delta_net,
        HEAD_DIM,
        PARTIAL_ROTARY_FACTOR,
        NUM_LAYERS,
        SELF_ATTN_LAYERS,
    )

    MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"  # FP8 version fits on single 80GB GPU

    print("=" * 60)
    print(f"Debugging {MODEL_NAME} Functional Implementation")
    print("=" * 60)

    # Load model
    print("\n### Loading model... ###")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Single GPU for FP8
        trust_remote_code=True,
    )
    model.eval()

    weights = dict(model.state_dict())

    # Test inputs
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda:0")

    # Create DebugSession
    session = DebugSession(
        hf_model=model,
        functional_forward=qwen3_next_forward,
        weights=weights,
        dtype=torch.bfloat16,
    )

    # Test 1: Embeddings
    print("\n### Test 1: Embeddings ###")
    embed_result = session.test_embeddings(input_ids)
    print_comparison_report(embed_result)

    # Test 2: Full forward pass
    print("\n### Test 2: Full Forward Pass ###")
    forward_result = session.compare_full_forward(input_ids)
    print_comparison_report(forward_result)

    if not forward_result["matches"]:
        print(f"  Max diff: {forward_result['max_diff']:.6e}")

        # Test 3: Layer-by-layer to find divergence
        print("\n### Test 3: Layer-by-Layer Analysis ###")

        # Capture HF internals for all layers
        captured = capture_hf_internals(model, input_ids, capture_layers=list(range(min(10, NUM_LAYERS))))

        if captured.position_embeddings:
            print(f"  HF position embeddings: cos {captured.position_embeddings[0].shape}, sin {captured.position_embeddings[1].shape}")

        # Run layer by layer manually
        print("\n  Running layer-by-layer comparison...")

        # Get embeddings (should match)
        hidden_hf = model.model.embed_tokens(input_ids)
        hidden_func = torch.nn.functional.embedding(input_ids, weights["model.embed_tokens.weight"])

        embed_diff = compare_tensors(hidden_hf, hidden_func, name="embeddings")
        print(f"  Embeddings: max_diff={embed_diff['max_diff']:.6e}")

        # Compute RoPE for functional
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        rotary_dim = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)
        cos, sin = compute_rope_embeddings(positions, rotary_dim, dtype=hidden_func.dtype)

        print(f"  Functional RoPE: cos {cos.shape}, sin {sin.shape}")

        # Compare layer by layer
        hidden_func = hidden_func.clone()

        for layer_idx in range(min(10, NUM_LAYERS)):
            # Get HF layer output
            hf_layer_out = captured.layer_outputs.get(layer_idx)
            if hf_layer_out is None:
                print(f"  Layer {layer_idx}: No HF capture available")
                continue

            # Get HF layer input (for functional)
            hf_layer_in = captured.layer_inputs.get(layer_idx)
            if hf_layer_in is None and layer_idx == 0:
                # First layer input is embeddings
                hf_layer_in = hidden_hf

            if hf_layer_in is None:
                print(f"  Layer {layer_idx}: No input capture available")
                continue

            # Run functional layer on HF input (to isolate layer issues)
            with torch.no_grad():
                func_layer_out = transformer_layer(
                    hf_layer_in.clone(),  # Use HF input to isolate layer
                    weights,
                    layer_idx,
                    cos, sin, rotary_dim,
                    None,  # attention_mask
                )

            layer_diff = compare_tensors(hf_layer_out, func_layer_out, name=f"layer_{layer_idx}")
            attn_type = "self_attn" if layer_idx in SELF_ATTN_LAYERS else "linear_attn"
            status = "PASS" if layer_diff["matches"] else "FAIL"
            print(f"  Layer {layer_idx:2d} ({attn_type:10s}): max_diff={layer_diff['max_diff']:.6e} [{status}]")

            if not layer_diff["matches"]:
                print(f"\n  ### FIRST DIVERGENCE AT LAYER {layer_idx} ({attn_type}) ###")

                # Debug this specific layer
                prefix = f"model.layers.{layer_idx}"

                # Test individual components
                print("\n  Debugging layer components...")

                # Input norm
                normed_hf = model.model.layers[layer_idx].input_layernorm(hf_layer_in)
                normed_func = rms_norm(hf_layer_in, weights[f"{prefix}.input_layernorm.weight"])
                norm_diff = compare_tensors(normed_hf, normed_func, name="input_norm")
                print(f"    input_layernorm: max_diff={norm_diff['max_diff']:.6e}")

                # Attention (varies by layer type)
                if layer_idx in SELF_ATTN_LAYERS:
                    # Test self-attention
                    with torch.no_grad():
                        hf_attn = model.model.layers[layer_idx].self_attn

                        # Get HF attention output (need to capture via hook)
                        attn_output_hf = None
                        def capture_attn(module, input, output):
                            nonlocal attn_output_hf
                            attn_output_hf = output[0] if isinstance(output, tuple) else output

                        hook = hf_attn.register_forward_hook(capture_attn)
                        _ = model.model.layers[layer_idx](hf_layer_in)
                        hook.remove()

                        # Run functional attention
                        attn_output_func = self_attention(
                            normed_func, weights, f"{prefix}.self_attn",
                            cos, sin, rotary_dim, None
                        )

                        if attn_output_hf is not None:
                            attn_diff = compare_tensors(attn_output_hf, attn_output_func, name="self_attn")
                            print(f"    self_attn: max_diff={attn_diff['max_diff']:.6e}")

                else:
                    # Test GatedDeltaNet
                    with torch.no_grad():
                        hf_attn = model.model.layers[layer_idx].linear_attn

                        # Capture HF output
                        attn_output_hf = None
                        def capture_attn(module, input, output):
                            nonlocal attn_output_hf
                            attn_output_hf = output[0] if isinstance(output, tuple) else output

                        hook = hf_attn.register_forward_hook(capture_attn)
                        _ = model.model.layers[layer_idx](hf_layer_in)
                        hook.remove()

                        # Run functional
                        attn_output_func = gated_delta_net(
                            normed_func, weights, f"{prefix}.linear_attn"
                        )

                        if attn_output_hf is not None:
                            attn_diff = compare_tensors(attn_output_hf, attn_output_func, name="linear_attn")
                            print(f"    linear_attn: max_diff={attn_diff['max_diff']:.6e}")

                break  # Stop at first divergence

    print("\n" + "=" * 60)
    print("Debug complete!")
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
        debug()
