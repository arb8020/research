#!/usr/bin/env python3
"""Dump the HuggingFace MoE source code for GLM-4.5.

Uses read_module_source() to get the actual forward() implementation.
"""
from __future__ import annotations

import os
import sys

# Add parent dir to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def main():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from tools import read_module_source

    print("Loading GLM-4.5 config...")
    config = AutoConfig.from_pretrained("zai-org/GLM-4.5", trust_remote_code=True)

    # Shrink model to just 4 layers to load faster
    config.num_hidden_layers = 4

    print(f"Loading model with {config.num_hidden_layers} layers...")
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
    )

    # Print MoE config
    print("\n### MoE Config ###")
    print(f"n_routed_experts: {config.n_routed_experts}")
    print(f"n_shared_experts: {config.n_shared_experts}")
    print(f"num_experts_per_tok: {config.num_experts_per_tok}")
    print(f"n_group: {config.n_group}")
    print(f"topk_group: {config.topk_group}")
    print(f"norm_topk_prob: {config.norm_topk_prob}")
    print(f"routed_scaling_factor: {config.routed_scaling_factor}")

    # Dump the MoE module source (layer 3 is first MoE layer)
    print("\n### MoE Module Source (model.layers.3.mlp) ###")
    moe_source = read_module_source(model, "model.layers.3.mlp")
    print(f"Class: {moe_source.class_name}")
    print(f"File: {moe_source.file_path}")
    print("-" * 60)
    print(moe_source.source)

    # Dump the Gate/Router source
    print("\n### Gate Module Source (model.layers.3.mlp.gate) ###")
    gate_source = read_module_source(model, "model.layers.3.mlp.gate")
    print(f"Class: {gate_source.class_name}")
    print(f"File: {gate_source.file_path}")
    print("-" * 60)
    print(gate_source.source)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str)
    parser.add_argument("--keep-alive", action="store_true")
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()

    if args.remote:
        from tools.functional_extractor.verify import run_on_gpu
        run_on_gpu(__file__, gpu_id=args.gpu_id, keep_alive=args.keep_alive)
    else:
        main()
