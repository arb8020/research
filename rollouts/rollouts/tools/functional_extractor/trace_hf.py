#!/usr/bin/env python3
"""Trace HuggingFace model execution to understand actual ops.

Uses torch.fx to trace the model and see the actual computation graph.
"""

from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def trace_model():
    import torch
    import torch.fx as fx
    from transformers import AutoModelForCausalLM

    print("=" * 60)
    print("HuggingFace Model Tracing")
    print("=" * 60)

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    # Test inputs
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, :2] = 0  # Add some padding

    print(f"\nInput IDs: {input_ids.shape}")
    print(f"Attention mask: {attention_mask.tolist()}")

    # Method 1: torch.fx symbolic trace
    print("\n### torch.fx Symbolic Trace ###")
    try:
        traced = fx.symbolic_trace(model)
        print("Symbolic trace succeeded!")
        print(f"Graph nodes: {len(list(traced.graph.nodes))}")
        for node in list(traced.graph.nodes)[:20]:
            print(f"  {node.op}: {node.target}")
    except Exception as e:
        print(f"Symbolic trace failed: {e}")

    # Method 2: torch.compile trace (dynamo)
    print("\n### torch.compile Trace (Dynamo) ###")
    import torch._dynamo as dynamo

    def print_graph(gm, example_inputs):
        print(f"Captured graph with {len(list(gm.graph.nodes))} nodes")
        for i, node in enumerate(gm.graph.nodes):
            if i < 30:
                print(f"  {node.op}: {node.target}")
        return gm

    try:
        dynamo.reset()
        compiled = torch.compile(model, backend=print_graph)
        with torch.no_grad():
            out = compiled(input_ids, attention_mask=attention_mask)
        print(f"Output shape: {out.logits.shape}")
    except Exception as e:
        print(f"Compile trace failed: {e}")

    # Method 3: torch.export (newer API)
    print("\n### torch.export ###")
    try:
        from torch.export import export

        exported = export(model, (input_ids,), kwargs={"attention_mask": attention_mask})
        print(f"Exported graph: {len(list(exported.graph.nodes))} nodes")

        # Look for SDPA or attention-related ops
        sdpa_nodes = []
        call_nodes = []
        for node in exported.graph.nodes:
            target_str = str(node.target)
            # Only look at call_function nodes, not placeholders
            if node.op == "call_function":
                call_nodes.append(node)
                if (
                    "sdpa" in target_str.lower()
                    or "scaled_dot" in target_str.lower()
                    or "scaled_dot_product" in target_str
                ):
                    sdpa_nodes.append(node)

        print(f"Total call_function nodes: {len(call_nodes)}")
        print(f"SDPA nodes found: {len(sdpa_nodes)}")
        for node in sdpa_nodes[:3]:
            print(f"  {node.op}: {node.target}")
            print(f"    kwargs: {node.kwargs}")

        # Show unique call_function targets
        from collections import Counter

        targets = Counter(str(n.target) for n in call_nodes)
        print("\nMost common ops:")
        for target, count in targets.most_common(15):
            print(f"  {target}: {count}")
    except Exception as e:
        print(f"Export failed: {e}")

    # Method 4: Hook-based tracing (capture actual tensor ops)
    print("\n### Hook-Based Op Tracing ###")
    ops_called = []

    def trace_hook(module, input, output):
        ops_called.append((type(module).__name__, module))

    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(trace_hook))

    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    print(f"Modules called: {len(ops_called)}")
    # Group by type
    from collections import Counter

    type_counts = Counter(op[0] for op in ops_called)
    for op_type, count in type_counts.most_common(20):
        print(f"  {op_type}: {count}")

    # Method 5: Show the actual forward path with attention mask
    print("\n### Forward Path Analysis ###")

    # Check which attention implementation is being used
    print(f"Model type: {type(model).__name__}")
    print(f"Attention type: {type(model.model.layers[0].self_attn).__name__}")

    # Check config
    config = model.config
    print(f"use_sdpa: {getattr(config, '_attn_implementation', 'unknown')}")

    # Look at what _update_causal_mask does
    print("\n### _update_causal_mask behavior ###")
    # Run the actual mask update to see what it returns
    cache_position = torch.arange(4, device="cuda:0")
    past_key_values = None

    # Get the mask that the model actually uses
    if hasattr(model.model, "_update_causal_mask"):
        causal_mask = model.model._update_causal_mask(
            attention_mask=attention_mask,
            input_tensor=model.model.embed_tokens(input_ids),
            cache_position=cache_position,
            past_key_values=past_key_values,
            output_attentions=False,
        )
        if causal_mask is None:
            print("_update_causal_mask returned None (using is_causal=True path)")
        else:
            print(
                f"_update_causal_mask returned mask: shape={causal_mask.shape}, dtype={causal_mask.dtype}"
            )
            for i in range(4):
                print(f"  HF mask at pos {i}: {causal_mask[0, 0, i, :].tolist()}")

    # Compare with my mask
    print("\n### My mask construction ###")
    from qwen_functional import create_causal_mask

    my_mask = create_causal_mask(4, input_ids.device, torch.bfloat16, attention_mask)
    if my_mask is None:
        print("My mask is None (using is_causal=True)")
    else:
        print(f"My mask shape: {my_mask.shape}, dtype: {my_mask.dtype}")
        for i in range(4):
            print(f"  My mask at pos {i}: {my_mask[0, 0, i, :].tolist()}")

    # Check the difference
    if causal_mask is not None and my_mask is not None:
        print("\n### Mask Comparison ###")
        match = torch.allclose(causal_mask, my_mask)
        max_diff = (causal_mask - my_mask).abs().max().item()
        print(f"Masks match: {match}, max_diff: {max_diff}")


if __name__ == "__main__":
    import torch

    if torch.cuda.is_available():
        trace_model()
    else:
        print("No GPU available. Run on remote GPU.")
        import argparse

        from verify import run_on_gpu

        parser = argparse.ArgumentParser()
        parser.add_argument("--gpu-id", type=str)
        args = parser.parse_args()
        run_on_gpu(__file__, gpu_id=args.gpu_id, keep_alive=True, vram_gb=16)
