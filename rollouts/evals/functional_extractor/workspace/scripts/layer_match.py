#!/usr/bin/env python3
"""Compare layer-by-layer activations between HF model and functional implementation.

This is the key debugging tool - it shows exactly where the implementation diverges.

Usage:
    python layer_match.py HuggingFaceTB/SmolLM2-135M functional.py
    python layer_match.py HuggingFaceTB/SmolLM2-135M functional.py --verbose
    python layer_match.py HuggingFaceTB/SmolLM2-135M functional.py --test-inputs "[[1,2,3,4]]"

Output:
    Per-layer comparison showing max_diff and whether torch.allclose passes.
    If mismatch found, shows the first divergent layer with debug info.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_module_from_file(file_path: str, module_name: str = "functional"):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_activations(model, input_ids):
    """Run forward pass and collect all intermediate activations via hooks."""
    import torch

    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            # Handle tuple outputs (common in transformer layers)
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            activations[name] = out.detach().clone()
        return hook

    handles = []

    # Register hooks on key layers
    # Embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        handles.append(model.model.embed_tokens.register_forward_hook(make_hook('embed_tokens')))

    # Each transformer layer
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(make_hook(f'layer_{i}')))
            # Also hook attention and MLP sublayers if available
            if hasattr(layer, 'self_attn'):
                handles.append(layer.self_attn.register_forward_hook(make_hook(f'layer_{i}.self_attn')))
            if hasattr(layer, 'mlp'):
                handles.append(layer.mlp.register_forward_hook(make_hook(f'layer_{i}.mlp')))

    # Final norm
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        handles.append(model.model.norm.register_forward_hook(make_hook('final_norm')))

    # LM head
    if hasattr(model, 'lm_head'):
        handles.append(model.lm_head.register_forward_hook(make_hook('lm_head')))

    # Run forward pass
    with torch.no_grad():
        output = model(input_ids)

    # Remove hooks
    for h in handles:
        h.remove()

    activations['logits'] = output.logits.detach().clone()

    return activations


def compare_activations(hf_activations: dict, func_activations: dict, rtol: float = 1e-5, atol: float = 1e-5):
    """Compare two sets of activations layer by layer."""
    import torch

    results = []

    # Get all layer names from HF (the reference)
    for name in hf_activations:
        hf_act = hf_activations[name]

        if name not in func_activations:
            results.append({
                'layer': name,
                'status': 'MISSING',
                'error': f'Layer {name} not found in functional implementation',
            })
            continue

        func_act = func_activations[name]

        # Check shapes
        if hf_act.shape != func_act.shape:
            results.append({
                'layer': name,
                'status': 'SHAPE_MISMATCH',
                'hf_shape': list(hf_act.shape),
                'func_shape': list(func_act.shape),
            })
            continue

        # Compute difference
        diff = (hf_act.float() - func_act.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Check if close
        matches = torch.allclose(hf_act.float(), func_act.float(), rtol=rtol, atol=atol)

        results.append({
            'layer': name,
            'status': 'PASS' if matches else 'FAIL',
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'shape': list(hf_act.shape),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare layer activations")
    parser.add_argument("model_name", help="HuggingFace model name")
    parser.add_argument("functional_file", help="Path to functional.py")
    parser.add_argument("--test-inputs", type=str, default="[[1,2,3,4]]",
                        help="JSON list of test inputs")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", help="Output JSON file")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    test_inputs = json.loads(args.test_inputs)

    # Load functional module
    if not Path(args.functional_file).exists():
        print(f"ERROR: {args.functional_file} not found", file=sys.stderr)
        sys.exit(2)

    try:
        func_module = load_module_from_file(args.functional_file)
    except Exception as e:
        print(f"ERROR: Could not load functional module: {e}", file=sys.stderr)
        sys.exit(2)

    # Find forward function
    forward_fn = None
    for name in ["forward", "functional_forward"]:
        if hasattr(func_module, name):
            forward_fn = getattr(func_module, name)
            break
    if forward_fn is None:
        for attr_name in dir(func_module):
            if attr_name.endswith("_forward") and callable(getattr(func_module, attr_name)):
                forward_fn = getattr(func_module, attr_name)
                break

    if forward_fn is None:
        print("ERROR: No forward function found", file=sys.stderr)
        sys.exit(2)

    # Check if functional module has layer hooks
    get_activations_fn = getattr(func_module, 'get_activations', None)

    print(f"Loading model {args.model_name}...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
    ).cpu()
    model.eval()

    weights = dict(model.state_dict())

    all_results = []

    for i, input_seq in enumerate(test_inputs):
        input_ids = torch.tensor([input_seq])

        print(f"\nTest {i+1}: input_ids={input_seq}", file=sys.stderr)

        # Get HF activations
        hf_activations = collect_activations(model, input_ids)

        # Get functional activations
        if get_activations_fn:
            # Functional module provides activation collection
            func_activations = get_activations_fn(input_ids, weights)
        else:
            # Just compare final output
            with torch.no_grad():
                func_output = forward_fn(input_ids, weights)
            func_activations = {'logits': func_output}

        # Compare
        results = compare_activations(hf_activations, func_activations, args.rtol, args.atol)

        # Print results
        first_fail = None
        for r in results:
            status = r['status']
            layer = r['layer']

            if status == 'PASS':
                if args.verbose:
                    print(f"  {layer}: PASS (max_diff={r['max_diff']:.2e})")
            elif status == 'FAIL':
                print(f"  {layer}: FAIL (max_diff={r['max_diff']:.2e})")
                if first_fail is None:
                    first_fail = r
            elif status == 'MISSING':
                if args.verbose:
                    print(f"  {layer}: MISSING")
            elif status == 'SHAPE_MISMATCH':
                print(f"  {layer}: SHAPE_MISMATCH hf={r['hf_shape']} func={r['func_shape']}")
                if first_fail is None:
                    first_fail = r

        all_results.append({
            'input': input_seq,
            'layers': results,
            'first_failure': first_fail,
        })

        # Summary
        passes = sum(1 for r in results if r['status'] == 'PASS')
        fails = sum(1 for r in results if r['status'] in ('FAIL', 'SHAPE_MISMATCH'))

        if fails == 0:
            print(f"  All {passes} layers match!")
        else:
            print(f"  {fails} layers failed, {passes} passed")
            if first_fail and first_fail['status'] == 'FAIL':
                print(f"  First failure: {first_fail['layer']} (max_diff={first_fail['max_diff']:.2e})")

    # Output JSON if requested
    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        print(f"\nWrote results to {args.output}", file=sys.stderr)

    # Exit code
    any_fail = any(
        r.get('first_failure') is not None
        for r in all_results
    )
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
