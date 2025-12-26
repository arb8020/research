#!/usr/bin/env python3
"""Reusable debugging toolkit for functional model extraction.

This toolkit provides model-agnostic functions for debugging functional
implementations against HuggingFace models. Key capabilities:

1. capture_hf_internals() - Hook HF model to capture position embeddings, masks
2. compare_layer_by_layer() - Run both implementations layer by layer and find divergence
3. diff_analysis() - Analyze per-position diffs to pinpoint issues
4. component_test() - Test individual components (embed, norm, attention, mlp)

Usage:
    from debug_toolkit import DebugSession

    session = DebugSession(hf_model, functional_forward, weights)
    session.compare_full_forward(input_ids, attention_mask)
    session.find_divergence_layer()
    session.analyze_attention(layer_idx=0)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CapturedInternals:
    """Captured internal values from HF model forward pass."""

    position_embeddings: tuple[Tensor, Tensor] | None = None  # (cos, sin)
    attention_mask: Tensor | None = None
    hidden_states_per_layer: list[Tensor] = field(default_factory=list)
    layer_inputs: dict[int, Tensor] = field(default_factory=dict)
    layer_outputs: dict[int, Tensor] = field(default_factory=dict)


def capture_hf_internals(
    model: torch.nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    capture_layers: list[int] | None = None,
) -> CapturedInternals:
    """Capture internal values from HF model during forward pass.

    Args:
        model: HuggingFace model (e.g., AutoModelForCausalLM)
        input_ids: Input token IDs
        attention_mask: Optional attention mask
        capture_layers: Which layer indices to capture (default: all)

    Returns:
        CapturedInternals with position embeddings, masks, and per-layer hidden states
    """
    result = CapturedInternals()
    hooks = []

    # Detect model type
    if hasattr(model, "model"):
        base_model = model.model
    else:
        base_model = model

    # Get layers list
    if hasattr(base_model, "layers"):
        layers = base_model.layers
    elif hasattr(base_model, "h"):  # GPT-2 style
        layers = base_model.h
    else:
        layers = []

    num_layers = len(layers)
    if capture_layers is None:
        capture_layers = list(range(num_layers))

    # Hook to capture position embeddings and mask from first layer
    def capture_pos_and_mask(module, args, kwargs) -> None:
        if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
            pos_emb = kwargs["position_embeddings"]
            result.position_embeddings = (pos_emb[0].clone(), pos_emb[1].clone())
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            mask = kwargs["attention_mask"]
            if isinstance(mask, Tensor):
                result.attention_mask = mask.clone()
        return None

    if len(layers) > 0:
        hooks.append(layers[0].register_forward_pre_hook(capture_pos_and_mask, with_kwargs=True))

    # Hooks to capture layer inputs/outputs
    def make_layer_hook(layer_idx: int):
        def hook(module, input, output) -> None:
            if isinstance(input, tuple) and len(input) > 0:
                result.layer_inputs[layer_idx] = (
                    input[0].clone() if isinstance(input[0], Tensor) else None
                )
            if isinstance(output, tuple) and len(output) > 0:
                result.layer_outputs[layer_idx] = (
                    output[0].clone() if isinstance(output[0], Tensor) else None
                )
            elif isinstance(output, Tensor):
                result.layer_outputs[layer_idx] = output.clone()

        return hook

    for idx in capture_layers:
        if idx < len(layers):
            hooks.append(layers[idx].register_forward_hook(make_layer_hook(idx)))

    # Run forward pass
    with torch.no_grad():
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        _ = model(input_ids, **kwargs)

    # Clean up hooks
    for h in hooks:
        h.remove()

    return result


def compare_tensors(
    t1: Tensor,
    t2: Tensor,
    name: str = "tensor",
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict:
    """Compare two tensors and return detailed comparison results.

    Returns:
        Dict with: matches, max_diff, mean_diff, shape_match, dtype_match
    """
    result = {
        "name": name,
        "shape_match": t1.shape == t2.shape,
        "dtype_match": t1.dtype == t2.dtype,
        "t1_shape": tuple(t1.shape),
        "t2_shape": tuple(t2.shape),
    }

    if not result["shape_match"]:
        result["matches"] = False
        result["max_diff"] = float("inf")
        return result

    diff = (t1 - t2).abs()
    result["max_diff"] = diff.max().item()
    result["mean_diff"] = diff.mean().item()
    result["matches"] = torch.allclose(t1, t2, rtol=rtol, atol=atol)

    # Find location of max diff
    max_idx = diff.argmax()
    result["max_diff_idx"] = max_idx.item()

    return result


def per_position_diff(
    t1: Tensor,
    t2: Tensor,
    attention_mask: Tensor | None = None,
) -> list[dict]:
    """Compute per-position diff for sequence tensors.

    Args:
        t1, t2: Tensors of shape (batch, seq_len, ...)
        attention_mask: Optional mask of shape (batch, seq_len)

    Returns:
        List of per-position diff info
    """
    assert t1.shape == t2.shape, f"Shape mismatch: {t1.shape} vs {t2.shape}"
    batch, seq_len = t1.shape[:2]

    results = []
    for pos in range(seq_len):
        pos_diff = (t1[:, pos] - t2[:, pos]).abs()
        is_padding = False
        if attention_mask is not None:
            # Check if any batch item has this as padding
            is_padding = (attention_mask[:, pos] == 0).any().item()

        results.append({
            "position": pos,
            "max_diff": pos_diff.max().item(),
            "mean_diff": pos_diff.mean().item(),
            "is_padding": is_padding,
        })

    return results


@dataclass
class DebugSession:
    """Interactive debugging session for comparing HF vs functional implementations.

    Example:
        session = DebugSession(hf_model, my_forward, weights)
        session.test_embeddings(input_ids)
        session.test_position_embeddings(input_ids, attention_mask)
        session.compare_full_forward(input_ids, attention_mask)
        session.find_divergence_layer(input_ids, attention_mask)
    """

    hf_model: torch.nn.Module
    functional_forward: Callable
    weights: dict[str, Tensor]
    device: torch.device = None
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.device is None:
            self.device = next(self.hf_model.parameters()).device

    def test_embeddings(self, input_ids: Tensor) -> dict:
        """Test that embedding layer matches."""
        # HF embeddings
        if hasattr(self.hf_model, "model"):
            hf_embed = self.hf_model.model.embed_tokens(input_ids)
        else:
            hf_embed = self.hf_model.embed_tokens(input_ids)

        # Functional embeddings
        embed_key = None
        for key in ["model.embed_tokens.weight", "transformer.wte.weight", "embed_tokens.weight"]:
            if key in self.weights:
                embed_key = key
                break

        if embed_key is None:
            return {"error": "Could not find embedding weights"}

        func_embed = F.embedding(input_ids, self.weights[embed_key])

        return compare_tensors(hf_embed, func_embed, name="embeddings")

    def test_position_embeddings(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        compute_rope_fn: Callable | None = None,
    ) -> dict:
        """Test position embeddings (RoPE) match.

        Args:
            input_ids: Input tokens
            attention_mask: Optional attention mask
            compute_rope_fn: Your function to compute (cos, sin) from position_ids
        """
        # Capture HF's position embeddings
        captured = capture_hf_internals(self.hf_model, input_ids, attention_mask)

        if captured.position_embeddings is None:
            return {"error": "Could not capture position embeddings from HF model"}

        hf_cos, hf_sin = captured.position_embeddings

        result = {
            "hf_cos_shape": tuple(hf_cos.shape),
            "hf_sin_shape": tuple(hf_sin.shape),
        }

        if compute_rope_fn is not None:
            # Compute functional position embeddings
            seq_len = input_ids.shape[1]
            positions = (
                torch.arange(seq_len, device=self.device)
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )

            func_cos, func_sin = compute_rope_fn(positions, dtype=self.dtype)

            result["cos_comparison"] = compare_tensors(hf_cos, func_cos, name="cos")
            result["sin_comparison"] = compare_tensors(hf_sin, func_sin, name="sin")

            # Check if matches
            result["matches"] = (
                result["cos_comparison"]["matches"] and result["sin_comparison"]["matches"]
            )

            # If not matching, show first few values
            if not result["matches"]:
                result["hf_cos_sample"] = hf_cos[0, :5, :4].tolist()
                result["func_cos_sample"] = func_cos[0, :5, :4].tolist()

        return result

    def test_attention_mask(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        create_mask_fn: Callable | None = None,
    ) -> dict:
        """Test attention mask construction matches.

        Args:
            create_mask_fn: Your function to create 4D causal mask from 2D attention_mask
        """
        captured = capture_hf_internals(self.hf_model, input_ids, attention_mask)

        if captured.attention_mask is None:
            return {"hf_mask": "None (using is_causal=True path)"}

        result = {
            "hf_mask_shape": tuple(captured.attention_mask.shape),
            "hf_mask_dtype": str(captured.attention_mask.dtype),
        }

        if create_mask_fn is not None:
            seq_len = input_ids.shape[1]
            func_mask = create_mask_fn(seq_len, self.device, self.dtype, attention_mask)

            if func_mask is None:
                result["func_mask"] = "None"
                result["matches"] = captured.attention_mask is None
            else:
                result["func_mask_shape"] = tuple(func_mask.shape)
                result["mask_comparison"] = compare_tensors(
                    captured.attention_mask, func_mask, name="attention_mask"
                )
                result["matches"] = result["mask_comparison"]["matches"]

                # Show sample values
                if not result["matches"]:
                    pos = attention_mask[0].sum().item()  # First non-padding position
                    result["hf_mask_sample"] = captured.attention_mask[0, 0, int(pos), :].tolist()
                    result["func_mask_sample"] = func_mask[0, 0, int(pos), :].tolist()

        return result

    def compare_full_forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> dict:
        """Compare full forward pass outputs."""
        with torch.no_grad():
            kwargs = {}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask

            hf_out = self.hf_model(input_ids, **kwargs).logits
            func_out = self.functional_forward(input_ids, self.weights, **kwargs)

        result = compare_tensors(hf_out, func_out, name="logits")

        # Per-position analysis
        if attention_mask is not None:
            result["per_position"] = per_position_diff(hf_out, func_out, attention_mask)

        return result

    def find_divergence_layer(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        layer_forward_fn: Callable | None = None,
        threshold: float = 1e-5,
    ) -> dict:
        """Find the first layer where HF and functional implementations diverge.

        Args:
            layer_forward_fn: Function(hidden, weights, layer_idx, cos, sin, mask) -> hidden
            threshold: Diff threshold to consider as divergence
        """
        captured = capture_hf_internals(self.hf_model, input_ids, attention_mask)

        result = {
            "num_layers": len(captured.layer_outputs),
            "divergence_layer": None,
            "per_layer_diff": [],
        }

        for layer_idx, hf_output in sorted(captured.layer_outputs.items()):
            diff = {
                "layer": layer_idx,
                "hf_output_shape": tuple(hf_output.shape),
            }

            if layer_forward_fn is not None and layer_idx in captured.layer_inputs:
                # Run functional layer
                hf_input = captured.layer_inputs[layer_idx]
                cos, sin = (
                    captured.position_embeddings if captured.position_embeddings else (None, None)
                )

                with torch.no_grad():
                    func_output = layer_forward_fn(
                        hf_input, self.weights, layer_idx, cos, sin, captured.attention_mask
                    )

                comparison = compare_tensors(hf_output, func_output, name=f"layer_{layer_idx}")
                diff["comparison"] = comparison
                diff["max_diff"] = comparison["max_diff"]
                diff["matches"] = comparison["matches"]

                if not comparison["matches"] and result["divergence_layer"] is None:
                    result["divergence_layer"] = layer_idx

            result["per_layer_diff"].append(diff)

        return result


def print_comparison_report(comparison: dict, verbose: bool = True) -> None:
    """Pretty-print a comparison result."""
    name = comparison.get("name", "comparison")
    matches = comparison.get("matches", False)
    max_diff = comparison.get("max_diff", float("inf"))

    status = "PASS" if matches else "FAIL"
    symbol = "✓" if matches else "✗"

    print(f"  {name}: max_diff={max_diff:.2e} [{status}] {symbol}")

    if verbose and not matches:
        if "shape_match" in comparison and not comparison["shape_match"]:
            print(f"    Shape mismatch: {comparison['t1_shape']} vs {comparison['t2_shape']}")
        if "max_diff_idx" in comparison:
            print(f"    Max diff at index: {comparison['max_diff_idx']}")


def print_layer_report(layer_result: dict) -> None:
    """Pretty-print layer-by-layer divergence results."""
    print(f"\nLayer-by-layer analysis ({layer_result['num_layers']} layers):")

    if layer_result["divergence_layer"] is not None:
        print(f"  First divergence at layer: {layer_result['divergence_layer']}")
    else:
        print("  No divergence found - all layers match!")

    print("\n  Per-layer max diff:")
    for layer_diff in layer_result["per_layer_diff"]:
        layer_idx = layer_diff["layer"]
        if "max_diff" in layer_diff:
            max_diff = layer_diff["max_diff"]
            matches = layer_diff.get("matches", max_diff < 1e-5)
            symbol = "✓" if matches else "✗"
            print(f"    Layer {layer_idx:2d}: {max_diff:.2e} {symbol}")


# Convenience function for quick debugging
def quick_debug(
    hf_model: torch.nn.Module,
    functional_forward: Callable,
    weights: dict[str, Tensor],
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    compute_rope_fn: Callable | None = None,
    create_mask_fn: Callable | None = None,
):
    """Quick debug session - run all tests and print report."""
    session = DebugSession(hf_model, functional_forward, weights)

    print("=" * 60)
    print("Quick Debug Report")
    print("=" * 60)

    # Test embeddings
    print("\n### Embeddings ###")
    embed_result = session.test_embeddings(input_ids)
    print_comparison_report(embed_result)

    # Test position embeddings
    if compute_rope_fn:
        print("\n### Position Embeddings (RoPE) ###")
        rope_result = session.test_position_embeddings(input_ids, attention_mask, compute_rope_fn)
        if "cos_comparison" in rope_result:
            print_comparison_report(rope_result["cos_comparison"])
            print_comparison_report(rope_result["sin_comparison"])
        else:
            print(f"  {rope_result}")

    # Test attention mask
    if attention_mask is not None and create_mask_fn:
        print("\n### Attention Mask ###")
        mask_result = session.test_attention_mask(input_ids, attention_mask, create_mask_fn)
        if "mask_comparison" in mask_result:
            print_comparison_report(mask_result["mask_comparison"])
        else:
            print(f"  HF mask: {mask_result.get('hf_mask', 'unknown')}")

    # Test full forward
    print("\n### Full Forward ###")
    forward_result = session.compare_full_forward(input_ids, attention_mask)
    print_comparison_report(forward_result)

    # Per-position analysis
    if "per_position" in forward_result and attention_mask is not None:
        print("\n### Per-Position Diff ###")
        for pos_info in forward_result["per_position"]:
            pos = pos_info["position"]
            max_diff = pos_info["max_diff"]
            is_pad = pos_info["is_padding"]
            marker = " (padding)" if is_pad else ""
            symbol = "✓" if max_diff < 1e-4 else "✗"
            print(f"    pos {pos}: {max_diff:.4f} {symbol}{marker}")

    print("\n" + "=" * 60)

    return {
        "embeddings": embed_result,
        "full_forward": forward_result,
    }
