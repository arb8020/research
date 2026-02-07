#!/usr/bin/env python3
"""Verify functional implementation matches HuggingFace model.

Uses dtype-appropriate tolerances and logprobs-based comparison following
patterns from vLLM/SGLang CI testing.

Usage:
    python verify.py functional.py
    python verify.py functional.py --verbose
    python verify.py functional.py --mode logprobs  # Compare top-k logprobs
    python verify.py functional.py --mode strict    # Exact numerical match

Exit codes:
    0: PASS - outputs match within tolerance
    1: FAIL - outputs do not match
    2: ERROR - could not run verification
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

# Dtype-specific tolerances (from vLLM/SGLang/Transformers patterns)
TOLERANCES = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-4},
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}

# Default test prompts - variety of lengths and token types
DEFAULT_TEST_INPUTS = [
    [1, 2, 3, 4],  # Short
    list(range(1, 33)),  # 32 tokens
    [100, 200, 300, 400, 500, 600, 700, 800],  # Different vocab range
    [1] * 16,  # Repeated tokens
]


def load_module_from_file(file_path: str, module_name: str = "functional"):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_tolerances(dtype: torch.dtype, rtol_override: float | None = None, atol_override: float | None = None):
    """Get appropriate tolerances for dtype."""
    defaults = TOLERANCES.get(dtype, TOLERANCES[torch.float32])
    return {
        "rtol": rtol_override if rtol_override is not None else defaults["rtol"],
        "atol": atol_override if atol_override is not None else defaults["atol"],
    }


def check_logprobs_close(
    hf_logits: torch.Tensor,
    func_logits: torch.Tensor,
    top_k: int = 10,
    verbose: bool = False,
) -> tuple[bool, dict]:
    """Check if logprobs agree using vLLM-style comparison.

    For each position, verify that:
    1. Top-k tokens overlap significantly
    2. If top token differs, each implementation's top token is in the other's top-k

    Returns (passed, details_dict)
    """
    # Get log probabilities
    hf_logprobs = torch.log_softmax(hf_logits.float(), dim=-1)
    func_logprobs = torch.log_softmax(func_logits.float(), dim=-1)

    batch_size, seq_len, vocab_size = hf_logits.shape

    results = {
        "positions_checked": 0,
        "positions_matched": 0,
        "top1_matches": 0,
        "topk_overlap_avg": 0.0,
        "first_divergence": None,
        "max_logprob_diff": 0.0,
    }

    total_overlap = 0.0

    for b in range(batch_size):
        for pos in range(seq_len):
            results["positions_checked"] += 1

            # Get top-k for each
            hf_topk_vals, hf_topk_ids = hf_logprobs[b, pos].topk(top_k)
            func_topk_vals, func_topk_ids = func_logprobs[b, pos].topk(top_k)

            hf_top1 = hf_topk_ids[0].item()
            func_top1 = func_topk_ids[0].item()

            # Check top-1 match
            if hf_top1 == func_top1:
                results["top1_matches"] += 1
                results["positions_matched"] += 1

            # Check overlap
            hf_set = set(hf_topk_ids.tolist())
            func_set = set(func_topk_ids.tolist())
            overlap = len(hf_set & func_set) / top_k
            total_overlap += overlap

            # If top-1 differs, check if each is in other's top-k (vLLM pattern)
            if hf_top1 != func_top1:
                hf_in_func = hf_top1 in func_set
                func_in_hf = func_top1 in hf_set

                if hf_in_func and func_in_hf:
                    results["positions_matched"] += 1
                elif results["first_divergence"] is None:
                    results["first_divergence"] = {
                        "position": pos,
                        "hf_top1": hf_top1,
                        "func_top1": func_top1,
                        "hf_in_func_topk": hf_in_func,
                        "func_in_hf_topk": func_in_hf,
                    }

            # Track max logprob diff for top tokens
            for i in range(min(5, top_k)):
                tid = hf_topk_ids[i].item()
                diff = abs(hf_logprobs[b, pos, tid].item() - func_logprobs[b, pos, tid].item())
                results["max_logprob_diff"] = max(results["max_logprob_diff"], diff)

    results["topk_overlap_avg"] = total_overlap / results["positions_checked"]

    # Pass if all positions matched (top-1 same or cross-contained in top-k)
    passed = results["positions_matched"] == results["positions_checked"]

    if verbose:
        print(f"  Logprobs check:")
        print(f"    Top-1 matches: {results['top1_matches']}/{results['positions_checked']}")
        print(f"    All positions valid: {results['positions_matched']}/{results['positions_checked']}")
        print(f"    Top-{top_k} overlap: {results['topk_overlap_avg']:.1%}")
        print(f"    Max logprob diff: {results['max_logprob_diff']:.2e}")
        if results["first_divergence"]:
            d = results["first_divergence"]
            print(f"    First divergence at pos {d['position']}: HF={d['hf_top1']}, Func={d['func_top1']}")

    return passed, results


def check_numerical_close(
    hf_logits: torch.Tensor,
    func_logits: torch.Tensor,
    rtol: float,
    atol: float,
    verbose: bool = False,
) -> tuple[bool, dict]:
    """Strict numerical comparison with tolerances."""
    diff = (hf_logits.float() - func_logits.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    passed = torch.allclose(hf_logits.float(), func_logits.float(), rtol=rtol, atol=atol)

    results = {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rtol": rtol,
        "atol": atol,
    }

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  Numerical check: {status}")
        print(f"    max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        print(f"    tolerances: rtol={rtol:.0e}, atol={atol:.0e}")

    return passed, results


def main():
    parser = argparse.ArgumentParser(description="Verify functional model implementation")
    parser.add_argument("functional_file", help="Path to functional.py")
    parser.add_argument("--model-info", default="model_info.json", help="Path to model_info.json")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed output")
    parser.add_argument("--rtol", type=float, default=None, help="Override relative tolerance")
    parser.add_argument("--atol", type=float, default=None, help="Override absolute tolerance")
    parser.add_argument(
        "--mode",
        choices=["logprobs", "numerical", "both"],
        default="both",
        help="Comparison mode (default: both)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for logprobs comparison")
    parser.add_argument(
        "--test-inputs",
        type=str,
        default=None,
        help="JSON list of test input sequences",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Model dtype (default: bfloat16)",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Load model info
    model_info_path = Path(args.model_info)
    if not model_info_path.exists():
        print(f"ERROR: {args.model_info} not found", file=sys.stderr)
        sys.exit(2)

    with open(model_info_path) as f:
        model_info = json.load(f)

    model_name = model_info["model_name"]

    # Get test inputs
    if args.test_inputs:
        test_inputs = json.loads(args.test_inputs)
    elif "test_inputs" in model_info:
        test_inputs = model_info["test_inputs"]
    else:
        test_inputs = DEFAULT_TEST_INPUTS

    # Check functional file exists
    if not Path(args.functional_file).exists():
        print(f"ERROR: {args.functional_file} not found", file=sys.stderr)
        sys.exit(2)

    # Import transformers
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}", file=sys.stderr)
        sys.exit(2)

    # Load functional module
    try:
        functional_module = load_module_from_file(args.functional_file)
    except Exception as e:
        print(f"ERROR: Could not load {args.functional_file}: {e}", file=sys.stderr)
        sys.exit(2)

    # Find the forward function
    forward_fn = None
    for name in ["forward", "functional_forward", f"{model_name.split('/')[-1].lower()}_forward"]:
        if hasattr(functional_module, name):
            forward_fn = getattr(functional_module, name)
            break

    if forward_fn is None:
        for attr_name in dir(functional_module):
            if attr_name.endswith("_forward") and callable(getattr(functional_module, attr_name)):
                forward_fn = getattr(functional_module, attr_name)
                break

    if forward_fn is None:
        print(f"ERROR: No forward function found in {args.functional_file}", file=sys.stderr)
        print("Expected one of: forward, functional_forward, <model>_forward", file=sys.stderr)
        sys.exit(2)

    # Get tolerances
    tols = get_tolerances(dtype, args.rtol, args.atol)

    if args.verbose:
        print(f"Model: {model_name}")
        print(f"Forward function: {forward_fn.__name__}")
        print(f"Dtype: {dtype}")
        print(f"Mode: {args.mode}")
        print(f"Test inputs: {len(test_inputs)} sequences")
        print(f"Tolerances: rtol={tols['rtol']:.0e}, atol={tols['atol']:.0e}")
        print()

    # Load HuggingFace model
    if args.verbose:
        print("Loading HuggingFace model...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cuda:0",
        )
        model.eval()
    except Exception as e:
        print(f"ERROR: Could not load model {model_name}: {e}", file=sys.stderr)
        sys.exit(2)

    # Get weights
    weights = dict(model.state_dict())

    # Run verification
    all_pass = True
    max_diff_overall = 0.0
    total_positions = 0
    matched_positions = 0

    for i, input_seq in enumerate(test_inputs):
        input_ids = torch.tensor([input_seq], device="cuda:0")

        if args.verbose:
            print(f"\nTest {i + 1}/{len(test_inputs)}: {len(input_seq)} tokens")

        try:
            with torch.no_grad():
                hf_output = model(input_ids).logits
                functional_output = forward_fn(input_ids, weights)
        except Exception as e:
            print(f"ERROR: Forward pass failed: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(2)

        # Check shapes match
        if hf_output.shape != functional_output.shape:
            print(f"FAIL: Shape mismatch - HF: {hf_output.shape}, Functional: {functional_output.shape}")
            all_pass = False
            continue

        # Run comparison(s)
        test_pass = True

        if args.mode in ("numerical", "both"):
            num_pass, num_results = check_numerical_close(
                hf_output, functional_output, tols["rtol"], tols["atol"], args.verbose
            )
            max_diff_overall = max(max_diff_overall, num_results["max_diff"])
            if not num_pass:
                test_pass = False

        if args.mode in ("logprobs", "both"):
            lp_pass, lp_results = check_logprobs_close(hf_output, functional_output, args.top_k, args.verbose)
            total_positions += lp_results["positions_checked"]
            matched_positions += lp_results["positions_matched"]
            if not lp_pass:
                test_pass = False

        if not test_pass:
            all_pass = False

    # Final result
    print()
    if all_pass:
        summary = f"PASS (max_diff={max_diff_overall:.2e}"
        if args.mode in ("logprobs", "both"):
            summary += f", positions={matched_positions}/{total_positions}"
        summary += ")"
        print(summary)
        sys.exit(0)
    else:
        summary = f"FAIL (max_diff={max_diff_overall:.2e}"
        if args.mode in ("logprobs", "both"):
            summary += f", positions={matched_positions}/{total_positions}"
        summary += ")"
        print(summary)
        sys.exit(1)


if __name__ == "__main__":
    main()
