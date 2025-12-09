#!/usr/bin/env python3
"""Test functional Qwen implementation against HuggingFace model.

Run locally (requires GPU):
    python -m tools.functional_extractor.test_qwen

Run on remote GPU:
    python -m tools.functional_extractor.test_qwen --remote
    python -m tools.functional_extractor.test_qwen --remote --gpu-id <id>
"""

from __future__ import annotations

import argparse
import sys


def test_on_gpu():
    """Run full verification test on GPU."""
    import torch
    from transformers import AutoModelForCausalLM

    # Import functional implementation
    from tools.functional_extractor.qwen_functional import qwen_forward

    print("=" * 60)
    print("Qwen2.5-0.5B Functional Implementation Test")
    print("=" * 60)

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    weights = {k: v for k, v in model.state_dict().items()}

    # Test cases
    test_inputs = [
        [1, 2, 3, 4],  # Short sequence
        [100, 200, 300, 400, 500],  # Different tokens
        [1],  # Single token
        list(range(1, 33)),  # Longer sequence (32 tokens)
    ]

    print(f"\nRunning {len(test_inputs)} test cases...")
    print("-" * 60)

    all_passed = True
    for i, input_seq in enumerate(test_inputs):
        input_ids = torch.tensor([input_seq], device="cuda:0")

        with torch.no_grad():
            original_logits = model(input_ids).logits
            functional_logits = qwen_forward(input_ids, weights)

        matches = torch.allclose(original_logits, functional_logits, rtol=1e-5, atol=1e-5)
        max_diff = (original_logits - functional_logits).abs().max().item()

        status = "PASS" if matches else "FAIL"
        print(f"Test {i+1}: seq_len={len(input_seq):2d}, max_diff={max_diff:.2e}, {status}")

        if not matches:
            all_passed = False
            # Debug: find where the difference is largest
            diff = (original_logits - functional_logits).abs()
            max_idx = diff.argmax()
            print(f"  Max diff at index: {max_idx.item()}")
            print(f"  Original value: {original_logits.view(-1)[max_idx].item():.6f}")
            print(f"  Functional value: {functional_logits.view(-1)[max_idx].item():.6f}")

    print("-" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("Functional implementation is numerically identical to HF model.")
    else:
        print("SOME TESTS FAILED!")
        print("Functional implementation has numerical differences.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test functional Qwen implementation")
    parser.add_argument("--remote", action="store_true", help="Run on remote GPU")
    parser.add_argument("--gpu-id", type=str, help="Reuse existing GPU instance")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU alive after test")
    args = parser.parse_args()

    if args.remote:
        from tools.functional_extractor.verify import run_on_gpu

        run_on_gpu(
            script_path=__file__,
            gpu_id=args.gpu_id,
            keep_alive=args.keep_alive or bool(args.gpu_id),
            vram_gb=16,
        )
    else:
        # Check if we have GPU
        import torch

        if not torch.cuda.is_available():
            print("No GPU available locally. Use --remote to run on remote GPU.")
            sys.exit(1)

        test_on_gpu()


if __name__ == "__main__":
    main()
