#!/usr/bin/env python3
"""Test functional Llama implementation against HuggingFace model.

Run locally (requires GPU):
    python -m tools.functional_extractor.test_llama

Run on remote GPU:
    python -m tools.functional_extractor.test_llama --remote
    python -m tools.functional_extractor.test_llama --remote --gpu-id <id>
"""

from __future__ import annotations

import argparse
import sys


def test_on_gpu():
    """Run full verification test on GPU."""
    import os
    import sys

    # Add parent dirs to path for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import torch
    from transformers import AutoModelForCausalLM

    from llama_functional import llama_forward
    from test_template import run_test_suite

    print("Loading Llama 3.2 1B for testing...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    weights = dict(model.state_dict())

    # Clean up model to save memory before running suite
    del model
    torch.cuda.empty_cache()

    # Run the test suite
    result = run_test_suite(
        model_name="meta-llama/Llama-3.2-1B",
        functional_forward=llama_forward,
        weights=weights,
        device="cuda:0",
        dtype=torch.bfloat16,
    )

    if not result.all_passed:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test functional Llama implementation")
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
            vram_gb=24,  # Llama 3.2 1B needs more VRAM than Qwen 0.5B
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
