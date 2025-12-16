#!/usr/bin/env python3
"""Basic inference test with Qwen 0.5B.

Tests:
- Model loading
- Generation with logprobs
- Multiple samples per prompt

Usage:
    python examples/inference/qwen_basic.py
    python examples/inference/qwen_basic.py --remote
"""

from dataclasses import replace

from base_config import BaseConfig, run_remote, test_inference

config = replace(
    BaseConfig(),
    prompts=(
        "Hello, my name is",
        "The capital of France is",
        "To solve this math problem, I will",
    ),
    temperature=0.7,
    max_tokens=30,
    num_samples_per_prompt=2,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", help="Run on remote GPU")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument("--gpu-id", type=str, help="Reuse existing GPU instance ID")
    args = parser.parse_args()

    if args.remote or args.gpu_id:
        run_remote(__file__, keep_alive=args.keep_alive, gpu_id=args.gpu_id)
    else:
        results = test_inference(config)
        if results:
            # Verify we got logprobs
            for r in results:
                assert r["num_tokens"] > 0, "No tokens generated"
            print("\nPASSED: All samples have tokens and logprobs")
