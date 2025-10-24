#!/usr/bin/env python3
"""
Test script for JAX GPT-2 implementation.

Compares JAX implementation against HuggingFace reference to verify correctness.

Usage:
    python jax/test_gpt2.py
    python jax/test_gpt2.py --batches 10
"""

import sys
from pathlib import Path

# Add nano-inference root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import argparse
from typing import Dict

from utils.comparison import compare_logits, get_hf_logits
from utils.weights import download_gpt2_weights, load_gpt2_weights
from config import GPT2Config
from backends.jax.model import gpt2_forward
from backends.jax.loader import load_weights


def load_weights_for_jax() -> Dict[str, jax.Array]:
    """Download and load GPT-2 weights, convert to JAX arrays."""
    print("📦 Loading GPT-2 weights...")
    weights = load_weights("gpt2")

    print(f"✅ Loaded {len(weights)} weight tensors")
    total_params = sum(p.size for p in weights.values())
    print(f"   Total parameters: {total_params:,}")

    return weights


def generate_test_batches(k=5):
    """Generate k different test batches for comparison."""
    test_batches = []

    test_batches.append({
        "name": "Hello world",
        "tokens": np.array([[15496, 995]])
    })

    test_batches.append({
        "name": "The quick brown",
        "tokens": np.array([[464, 2068, 7586]])
    })

    test_batches.append({
        "name": "Single token",
        "tokens": np.array([[15496]])
    })

    test_batches.append({
        "name": "Longer sequence",
        "tokens": np.array([[464, 2068, 7586, 1976, 11687, 625, 262]])
    })

    if k >= 5:
        test_batches.append({
            "name": "Batch size 2",
            "tokens": np.array([[15496, 995], [464, 2068]])
        })

    return test_batches[:k]


def test_single_batch(input_ids_BT: np.ndarray, weights: Dict[str, jax.Array], config: GPT2Config):
    """Test a single batch and return comparison results."""
    # Get HuggingFace reference
    print("📚 Getting HuggingFace logits...")
    hf_logits = get_hf_logits(input_ids_BT, model_name="gpt2")
    print(f"   HF logits shape: {hf_logits.shape}")
    print(f"   HF logits range: [{hf_logits.min():.3f}, {hf_logits.max():.3f}]")

    # Get JAX implementation logits
    print("🔥 Getting JAX logits...")
    jax_input = jnp.array(input_ids_BT)
    jax_logits = gpt2_forward(jax_input, weights, config)
    jax_logits_np = np.array(jax_logits)

    print(f"   JAX logits shape: {jax_logits_np.shape}")
    print(f"   JAX logits range: [{jax_logits_np.min():.3f}, {jax_logits_np.max():.3f}]")

    # Compare logits
    print("⚖️  Comparing logits...")
    comparison = compare_logits(
        jax_logits_np,
        hf_logits,
        rtol=5e-3,  # 0.5% relative tolerance
        atol=1e-1,  # 0.1 absolute tolerance
        verbose=False
    )

    return comparison


def test_multiple_batches(weights: Dict[str, jax.Array], config: GPT2Config, k=5):
    """Test across k different batches."""
    print("🧪 Testing JAX GPT-2 vs HuggingFace across multiple batches")
    print("=" * 70)

    test_batches = generate_test_batches(k)
    results = []

    for i, batch in enumerate(test_batches):
        print(f"\n📊 Batch {i+1}/{k}: {batch['name']}")
        print("-" * 40)

        test_input = batch['tokens']
        print(f"Input shape: {test_input.shape}")
        print(f"Input tokens: {test_input.tolist()}")

        try:
            comparison = test_single_batch(test_input, weights, config)

            batch_result = {
                "name": batch['name'],
                "input_shape": test_input.shape,
                "max_abs_diff": comparison.get('max_abs_diff', float('inf')),
                "mean_abs_diff": comparison.get('mean_abs_diff', float('inf')),
                "all_close": comparison.get('all_close', False)
            }
            results.append(batch_result)

            print(f"Max absolute difference: {batch_result['max_abs_diff']:.6f}")
            print(f"Mean absolute difference: {batch_result['mean_abs_diff']:.6f}")
            print(f"All close (rtol=5e-3, atol=1e-1): {batch_result['all_close']}")

            if batch_result['all_close']:
                print("✅ PASS")
            else:
                print("❌ FAIL")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": batch['name'],
                "input_shape": test_input.shape,
                "max_abs_diff": float('inf'),
                "mean_abs_diff": float('inf'),
                "all_close": False,
                "error": str(e)
            })

    return results


def print_summary(results):
    """Print summary of all batch comparisons."""
    print("\n" + "=" * 70)
    print("📋 SUMMARY REPORT")
    print("=" * 70)

    total_batches = len(results)
    passed_batches = sum(1 for r in results if r['all_close'])

    print(f"Total batches tested: {total_batches}")
    print(f"Batches passed: {passed_batches}")
    print(f"Batches failed: {total_batches - passed_batches}")
    print(f"Pass rate: {passed_batches/total_batches*100:.1f}%")

    print("\nPer-batch results:")
    for i, result in enumerate(results):
        status = "✅ PASS" if result['all_close'] else "❌ FAIL"
        max_diff = result['max_abs_diff']
        if np.isinf(max_diff):
            diff_str = "ERROR"
        else:
            diff_str = f"{max_diff:8.6f}"
        print(f"  {i+1}. {result['name']:15} - Max diff: {diff_str} - {status}")

    if passed_batches == total_batches:
        print("\n🎉 ALL TESTS PASSED! Your GPT-2 implementation matches HuggingFace!")
    else:
        print(f"\n💡 {total_batches - passed_batches} tests failed.")

        valid_diffs = [r['max_abs_diff'] for r in results if not np.isinf(r['max_abs_diff'])]
        if valid_diffs:
            avg_max_diff = np.mean(valid_diffs)
            if avg_max_diff > 10:
                print("💡 Large differences suggest missing core components")
            elif avg_max_diff > 1:
                print("💡 Medium differences suggest architectural mismatches")
            else:
                print("💡 Small differences suggest numerical precision issues")


def main():
    parser = argparse.ArgumentParser(description="Test JAX GPT-2 implementation against HuggingFace")
    parser.add_argument("--batches", type=int, default=5,
                       help="Number of test batches to run (default: 5)")

    args = parser.parse_args()

    print("🚀 JAX GPT-2 Test Suite")
    print(f"Testing across {args.batches} input batches...")
    print()

    try:
        # Load weights once
        weights = load_weights_for_jax()
        config = GPT2Config()

        # Test across multiple batches
        results = test_multiple_batches(weights, config, k=args.batches)
        print_summary(results)

    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
