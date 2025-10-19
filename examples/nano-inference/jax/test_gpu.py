#!/usr/bin/env python3
"""
GPU test script for JAX GPT-2 implementation.
This script runs ON the remote GPU instance to verify GPU deployment.

Based on examples/jax_basic/gpu_test_script.py
"""

import sys
from pathlib import Path


def check_jax_gpu():
    """Check if JAX can detect and use GPU."""
    print("\n1. Importing JAX...")
    try:
        import jax
        import jax.numpy as jnp
        print("   ✅ JAX imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import JAX: {e}")
        return False

    print("\n2. Checking GPU devices...")
    devices = jax.devices()
    print(f"   Found {len(devices)} device(s):")
    for i, device in enumerate(devices):
        print(f"   [{i}] {device}")

    try:
        gpu_devices = jax.devices('gpu')
    except RuntimeError:
        gpu_devices = []

    if not gpu_devices:
        print("   ❌ No GPU devices found!")
        return False

    print(f"   ✅ Found {len(gpu_devices)} GPU device(s)")
    return True


def test_basic_gpu_computation():
    """Test basic GPU computation."""
    print("\n3. Running basic GPU computation...")
    try:
        import jax
        import jax.numpy as jnp

        # Matrix multiplication on GPU
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1000, 1000))
        y = jax.random.normal(key, (1000, 1000))
        result = jnp.matmul(x, y).block_until_ready()

        print(f"   ✅ Matrix multiplication successful")
        print(f"   Result shape: {result.shape}")
        print(f"   Result mean: {jnp.mean(result):.4f}")
        return True

    except Exception as e:
        print(f"   ❌ GPU computation failed: {e}")
        return False


def test_gpt2_gpu():
    """Test GPT-2 model on GPU."""
    print("\n4. Testing GPT-2 model on GPU...")

    try:
        import jax.numpy as jnp
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from config import GPT2Config
        from jax.model import gpt2_forward
        from jax.loader import load_weights

        print("   Loading GPT-2 weights...")
        weights = load_weights("gpt2")
        config = GPT2Config()

        print("   Running forward pass on GPU...")
        input_ids_BT = jnp.array([[15496, 995]])  # "Hello world"
        logits_BTV = gpt2_forward(input_ids_BT, weights, config)

        # Force computation
        logits_BTV = logits_BTV.block_until_ready()

        print(f"   ✅ GPT-2 forward pass successful")
        print(f"   Logits shape: {logits_BTV.shape}")
        print(f"   Logits range: [{logits_BTV.min():.3f}, {logits_BTV.max():.3f}]")
        return True

    except Exception as e:
        print(f"   ❌ GPT-2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_gpu_used():
    """Verify computation ran on GPU."""
    print("\n5. Verifying GPU was used...")
    try:
        import jax
        default_device = jax.devices()[0]
        print(f"   Default device: {default_device}")

        gpu_devices = jax.devices('gpu')
        if default_device not in gpu_devices:
            print("   ❌ Computation did not run on GPU!")
            return False

        print("   ✅ Computation ran on GPU")
        return True

    except Exception as e:
        print(f"   ❌ GPU verification failed: {e}")
        return False


def main():
    print("=" * 60)
    print("JAX GPT-2 GPU Test")
    print("=" * 60)

    tests = [
        ("JAX GPU Detection", check_jax_gpu),
        ("Basic GPU Computation", test_basic_gpu_computation),
        ("GPT-2 on GPU", test_gpt2_gpu),
        ("GPU Verification", verify_gpu_used),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"   ❌ Test crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
        if not success:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
        print("=" * 60)
        return True
    else:
        print("❌ Some tests failed")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
