#!/usr/bin/env python3
"""
Simple JAX GPU test script.
This script runs ON the remote GPU instance.
"""


def main():
    print("=" * 60)
    print("JAX GPU Integration Test")
    print("=" * 60)

    # Import JAX
    print("\n1. Importing JAX...")
    try:
        import jax
        import jax.numpy as jnp
        print("   ✅ JAX imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import JAX: {e}")
        return False

    # Check GPU availability
    print("\n2. Checking GPU devices...")
    devices = jax.devices()
    print(f"   Found {len(devices)} device(s):")
    for i, device in enumerate(devices):
        print(f"   [{i}] {device}")

    # Use JAX's type-based device query (cleaner than checking device_kind)
    try:
        gpu_devices = jax.devices('gpu')
    except RuntimeError:
        gpu_devices = []

    if not gpu_devices:
        print("   ❌ No GPU devices found!")
        return False
    print(f"   ✅ Found {len(gpu_devices)} GPU device(s)")

    # Run simple GPU computation
    print("\n3. Running GPU computation...")
    try:
        # Create matrices on GPU
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1000, 1000))
        y = jax.random.normal(key, (1000, 1000))

        # Matrix multiplication on GPU
        result = jnp.matmul(x, y)

        # Force computation and transfer back
        result_value = result.block_until_ready()

        print("   ✅ Matrix multiplication successful")
        print(f"   Result shape: {result_value.shape}")
        print(f"   Result mean: {jnp.mean(result_value):.4f}")

    except Exception as e:
        print(f"   ❌ GPU computation failed: {e}")
        return False

    # Verify computation ran on GPU
    print("\n4. Verifying GPU was used...")
    default_device = jax.devices()[0]
    print(f"   Default device: {default_device}")

    # Check if default device is a GPU
    try:
        gpu_devices = jax.devices('gpu')
        if default_device not in gpu_devices:
            print("   ❌ Computation did not run on GPU!")
            return False
    except RuntimeError:
        print("   ❌ No GPU devices available!")
        return False

    print("   ✅ Computation ran on GPU")

    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
