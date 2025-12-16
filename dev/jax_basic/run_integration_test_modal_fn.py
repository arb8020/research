#!/usr/bin/env python3
"""
Integration test: Modal Functions → JAX on GPU

Refactored to use @app.function() instead of Sandboxes for:
- Faster startup (warm pool)
- Better metrics/logging
- More idiomatic Modal usage

Compare with run_integration_test_modal.py (sandbox approach).
"""

import time

import modal

# Build image once (shared across all functions)
jax_image = modal.Image.from_registry(
    "nvcr.io/nvidia/pytorch:24.01-py3", add_python="3.11"
).pip_install("jax[cuda12]==0.4.23")

app = modal.App("jax-integration-test")


@app.function(
    gpu="T4",  # Can override with modal run --gpu=A100
    image=jax_image,
    timeout=600,
)
def run_jax_test():
    """Run JAX GPU test directly in Modal function."""
    import jax
    import jax.numpy as jnp

    print("=" * 70)
    print("JAX GPU Test")
    print("=" * 70)

    # Check JAX sees GPU
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    assert len(devices) > 0, "No JAX devices found"
    assert devices[0].platform == "gpu", f"Expected GPU, got {devices[0].platform}"

    # Run simple computation
    print("\nRunning matrix multiplication...")
    start = time.time()
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    y.block_until_ready()  # Wait for GPU
    elapsed = time.time() - start

    print(f"Computed 1000x1000 matmul in {elapsed:.4f}s")
    print(f"Result shape: {y.shape}, sum: {y.sum()}")

    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)

    return {
        "success": True,
        "device": str(devices[0]),
        "elapsed": elapsed,
    }


@app.local_entrypoint()
def main(gpu: str = "T4"):
    """Entry point for modal run."""
    print(f"Starting JAX test on {gpu}...")

    start = time.time()
    result = run_jax_test.remote()
    total_time = time.time() - start

    print(f"\nTotal wall time: {total_time:.2f}s")
    print(f"Result: {result}")

    # Compare with RunPod timing
    print("\n💡 For RunPod comparison, check:")
    print("  - Cold start time (first run)")
    print("  - Warm start time (second run)")
    print("  - Total cost per run")


@app.local_entrypoint()
def benchmark(gpu: str = "T4", iterations: int = 5):
    """Run multiple iterations to measure warm start performance.

    Usage:
        modal run run_integration_test_modal_fn.py::benchmark --gpu A100-40GB --iterations 10
    """
    times = []

    for i in range(iterations):
        print(f"\n=== Iteration {i + 1}/{iterations} ===")
        start = time.time()
        run_jax_test.remote()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Time: {elapsed:.2f}s")

    print("\n=== Results ===")
    print(f"Cold start: {times[0]:.2f}s")
    if len(times) > 1:
        warm_avg = sum(times[1:]) / len(times[1:])
        print(f"Warm average: {warm_avg:.2f}s")
        print(f"Speedup: {times[0] / warm_avg:.1f}x")
    print(f"All times: {[f'{t:.2f}s' for t in times]}")
