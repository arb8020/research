#!/usr/bin/env python3
"""
Integration test: Modal → JAX on GPU

This test validates the Modal workflow:
1. Build image with JAX dependencies
2. Create GPU sandbox
3. Run JAX test on GPU
4. Cleanup

Compare with run_integration_test.py (broker/bifrost approach).
"""

import argparse
import logging
import os
from pathlib import Path

import modal

logger = logging.getLogger(__name__)


def build_jax_image():
    """Build Modal image with JAX and CUDA support.

    Tiger Style: Assert inputs, explicit returns.
    """
    logger.info("Building JAX image...")

    # Start with NVIDIA PyTorch base (has CUDA 12.1)
    image = (
        modal.Image.from_registry(
            "nvcr.io/nvidia/pytorch:24.01-py3",
            add_python="3.11"
        )
        .pip_install(
            "jax[cuda12]==0.4.23",  # Specific version for reproducibility
        )
        .add_local_file(
            local_path=str(Path(__file__).parent / "gpu_test_script.py"),
            remote_path="/root/gpu_test_script.py"
        )
    )

    logger.info("Image built successfully")
    return image


def run_modal_jax_test(gpu_type: str = "T4"):
    """Run JAX GPU test on Modal.

    Tiger Style: Assert preconditions, explicit error handling.

    Modal SDK automatically reads credentials from ~/.modal.toml
    (uses the workspace marked with active=true).

    Args:
        gpu_type: GPU type to use (T4, L4, A10G, A100-40GB, A100-80GB, H100)
    """
    # Assert preconditions
    valid_gpus = ["T4", "L4", "A10G", "A100-40GB", "A100-80GB", "H100"]
    assert gpu_type in valid_gpus, f"Invalid GPU type. Choose from: {valid_gpus}"

    # Verify Modal config exists
    modal_config = Path.home() / ".modal.toml"
    assert modal_config.exists(), (
        "Modal config not found. Run: modal token new"
    )

    logger.info("=" * 70)
    logger.info("JAX GPU Integration Test - Modal")
    logger.info("=" * 70)
    logger.info(f"GPU type: {gpu_type}")

    # Build image
    image = build_jax_image()

    # Create Modal app
    logger.info("Creating Modal app...")
    app = modal.App.lookup("jax-integration-test", create_if_missing=True)

    # Create sandbox
    logger.info(f"Creating sandbox with {gpu_type}...")
    sandbox = modal.Sandbox.create(
        image=image,
        app=app,
        gpu=gpu_type,
        timeout=600,  # 10 minutes max
    )

    # Assert sandbox created
    assert sandbox is not None, "Sandbox creation failed"
    assert sandbox.object_id, "Sandbox missing ID"
    logger.info(f"Sandbox created: {sandbox.object_id}")

    try:
        # Run test script
        logger.info("Running GPU test...")
        process = sandbox.exec("python", "/root/gpu_test_script.py")
        process.wait()

        stdout = process.stdout.read()
        stderr = process.stderr.read()
        exit_code = process.returncode

        # Log output
        logger.info("=" * 70)
        logger.info("TEST OUTPUT:")
        logger.info("=" * 70)
        logger.info(stdout)
        if stderr:
            logger.warning("STDERR:")
            logger.warning(stderr)
        logger.info("=" * 70)

        # Validate
        assert exit_code == 0, f"Test failed with exit code {exit_code}"
        assert "All tests passed!" in stdout, "Test did not report success"

        logger.info("=" * 70)
        logger.info("🎉 INTEGRATION TEST PASSED")
        logger.info("=" * 70)

        # Estimate cost (approximate)
        gpu_prices = {
            "T4": 0.36,
            "L4": 0.70,
            "A10G": 1.12,
            "A100-40GB": 3.15,
            "A100-80GB": 4.20,
            "H100": 8.40
        }
        price_per_hour = gpu_prices.get(gpu_type, 1.0)
        estimated_cost = price_per_hour * (2 / 60)  # ~2 minutes
        logger.info(f"Estimated cost: ~${estimated_cost:.4f}")

        return True

    finally:
        # Cleanup
        logger.info("Cleaning up...")
        sandbox.terminate()
        logger.info("Sandbox terminated")


def main():
    """Entry point - handle logging setup and error reporting."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="JAX GPU Integration Test - Modal"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="T4",
        choices=["T4", "L4", "A10G", "A100-40GB", "A100-80GB", "H100"],
        help="GPU type to use (default: T4, cheapest)"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    try:
        run_modal_jax_test(gpu_type=args.gpu)
        return 0

    except KeyboardInterrupt:
        logger.error("\n⚠️ Test interrupted by user")
        return 1
    except AssertionError as e:
        logger.error(f"\n❌ Assertion failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
