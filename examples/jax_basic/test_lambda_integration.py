#!/usr/bin/env python3
"""
Lambda Labs Integration Test

End-to-end integration test validating Lambda Labs provider.

What it does:
1. Uses broker to search Lambda Labs GPU offers
2. Provisions a Lambda Labs instance
3. Waits for SSH to be ready
4. Runs nvidia-smi to verify GPU
5. Terminates the instance

Usage:
    uv run python examples/jax_basic/test_lambda_integration.py [--gpu-type GPU_TYPE] [--skip-create]

Environment variables:
    LAMBDA_API_KEY: Lambda Labs API key (required)
    SSH_KEY_PATH: Path to SSH key (optional, defaults to ~/.ssh/id_ed25519)
"""

import argparse
import logging
import os
from dotenv import load_dotenv
from broker import GPUClient
from shared.config import get_lambda_key
from shared.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)


def get_credentials():
    """Get Lambda Labs credentials from environment"""
    lambda_key = get_lambda_key()
    if not lambda_key:
        raise ValueError("LAMBDA_API_KEY environment variable not set")
    return {"lambdalabs": lambda_key}


def search_lambda_offers(gpu_client, gpu_type=None):
    """Search for Lambda Labs GPU offers"""
    logger.info("Searching for Lambda Labs GPU offers...")

    offers = gpu_client.search(
        query=f"gpu_type.contains('{gpu_type}')" if gpu_type else None,
        provider="lambdalabs",
        sort=lambda x: x.price_per_hour,
        reverse=False
    )

    if not offers:
        raise ValueError(f"No Lambda Labs offers found{f' for {gpu_type}' if gpu_type else ''}")

    logger.info(f"Found {len(offers)} offers")
    logger.info("\nTop 5 offers:")
    for i, offer in enumerate(offers[:5], 1):
        logger.info(f"  {i}. {offer.gpu_type} x{offer.gpu_count} - ${offer.price_per_hour:.2f}/hr - {offer.availability_zone}")

    return offers


def provision_instance(gpu_client, offers):
    """Provision a Lambda Labs instance"""
    logger.info(f"\nProvisioning: {offers[0].gpu_type} x{offers[0].gpu_count} at ${offers[0].price_per_hour:.2f}/hr")

    instance = gpu_client.create(
        query=offers[0],
        name="lambda-integration-test"
    )

    if not instance:
        raise RuntimeError("Failed to create instance")

    logger.info(f"✅ Instance created: {instance.id}")
    logger.info(f"   Status: {instance.status.value}")
    logger.info(f"   GPU: {instance.gpu_type} x{instance.gpu_count}")

    return instance


def wait_for_ssh(instance, timeout=900):
    """Wait for SSH to be ready on the instance"""
    logger.info("\nWaiting for SSH to be ready...")

    ssh_ready = instance.wait_until_ssh_ready(timeout=timeout)

    if not ssh_ready:
        raise RuntimeError("SSH did not become ready within timeout")

    logger.info(f"✅ SSH ready!")
    logger.info(f"   IP: {instance.public_ip}")
    logger.info(f"   Port: {instance.ssh_port}")
    logger.info(f"   Username: {instance.ssh_username}")


def test_gpu(instance):
    """Run nvidia-smi to verify GPU is accessible"""
    logger.info("\nRunning nvidia-smi...")

    result = instance.exec("nvidia-smi", timeout=60)

    if not result.success:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr}")

    logger.info("nvidia-smi output:")
    logger.info("-" * 80)
    logger.info(result.stdout)
    logger.info("-" * 80)
    logger.info("✅ nvidia-smi executed successfully!")

    # Verify GPU is detected
    if instance.gpu_type in result.stdout:
        logger.info(f"✅ Verified GPU type '{instance.gpu_type}' in output")
    else:
        logger.warning(f"GPU type '{instance.gpu_type}' not found in nvidia-smi output")


def run_lambda_integration_test(gpu_type=None, skip_create=False):
    """Run full Lambda Labs integration test"""
    credentials = get_credentials()
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    logger.info("=" * 80)
    logger.info("Lambda Labs Integration Test")
    logger.info("=" * 80)

    gpu_client = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)

    # Step 1: Search for offers
    offers = search_lambda_offers(gpu_client, gpu_type=gpu_type)

    if skip_create:
        logger.info("\nSkipping instance creation (--skip-create flag)")
        logger.info("✅ Search test passed!")
        return True

    instance = None
    try:
        # Step 2: Provision instance
        instance = provision_instance(gpu_client, offers)

        # Step 3: Wait for SSH
        wait_for_ssh(instance)

        # Step 4: Test GPU
        test_gpu(instance)

        # Step 5: Cleanup
        logger.info("\nCleaning up...")
        success = instance.terminate()

        if not success:
            logger.warning(f"Failed to terminate instance {instance.id}")
            logger.warning("Please manually terminate this instance")
            return False

        logger.info(f"✅ Instance {instance.id} terminated successfully")

        # Success!
        logger.info("\n" + "=" * 80)
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("=" * 80)
        logger.info("\nLambda Labs integration test completed successfully:")
        logger.info("  ✅ Search GPU offers")
        logger.info("  ✅ Create instance")
        logger.info("  ✅ Wait for SSH ready")
        logger.info("  ✅ Run nvidia-smi")
        logger.info("  ✅ Terminate instance")

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        if instance:
            logger.info("Cleaning up instance...")
            try:
                instance.terminate()
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error}")
                logger.warning(f"Please manually terminate instance: {instance.id}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Lambda Labs integration test")
    parser.add_argument("--gpu-type", help="Filter by GPU type (e.g., 'H100', 'A100')")
    parser.add_argument("--skip-create", action="store_true", help="Only test search, don't create instance")

    args = parser.parse_args()

    setup_logging()

    try:
        success = run_lambda_integration_test(gpu_type=args.gpu_type, skip_create=args.skip_create)
        if not success:
            logger.error("Integration test failed")
            return 1
        return 0
    except KeyboardInterrupt:
        logger.error("\nInterrupted")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
