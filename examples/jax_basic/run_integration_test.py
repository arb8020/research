#!/usr/bin/env python3
"""
Integration test: Broker → Bifrost → JAX on GPU

This test validates the complete workflow:
1. Provision GPU with broker
2. Deploy code with bifrost
3. Run JAX test on GPU
4. Cleanup
"""

import os
import logging

from broker import GPUClient, CloudType
from bifrost import BifrostClient


logger = logging.getLogger(__name__)


def get_credentials():
    """Read credentials from environment."""
    credentials = {}
    if runpod_key := os.getenv("RUNPOD_API_KEY"):
        credentials["runpod"] = runpod_key
    if vast_key := os.getenv("VAST_API_KEY"):
        credentials["vast"] = vast_key

    assert credentials, "No API keys found - set RUNPOD_API_KEY or VAST_API_KEY"
    return credentials


def search_cheapest_gpu(gpu_client):
    """Search for cheapest available GPU."""
    logger.info("Searching for cheapest GPU offer...")

    offers = gpu_client.search(
        query=gpu_client.cloud_type == CloudType.COMMUNITY,
        sort=lambda x: x.price_per_hour,
        reverse=False
    )

    assert offers, "No GPU offers found!"

    # Log top 3 cheapest
    logger.info(f"Found {len(offers)} offers. Top 3 cheapest:")
    for i, offer in enumerate(offers[:3]):
        logger.info(f"  [{i+1}] {offer.gpu_type} - ${offer.price_per_hour:.3f}/hr ({offer.provider})")

    return offers[0]


def provision_instance(gpu_client, offer):
    """Provision GPU instance and wait for SSH."""
    logger.info(f"Provisioning {offer.gpu_type} @ ${offer.price_per_hour:.3f}/hr")

    instance = gpu_client.create(
        offer,
        image="nvidia/cuda:12.1.0-base-ubuntu22.04",
        name="jax-integration-test"
    )

    assert instance, "Failed to create instance!"
    logger.info(f"Instance created: {instance.id}")

    # Wait for ready
    logger.info("Waiting for instance to be ready...")
    ready = instance.wait_until_ready(timeout=300)
    assert ready, "Instance failed to become ready!"

    # Wait for SSH
    logger.info("Waiting for SSH to be ready...")
    ssh_ready = instance.wait_until_ssh_ready(timeout=300)
    assert ssh_ready, "SSH failed to become ready!"
    logger.info(f"SSH ready: {instance.ssh_address}")

    return instance


def deploy_and_test(bifrost_client):
    """Deploy JAX and run test script."""
    # Bootstrap: install uv if needed, then sync dependencies
    logger.info("Installing uv and example dependencies...")
    bootstrap_cmd = [
        # Install uv if not present
        """if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi""",
        # Sync dependencies with JAX extra
        "uv sync --extra example-jax-basic"
    ]

    bifrost_client.push(bootstrap_cmd=bootstrap_cmd)

    # Run test (already deployed by push, just execute it)
    logger.info("Running GPU test...")
    result = bifrost_client.exec("uv run python examples/jax_basic/gpu_test_script.py")

    # Log output
    logger.info("=" * 70)
    logger.info("TEST OUTPUT:")
    logger.info("=" * 70)
    logger.info(result.stdout)
    if result.stderr:
        logger.warning("STDERR:")
        logger.warning(result.stderr)
    logger.info("=" * 70)

    # Validate
    assert result.exit_code == 0, f"Test failed with exit code {result.exit_code}"
    assert "All tests passed!" in result.stdout, "Test did not report success"

    return result


def run_integration_test():
    """Run the full integration test."""
    # Setup
    credentials = get_credentials()
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    logger.info("=" * 70)
    logger.info("JAX GPU Integration Test - Broker → Bifrost → GPU")
    logger.info("=" * 70)
    logger.info(f"Providers: {list(credentials.keys())}")
    logger.info(f"SSH key: {ssh_key_path}")

    # Initialize clients
    gpu_client = GPUClient(
        credentials=credentials,
        ssh_key_path=ssh_key_path
    )

    # Search and provision
    offer = search_cheapest_gpu(gpu_client)
    instance = provision_instance(gpu_client, offer)

    # Deploy and test
    bifrost_client = BifrostClient(
        ssh_connection=instance.ssh_address,
        ssh_key_path=ssh_key_path
    )

    deploy_and_test(bifrost_client)

    # Cleanup
    logger.info("Cleaning up...")
    instance.terminate()
    logger.info("Instance terminated")

    logger.info(f"Cost estimate: ~${offer.price_per_hour * 0.1:.4f}")
    return True


def main():
    """Entry point - handle logging setup and error reporting."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    try:
        success = run_integration_test()
        logger.info("=" * 70)
        logger.info("🎉 INTEGRATION TEST PASSED")
        logger.info("=" * 70)
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
