#!/usr/bin/env python3

import argparse
import os
import logging
from typing import Literal, TypeAlias
from dotenv import load_dotenv
from broker import GPUClient, CloudType
from broker.client import ClientGPUInstance
from bifrost import BifrostClient
from shared.config import get_runpod_key, get_prime_key
from shared.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)

# Type aliases for provision result
ProvisionError: TypeAlias = Literal["create_failed", "ready_timeout", "ssh_timeout"]
ProvisionResult: TypeAlias = (
    tuple[Literal[True], ClientGPUInstance, None] |
    tuple[Literal[False], None, ProvisionError]
)


def get_credentials(provider_filter=None):
    credentials = {}
    if runpod_key := get_runpod_key():
        credentials["runpod"] = runpod_key
    if prime_key := get_prime_key():
        credentials["primeintellect"] = prime_key

    if provider_filter:
        if provider_filter not in credentials:
            raise ValueError(f"Provider '{provider_filter}' not found. Set {provider_filter.upper()}_API_KEY")
        credentials = {provider_filter: credentials[provider_filter]}

    assert credentials, "No API keys found - set RUNPOD_API_KEY or PRIME_API_KEY"
    return credentials


def search_cheapest_gpus(gpu_client, max_offers=5):
    offers = gpu_client.search(
        query=gpu_client.cloud_type == CloudType.COMMUNITY,
        sort=lambda x: x.price_per_hour,
        reverse=False
    )
    assert offers, "No GPU offers found"
    return offers[:max_offers]


def provision_instance(gpu_client, offers) -> ProvisionResult:
    instance = gpu_client.create(
        offers,
        image="runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204",
        name="jax-integration-test",
        n_offers=len(offers)
    )
    if not instance:
        logger.error("Failed to create instance")
        return (False, None, "create_failed")

    ready = instance.wait_until_ready(timeout=300)
    if not ready:
        logger.error(f"Instance {instance.id} failed to become ready, terminating...")
        gpu_client.terminate_instance(instance.id, instance.provider)
        return (False, None, "ready_timeout")

    ssh_ready = instance.wait_until_ssh_ready(timeout=300)
    if not ssh_ready:
        logger.error(f"SSH failed to become ready on {instance.id}, terminating...")
        gpu_client.terminate_instance(instance.id, instance.provider)
        return (False, None, "ssh_timeout")

    return (True, instance, None)


def deploy_and_test(bifrost_client):
    bootstrap_cmd = [
        """if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi""",
        "uv sync --extra example-jax-basic"
    ]

    bifrost_client.push(bootstrap_cmd=bootstrap_cmd)

    logger.info("Running GPU test...")
    result = bifrost_client.exec("uv run python examples/jax_basic/gpu_test_script.py")

    logger.info("=" * 70)
    logger.info("TEST OUTPUT:")
    logger.info("=" * 70)
    logger.info(result.stdout)
    if result.stderr:
        logger.warning("STDERR:")
        logger.warning(result.stderr)
    logger.info("=" * 70)

    assert result.exit_code == 0, f"Test failed with exit code {result.exit_code}"
    assert "All tests passed!" in result.stdout, "Test did not report success"

    return result


def run_integration_test(provider=None):
    credentials = get_credentials(provider_filter=provider)
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    logger.info("=" * 70)
    logger.info("JAX GPU Integration Test")
    logger.info("=" * 70)

    gpu_client = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)

    offers = search_cheapest_gpus(gpu_client, max_offers=5)
    success, instance, error = provision_instance(gpu_client, offers)

    if not success:
        logger.error(f"Provisioning failed: {error}")
        return False

    bifrost_client = BifrostClient(
        ssh_connection=instance.ssh_connection_string(),
        ssh_key_path=ssh_key_path
    )

    deploy_and_test(bifrost_client)

    logger.info("Cleaning up...")
    instance.terminate()
    logger.info("Instance terminated")

    logger.info(f"Cost estimate: ~${instance.price_per_hour * 0.1:.4f}")
    return True


def main():
    parser = argparse.ArgumentParser(description="JAX GPU Integration Test")
    parser.add_argument("--provider", type=str, choices=["runpod", "primeintellect"])
    args = parser.parse_args()

    setup_logging()

    try:
        success = run_integration_test(provider=args.provider)
        if not success:
            logger.error("Integration test failed")
            return 1
        logger.info("=" * 70)
        logger.info("INTEGRATION TEST PASSED")
        logger.info("=" * 70)
        return 0
    except KeyboardInterrupt:
        logger.error("\nInterrupted")
        return 1
    except AssertionError as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
