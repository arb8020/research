#!/usr/bin/env python3

import argparse
import os
import logging
from typing import Literal, TypeAlias
from dotenv import load_dotenv
from broker import GPUClient, CloudType
from broker.client import ClientGPUInstance
from bifrost import BifrostClient
from shared.config import get_runpod_key, get_prime_key, get_lambda_key
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
    if lambda_key := get_lambda_key():
        credentials["lambdalabs"] = lambda_key

    if provider_filter:
        if provider_filter not in credentials:
            raise ValueError(f"Provider '{provider_filter}' not found. Set {provider_filter.upper()}_API_KEY")
        credentials = {provider_filter: credentials[provider_filter]}

    assert credentials, "No API keys found - set RUNPOD_API_KEY, PRIME_API_KEY, or LAMBDA_API_KEY"
    return credentials


def search_cheapest_gpus(gpu_client, max_offers=5, provider=None):
    # Lambda Labs only supports SECURE cloud type, others support COMMUNITY
    if provider == "lambdalabs":
        cloud_filter = gpu_client.cloud_type == CloudType.SECURE
    else:
        cloud_filter = gpu_client.cloud_type == CloudType.COMMUNITY

    offers = gpu_client.search(
        query=cloud_filter,
        sort=lambda x: x.price_per_hour,
        reverse=False
    )
    assert offers, "No GPU offers found"
    return offers[:max_offers]


def provision_instance(gpu_client, offers, provider=None, ready_timeout=None, ssh_timeout=None) -> ProvisionResult:
    instance = gpu_client.create(
        offers,
        image="runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204",
        name="jax-integration-test",
        n_offers=len(offers)
    )
    if not instance:
        logger.error("Failed to create instance")
        return (False, None, "create_failed")

    # Provider-specific timeouts (explicit is better than implicit)
    # Lambda instances can take 10-15min for both ready and SSH
    # RunPod/PrimeIntellect are typically ready in 3-5min
    if ready_timeout is None:
        ready_timeout = 900 if provider == "lambdalabs" else 300
    if ssh_timeout is None:
        ssh_timeout = 900 if provider == "lambdalabs" else 300

    ready = instance.wait_until_ready(timeout=ready_timeout)
    if not ready:
        logger.error(f"Instance {instance.id} failed to become ready after {ready_timeout} seconds, terminating...")
        gpu_client.terminate_instance(instance.id, instance.provider)
        return (False, None, "ready_timeout")

    ssh_ready = instance.wait_until_ssh_ready(timeout=ssh_timeout)
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
    # Ensure uv is in PATH (it's installed in ~/.cargo/bin during bootstrap)
    test_cmd = """
export PATH="$HOME/.cargo/bin:$PATH"
uv run python examples/jax_basic/gpu_test_script.py
""".strip()
    result = bifrost_client.exec(test_cmd)

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


def run_integration_test(provider=None, ready_timeout=None, ssh_timeout=None):
    credentials = get_credentials(provider_filter=provider)
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    # Lambda Labs uses a provider-specific key (downloaded from their dashboard)
    ssh_key_paths = {
        "lambdalabs": os.getenv("LAMBDA_SSH_KEY_PATH", "~/.ssh/lambda/lambda_key.pem")
    }

    logger.info("=" * 70)
    logger.info("JAX GPU Integration Test")
    if provider:
        logger.info(f"Provider: {provider}")
    logger.info("=" * 70)

    gpu_client = GPUClient(
        credentials=credentials,
        ssh_key_path=ssh_key_path,
        ssh_key_paths=ssh_key_paths
    )

    offers = search_cheapest_gpus(gpu_client, max_offers=5, provider=provider)
    success, instance, error = provision_instance(
        gpu_client, offers, provider=provider, ready_timeout=ready_timeout, ssh_timeout=ssh_timeout
    )

    if not success:
        logger.error(f"Provisioning failed: {error}")
        return False

    # Get provider-specific SSH key (with fallback to default)
    provider_ssh_key = gpu_client.get_ssh_key_path(provider=instance.provider)
    assert provider_ssh_key is not None, "SSH key path must be configured"

    bifrost_client = BifrostClient(
        ssh_connection=instance.ssh_connection_string(),
        ssh_key_path=provider_ssh_key
    )

    deploy_and_test(bifrost_client)

    logger.info("Cleaning up...")
    instance.terminate()
    logger.info("Instance terminated")

    logger.info(f"Cost estimate: ~${instance.price_per_hour * 0.1:.4f}")
    return True


def main():
    parser = argparse.ArgumentParser(description="JAX GPU Integration Test")
    parser.add_argument("--provider", type=str, choices=["runpod", "primeintellect", "lambdalabs"],
                        help="Cloud provider to use")
    parser.add_argument("--ready-timeout", type=int, default=None,
                        help="Timeout in seconds for instance to become ready (default: 900 for Lambda Labs, 300 for others)")
    parser.add_argument("--ssh-timeout", type=int, default=None,
                        help="Timeout in seconds for SSH to become ready (default: 900 for Lambda Labs, 300 for others)")
    args = parser.parse_args()

    setup_logging()

    try:
        success = run_integration_test(
            provider=args.provider,
            ready_timeout=args.ready_timeout,
            ssh_timeout=args.ssh_timeout
        )
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
