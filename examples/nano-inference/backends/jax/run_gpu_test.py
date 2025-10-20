#!/usr/bin/env python3
"""
Deploy and test JAX GPT-2 implementation on GPU.

Supports both provisioning new instances and reusing existing ones for fast iteration.

Usage:
    # Provision new GPU and run test
    python backends/jax/run_gpu_test.py

    # Reuse existing instance (fast iteration)
    python backends/jax/run_gpu_test.py --use-existing jax-gpt2-dev

    # Or with direct SSH connection
    python backends/jax/run_gpu_test.py --use-existing root@123.45.67.89:22
"""

import argparse
import logging
import os
from typing import Literal, TypeAlias
from dotenv import load_dotenv
from broker import GPUClient, CloudType, GPUInstance
from bifrost import BifrostClient
from shared.config import get_runpod_key, get_prime_key
from shared.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)

ProvisionError: TypeAlias = Literal["create_failed", "ready_timeout", "ssh_timeout"]
ProvisionResult: TypeAlias = (
    tuple[Literal[True], GPUInstance, None] |
    tuple[Literal[False], None, ProvisionError]
)


def get_credentials(provider_filter=None):
    """Get cloud provider credentials from environment."""
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
    """Search for cheapest available GPUs."""
    offers = gpu_client.search(
        query=gpu_client.cloud_type == CloudType.COMMUNITY,
        sort=lambda x: x.price_per_hour,
        reverse=False
    )
    assert offers, "No GPU offers found"
    return offers[:max_offers]


def provision_instance(gpu_client, offers, instance_name) -> ProvisionResult:
    """Provision a new GPU instance."""
    instance = gpu_client.create(
        offers,
        image="runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204",
        name=instance_name,
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

    ssh_ready = instance.wait_until_ssh_ready(timeout=900)
    if not ssh_ready:
        logger.error(f"SSH failed to become ready on {instance.id}, terminating...")
        gpu_client.terminate_instance(instance.id, instance.provider)
        return (False, None, "ssh_timeout")

    return (True, instance, None)


def deploy_code(bifrost_client, use_existing=False):
    """Deploy code to GPU instance."""
    if not use_existing:
        # Full bootstrap on new instance
        bootstrap_cmd = [
            """if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi""",
            "uv sync --extra example-nano-inference-jax"
        ]
        bifrost_client.push(bootstrap_cmd=bootstrap_cmd)
    else:
        # Sync code and update dependencies on existing instance
        bootstrap_cmd = [
            "uv sync --extra example-nano-inference-jax"
        ]
        bifrost_client.push(bootstrap_cmd=bootstrap_cmd)


def find_instance_by_name_or_id(gpu_client, identifier):
    """Find instance by name or ID."""
    instances = gpu_client.list_instances()
    for instance in instances:
        if instance.name == identifier or instance.id == identifier:
            return instance
    return None


def run_gpu_test(bifrost_client):
    """Run GPU correctness test on remote instance."""
    logger.info("Running GPU correctness test (comparing against HuggingFace)...")

    result = bifrost_client.exec(
        "uv run python examples/nano-inference/backends/jax/test_gpt2.py --batches 5",
        working_dir="~/.bifrost/workspace"
    )

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


def run_deploy_and_test(provider=None, use_existing=None, name=None, terminate=False):
    """Deploy code and run GPU test."""
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")
    instance_name = name or "jax-gpt2-dev"

    if use_existing:
        # Use existing instance (fast iteration)
        ssh_connection = None

        if "@" in use_existing and ":" in use_existing:
            # Direct SSH connection string
            ssh_connection = use_existing
        else:
            # Instance name or ID - need to look it up
            credentials = get_credentials(provider_filter=provider)
            gpu_client = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)
            instance = find_instance_by_name_or_id(gpu_client, use_existing)

            if instance:
                ssh_connection = instance.ssh_connection_string()
            else:
                raise ValueError(f"Instance not found (name or ID): {use_existing}")

        bifrost_client = BifrostClient(ssh_connection=ssh_connection, ssh_key_path=ssh_key_path)
        deploy_code(bifrost_client, use_existing=True)

        logger.info(f"Deployed to {use_existing}")
        run_gpu_test(bifrost_client)
        logger.info("✅ Test completed on existing instance")
        return None

    else:
        # Provision new instance
        credentials = get_credentials(provider_filter=provider)
        gpu_client = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)

        offers = search_cheapest_gpus(gpu_client, max_offers=5)
        success, instance, error = provision_instance(gpu_client, offers, instance_name)

        if not success:
            logger.error(f"Provisioning failed: {error}")
            return None

        bifrost_client = BifrostClient(
            ssh_connection=instance.ssh_connection_string(),
            ssh_key_path=ssh_key_path
        )
        deploy_code(bifrost_client, use_existing=False)

        logger.info(f"Deployed: {instance_name}")
        logger.info(f"  SSH: {instance.ssh_connection_string()}")
        logger.info(f"  Iterate: python backends/jax/run_gpu_test.py --use-existing {instance_name}")

        run_gpu_test(bifrost_client)

        if terminate:
            logger.info("Terminating instance...")
            instance.terminate()
            logger.info(f"Instance terminated. Cost: ~${instance.price_per_hour * 0.1:.4f}")
        else:
            logger.info(f"Instance kept running: {instance_name}")
            logger.info(f"  To terminate: broker terminate {instance.id} --provider {instance.provider}")
            logger.info(f"  Hourly cost: ${instance.price_per_hour:.4f}/hr")

        return instance


def main():
    parser = argparse.ArgumentParser(description="Deploy and test JAX GPT-2 on GPU")
    parser.add_argument("--provider", type=str, choices=["runpod", "primeintellect"],
                       help="Cloud provider to use")
    parser.add_argument("--use-existing", type=str, metavar="NAME_OR_SSH",
                       help="Use existing instance (name, ID, or SSH connection string)")
    parser.add_argument("--name", type=str,
                       help="Name for new instance (default: jax-gpt2-dev)")
    parser.add_argument("--terminate", action="store_true",
                       help="Terminate instance after test (only for new instances)")
    args = parser.parse_args()

    setup_logging()

    try:
        logger.info("=" * 70)
        logger.info("JAX GPT-2 GPU Test")
        logger.info("=" * 70)

        instance = run_deploy_and_test(
            provider=args.provider,
            use_existing=args.use_existing,
            name=args.name,
            terminate=args.terminate
        )

        logger.info("=" * 70)
        logger.info("✅ GPU TEST PASSED")
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
