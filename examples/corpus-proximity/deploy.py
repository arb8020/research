#!/usr/bin/env python3
"""Deploy corpus-proximity code to GPU instance."""

import argparse
import logging
import os
from typing import Literal, TypeAlias
from dotenv import load_dotenv
from broker import GPUClient, CloudType
from bifrost import BifrostClient
from shared.config import get_runpod_key, get_prime_key
from shared.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)

# Type aliases for provision result
ProvisionError: TypeAlias = Literal["create_failed", "ready_timeout", "ssh_timeout"]
ProvisionResult: TypeAlias = (
    tuple[Literal[True], "ClientGPUInstance", None] |
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


def provision_instance(gpu_client, offers, instance_name) -> ProvisionResult:
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
    if not use_existing:
        bootstrap_cmd = [
            """if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi""",
            "uv sync --extra example-corpus-proximity"
        ]
        bifrost_client.push(bootstrap_cmd=bootstrap_cmd)
    else:
        bifrost_client.push()


def find_instance_by_name_or_id(gpu_client, identifier):
    """Find instance by name or ID."""
    instances = gpu_client.list_instances()
    for instance in instances:
        if instance.name == identifier or instance.id == identifier:
            return instance
    return None


def run_deploy(provider=None, use_existing=None, name=None):
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")
    instance_name = name or "corpus-proximity-dev"

    if use_existing:
        ssh_connection = None

        if "@" in use_existing and ":" in use_existing:
            ssh_connection = use_existing
        else:
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
        return bifrost_client
    else:
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
        logger.info(f"  Iterate: python {__file__} --use-existing {instance_name}")

        return bifrost_client


def main():
    parser = argparse.ArgumentParser(description="Deploy corpus-proximity to GPU")
    parser.add_argument("--provider", type=str, choices=["runpod", "primeintellect"])
    parser.add_argument("--use-existing", type=str, metavar="NAME_OR_SSH")
    parser.add_argument("--name", type=str)
    parser.add_argument("--config", type=str, required=True, help="Path to config file (e.g., configs/02_small_test_03.py)")
    args = parser.parse_args()

    setup_logging()

    try:
        bifrost_client = run_deploy(provider=args.provider, use_existing=args.use_existing, name=args.name)
        if not bifrost_client:
            logger.error("Deployment failed")
            return 1

        # Run prepare_data.py
        logger.info(f"Running prepare_data.py with config {args.config}...")
        cmd = f"cd ~/.bifrost/workspace && uv run python examples/corpus-proximity/prepare_data.py examples/corpus-proximity/{args.config}"
        result = bifrost_client.exec(cmd)

        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")

        logger.info(f"prepare_data.py completed with exit code: {result.exit_code}")
        if result.exit_code != 0:
            logger.error("prepare_data.py failed")
            return 1

        # Run embed_chunks.py
        logger.info(f"Running embed_chunks.py with config {args.config}...")
        cmd = f"cd ~/.bifrost/workspace && uv run python examples/corpus-proximity/embed_chunks.py examples/corpus-proximity/{args.config}"
        result = bifrost_client.exec(cmd)

        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")

        logger.info(f"embed_chunks.py completed with exit code: {result.exit_code}")
        if result.exit_code != 0:
            logger.error("embed_chunks.py failed")
            return 1

        # Run test_search.py
        logger.info(f"Running test_search.py with config {args.config}...")
        cmd = f"cd ~/.bifrost/workspace && uv run python examples/corpus-proximity/test_search.py examples/corpus-proximity/{args.config}"
        result = bifrost_client.exec(cmd)

        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")

        logger.info(f"test_search.py completed with exit code: {result.exit_code}")
        if result.exit_code != 0:
            logger.error("test_search.py failed")
            return 1

        # Run cluster_corpus.py
        logger.info(f"Running cluster_corpus.py with config {args.config}...")
        cmd = f"cd ~/.bifrost/workspace && uv run python examples/corpus-proximity/cluster_corpus.py examples/corpus-proximity/{args.config}"
        result = bifrost_client.exec(cmd)

        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")

        logger.info(f"cluster_corpus.py completed with exit code: {result.exit_code}")
        if result.exit_code != 0:
            logger.error("cluster_corpus.py failed")
            return 1

        # Run name_clusters.py --name
        logger.info(f"Running name_clusters.py --name with config {args.config}...")
        cmd = f"cd ~/.bifrost/workspace && uv run python examples/corpus-proximity/name_clusters.py examples/corpus-proximity/{args.config} --name"
        result = bifrost_client.exec(cmd)

        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")

        logger.info(f"name_clusters.py completed with exit code: {result.exit_code}")
        if result.exit_code != 0:
            logger.error("name_clusters.py failed")
            return 1

        logger.info("All steps completed successfully!")
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
