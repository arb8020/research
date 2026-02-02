"""Remote execution utilities for rollouts.

Unified node acquisition pattern from kerbal/tests/test_integration.py.
Supports static SSH, existing broker instances, or provisioning new ones.

Usage:
    from .remote import acquire_node, get_broker_credentials

    # Static SSH
    client, instance = acquire_node(ssh="root@gpu:22")

    # Reuse existing instance
    client, instance = acquire_node(node_id="runpod:abc123")

    # Provision new instance
    client, instance = acquire_node(provision=True, gpu_type="A100")

    try:
        # ... do work with client ...
    finally:
        release_node(instance, keep_alive=False)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bifrost import BifrostClient
    from broker.client import ClientGPUInstance

logger = logging.getLogger(__name__)


def get_broker_credentials() -> dict[str, str]:
    """Load broker credentials. Env vars win, then ~/.broker/credentials.toml.

    Delegates to broker.credentials for the actual loading logic.
    """
    from broker.credentials import get_credentials

    return get_credentials()


def acquire_node(
    ssh: str | None = None,
    node_id: str | None = None,
    provision: bool = False,
    gpu_type: str = "A100",
    gpu_count: int = 1,
    ssh_key_path: str = "~/.ssh/id_ed25519",
    provider: str | None = None,
    container_disk_gb: int = 100,
    ssh_timeout: int = 600,
) -> tuple[BifrostClient, ClientGPUInstance | None]:
    """Acquire a node for remote execution.

    Three modes (mutually exclusive):
        1. ssh: Connect to static SSH endpoint
        2. node_id: Reuse existing broker instance
        3. provision: Provision new instance via broker

    Args:
        ssh: Static SSH connection string (e.g., "root@gpu:22")
        node_id: Existing instance ID (format: "provider:id")
        provision: Whether to provision a new instance
        gpu_type: GPU type to search for when provisioning (e.g., "A100", "4090")
        gpu_count: Number of GPUs to provision
        ssh_key_path: Path to SSH private key
        provider: Specific provider to use (runpod, lambdalabs, vast, primeintellect)
        container_disk_gb: Disk size for provisioned instances
        ssh_timeout: Timeout in seconds waiting for SSH

    Returns:
        (BifrostClient, instance) - instance is None for static SSH

    Raises:
        ValueError: If no acquisition mode specified
        AssertionError: If credentials missing or instance not found

    Example:
        >>> client, instance = acquire_node(provision=True, gpu_type="A100")
        >>> try:
        ...     workspace = client.push("~/.bifrost/workspaces/my-project")
        ...     client.exec(f"cd {workspace} && python train.py")
        ... finally:
        ...     release_node(instance, keep_alive=False)
    """
    from bifrost import BifrostClient
    from broker import GPUClient

    if ssh:
        # Static node - just connect
        logger.info("Connecting to static node: %s", ssh)
        client = BifrostClient(ssh, ssh_key_path=ssh_key_path)
        return client, None

    elif node_id:
        # Existing instance - look it up
        node_provider, instance_id = node_id.split(":", 1)
        logger.info("Connecting to existing instance: %s:%s", node_provider, instance_id)

        credentials = get_broker_credentials()
        assert credentials, "No broker credentials found in environment"

        broker = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)
        instance = broker.get_instance(instance_id, node_provider)
        assert instance, f"Instance not found: {node_id}"

        logger.info("  GPU: %sx %s", instance.gpu_count, instance.gpu_type)
        logger.info("  Waiting for SSH...")
        instance.wait_until_ssh_ready(timeout=ssh_timeout)

        key_path = broker.get_ssh_key_path(node_provider)
        assert key_path, f"No SSH key configured for {node_provider}"
        client = BifrostClient(
            instance.ssh_connection_string(),
            ssh_key_path=key_path,
        )
        return client, instance

    elif provision:
        # Provision new instance
        logger.info("Provisioning new instance (%sx %s)...", gpu_count, gpu_type)

        credentials = get_broker_credentials()
        assert credentials, "No broker credentials found in environment"

        # If provider specified, only use that provider's credentials
        if provider:
            assert provider in credentials, f"No credentials for {provider}"
            credentials = {provider: credentials[provider]}

        broker = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)

        # Build query
        query = broker.gpu_type.contains(gpu_type)

        instance = broker.create(
            query,
            gpu_count=gpu_count,
            cloud_type="secure",
            container_disk_gb=container_disk_gb,
            sort=lambda x: x.price_per_hour,
        )

        logger.info("  Instance ID: %s:%s", instance.provider, instance.id)
        logger.info("  GPU: %sx %s", instance.gpu_count, instance.gpu_type)
        logger.info("  Price: $%.2f/hr", instance.price_per_hour)
        logger.info("  Waiting for SSH...")

        if not instance.wait_until_ssh_ready(timeout=ssh_timeout):
            instance.terminate()
            raise AssertionError(f"SSH not ready after {ssh_timeout}s")

        key_path = broker.get_ssh_key_path(instance.provider)
        assert key_path, f"No SSH key configured for {instance.provider}"
        client = BifrostClient(
            instance.ssh_connection_string(),
            ssh_key_path=key_path,
        )
        return client, instance

    else:
        raise ValueError("Must specify ssh, node_id, or provision=True")


def release_node(
    instance: ClientGPUInstance | None,
    keep_alive: bool = False,
) -> None:
    """Release a node after use.

    Args:
        instance: Instance from acquire_node (None for static SSH)
        keep_alive: If True, print reuse instructions instead of terminating

    Example:
        >>> client, instance = acquire_node(provision=True)
        >>> try:
        ...     # do work
        ... finally:
        ...     release_node(instance, keep_alive=args.keep_alive)
    """
    if instance is None:
        return

    if keep_alive:
        logger.info("")
        logger.info("💡 Instance kept alive: %s:%s", instance.provider, instance.id)
        logger.info("   Reuse with: --node-id %s:%s", instance.provider, instance.id)
        logger.info("   SSH: %s", instance.ssh_connection_string())
    else:
        logger.info("")
        logger.info("Terminating instance %s:%s...", instance.provider, instance.id)
        instance.terminate()
        logger.info("Instance terminated.")


def add_remote_args(parser: object) -> None:
    """Add standard remote execution arguments to an argparse parser.

    Adds mutually exclusive group:
        --ssh: Static SSH connection
        --node-id: Reuse existing instance
        --provision: Provision new instance

    Plus common options:
        --keep-alive: Don't terminate after completion
        --gpu-type: GPU type for provisioning
        --gpu-count: Number of GPUs
        --provider: Specific cloud provider

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_remote_args(parser)
        >>> args = parser.parse_args()
        >>> client, instance = acquire_node(
        ...     ssh=args.ssh,
        ...     node_id=args.node_id,
        ...     provision=args.provision,
        ...     gpu_type=args.gpu_type,
        ...     gpu_count=args.gpu_count,
        ...     provider=args.provider,
        ... )
    """
    # Node acquisition (mutually exclusive)
    node_group = parser.add_mutually_exclusive_group()
    node_group.add_argument("--ssh", help="Static SSH connection (e.g., root@gpu:22)")
    node_group.add_argument("--node-id", help="Existing instance ID (e.g., runpod:abc123)")
    node_group.add_argument("--provision", action="store_true", help="Provision new instance")

    # Common options
    parser.add_argument(
        "--keep-alive", action="store_true", help="Don't terminate instance after completion"
    )
    parser.add_argument(
        "--gpu-type", default="A100", help="GPU type for provisioning (default: A100)"
    )
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument(
        "--provider", help="Cloud provider (runpod, lambdalabs, vast, primeintellect)"
    )
