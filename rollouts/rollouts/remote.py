"""Remote execution utilities for rollouts.

Unified node acquisition pattern from kerbal/tests/test_integration.py.
Supports static SSH, existing broker instances, or provisioning new ones.

Usage:
    from rollouts.remote import acquire_node, get_broker_credentials

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

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()

if TYPE_CHECKING:
    from bifrost import BifrostClient
    from broker.client import ClientGPUInstance


def get_broker_credentials() -> dict[str, str]:
    """Load broker credentials from environment.

    Checks for:
        - RUNPOD_API_KEY
        - VAST_API_KEY
        - PRIME_API_KEY
        - LAMBDA_API_KEY

    Returns:
        Dict mapping provider name to API key
    """
    credentials = {}
    if key := os.getenv("RUNPOD_API_KEY"):
        credentials["runpod"] = key
    if key := os.getenv("VAST_API_KEY"):
        credentials["vast"] = key
    if key := os.getenv("PRIME_API_KEY"):
        credentials["primeintellect"] = key
    if key := os.getenv("LAMBDA_API_KEY"):
        credentials["lambdalabs"] = key
    return credentials


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
        print(f"Connecting to static node: {ssh}")
        client = BifrostClient(ssh, ssh_key_path=ssh_key_path)
        return client, None

    elif node_id:
        # Existing instance - look it up
        node_provider, instance_id = node_id.split(":", 1)
        print(f"Connecting to existing instance: {node_provider}:{instance_id}")

        credentials = get_broker_credentials()
        assert credentials, "No broker credentials found in environment"

        broker = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)
        instance = broker.get_instance(instance_id, node_provider)
        assert instance, f"Instance not found: {node_id}"

        print(f"  GPU: {instance.gpu_count}x {instance.gpu_type}")
        print("  Waiting for SSH...")
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
        print(f"Provisioning new instance ({gpu_count}x {gpu_type})...")

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

        print(f"  Instance ID: {instance.provider}:{instance.id}")
        print(f"  GPU: {instance.gpu_count}x {instance.gpu_type}")
        print(f"  Price: ${instance.price_per_hour:.2f}/hr")
        print("  Waiting for SSH...")

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
        print(f"\n💡 Instance kept alive: {instance.provider}:{instance.id}")
        print(f"   Reuse with: --node-id {instance.provider}:{instance.id}")
        print(f"   SSH: {instance.ssh_connection_string()}")
    else:
        print(f"\nTerminating instance {instance.provider}:{instance.id}...")
        instance.terminate()
        print("Instance terminated.")


def add_remote_args(parser) -> None:
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
