"""Node provisioning for bifrost.

Extracts the common acquire_node pattern from multiple deploy.py files.
This module handles the tri-modal node acquisition:
- ssh: Use existing SSH connection string
- node_id: Reuse existing broker instance
- provision: Provision new instance via broker

Lives in bifrost (not broker) because it returns a BifrostClient.

Tiger Style:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from broker.client import ClientGPUInstance

    from .client import BifrostClient


@dataclass(frozen=True)
class GPUQuery:
    """GPU provisioning query.

    Immutable specification of what GPU resources to provision.
    Used by acquire_node() when provision mode is selected.
    """

    type: str = "A100"
    count: int = 1
    max_price: float | None = None
    min_vram_gb: int | None = None
    min_cuda: str = "12.0"
    cloud_type: str = "secure"
    container_disk_gb: int = 100
    volume_disk_gb: int = 0

    # Provider credentials (optional - falls back to env vars)
    credentials: dict[str, str] = field(default_factory=dict)


def acquire_node(
    ssh: str | None = None,
    node_id: str | None = None,
    provision: GPUQuery | None = None,
    ssh_key_path: str | None = None,
    ssh_timeout: int = 600,
) -> tuple["BifrostClient", "ClientGPUInstance | None"]:
    """Acquire a node and return a BifrostClient.

    Tri-modal acquisition:
    - ssh: Connect to existing node via SSH string (e.g., "root@gpu:22")
    - node_id: Reuse existing broker instance (e.g., "runpod:abc123")
    - provision: Provision new instance via broker using GPUQuery

    Exactly one of ssh, node_id, or provision must be specified.

    Args:
        ssh: SSH connection string (user@host:port)
        node_id: Existing instance ID (provider:instance_id)
        provision: GPUQuery for provisioning new instance
        ssh_key_path: Path to SSH private key (default: ~/.ssh/id_ed25519)
        ssh_timeout: Timeout in seconds for SSH readiness (default: 600)

    Returns:
        (BifrostClient, ClientGPUInstance | None)
        Instance is None when using ssh mode (no broker instance).

    Raises:
        AssertionError: Invalid arguments
        RuntimeError: Failed to acquire node

    Example:
        # Static SSH connection
        client, _ = acquire_node(ssh="root@gpu.example.com:22")

        # Reuse existing broker instance
        client, instance = acquire_node(node_id="runpod:abc123")

        # Provision new instance
        client, instance = acquire_node(
            provision=GPUQuery(type="A100", count=2)
        )
    """
    # Validate exactly one mode specified
    modes = [ssh is not None, node_id is not None, provision is not None]
    assert sum(modes) == 1, "Must specify exactly one of: ssh, node_id, provision"

    # Default SSH key path
    if ssh_key_path is None:
        ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")

    # Import here to avoid circular dependency
    from .client import BifrostClient

    # Mode 1: Static SSH connection
    if ssh:
        return BifrostClient(ssh, ssh_key_path=ssh_key_path), None

    # Mode 2 & 3: Broker-based acquisition
    from broker.client import GPUClient

    # Build credentials from env vars if not provided
    credentials = {}
    if provision and provision.credentials:
        credentials = provision.credentials
    else:
        # Default credentials from environment
        # Only use runpod by default - other providers can be added via GPUQuery.credentials
        runpod_key = os.getenv("RUNPOD_API_KEY")

        if runpod_key:
            credentials["runpod"] = runpod_key

    assert credentials, "No provider credentials found. Set RUNPOD_API_KEY, VAST_API_KEY, or PRIME_API_KEY"

    broker = GPUClient(
        credentials=credentials,
        ssh_key_path=ssh_key_path,
    )

    # Mode 2: Reuse existing instance
    if node_id:
        assert ":" in node_id, f"node_id must be 'provider:instance_id', got: {node_id}"
        provider, instance_id = node_id.split(":", 1)

        print(f"Connecting to existing instance: {node_id}")
        instance = broker.get_instance(instance_id, provider)
        assert instance is not None, f"Instance not found: {node_id}"

        print(f"  GPU: {instance.gpu_count}x {instance.gpu_type}")
        print("  Waiting for SSH...")
        instance.wait_until_ssh_ready(timeout=ssh_timeout)

        key_path = broker.get_ssh_key_path(provider)
        client = BifrostClient(instance.ssh_connection_string(), ssh_key_path=key_path)
        return client, instance

    # Mode 3: Provision new instance
    assert provision is not None  # Type narrowing

    print(f"Provisioning new instance ({provision.count}x {provision.type})...")
    instance = broker.create(
        broker.gpu_type.contains(provision.type),
        gpu_count=provision.count,
        cloud_type=provision.cloud_type,
        container_disk_gb=provision.container_disk_gb,
        volume_disk_gb=provision.volume_disk_gb,
        sort=lambda x: x.price_per_hour,
        min_cuda_version=provision.min_cuda,
    )
    print(f"  Instance ID: {instance.provider}:{instance.id}")
    print(f"  GPU: {instance.gpu_count}x {instance.gpu_type}")

    print("  Waiting for SSH...")
    instance.wait_until_ssh_ready(timeout=ssh_timeout)

    key_path = broker.get_ssh_key_path(instance.provider)
    client = BifrostClient(instance.ssh_connection_string(), ssh_key_path=key_path)
    return client, instance
