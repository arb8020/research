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

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


class InstanceNotFoundError(Exception):
    """Raised when trying to connect to an instance that no longer exists.

    This typically happens when:
    - Training completed and the instance was terminated
    - Spot/preemptible instance was reclaimed
    - Instance was manually deleted
    - Instance ID is incorrect
    """

    pass


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
    exposed_ports: tuple[int, ...] = ()
    enable_http_proxy: bool = True  # False for raw TCP ports (e.g. LogsServer)
    name: str | None = None  # Instance name (e.g. "rollouts/run_20250127-143052")
    provider: str | None = None  # Filter to specific provider (e.g., "runpod", "vast")
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"  # Docker image

    # Provider credentials (optional - falls back to env vars)
    credentials: dict[str, str] = field(default_factory=dict)

    # RunPod network volume support (datacenter-locked)
    network_volume_id: str | None = None
    datacenter_id: str | None = None


async def acquire_node(
    ssh: str | None = None,
    node_id: str | None = None,
    provision: GPUQuery | None = None,
    ssh_key_path: str | None = None,
    ssh_timeout: int = 600,
) -> tuple[BifrostClient, ClientGPUInstance | None]:
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

    # Build credentials: explicit > broker.credentials (env vars + ~/.broker/credentials.toml)
    credentials = {}
    if provision and provision.credentials:
        credentials = provision.credentials
    else:
        from broker.credentials import get_credentials

        credentials = get_credentials()

    assert credentials, "No provider credentials found. Run: broker auth login <provider>"

    broker = GPUClient(
        credentials=credentials,
        ssh_key_path=ssh_key_path,
    )

    # Mode 2: Reuse existing instance
    if node_id:
        assert ":" in node_id, f"node_id must be 'provider:instance_id', got: {node_id}"
        provider, instance_id = node_id.split(":", 1)

        logger.info("Connecting to existing instance: %s", node_id)
        instance = await broker.get_instance(instance_id, provider)
        if instance is None:
            # Instance not found - provide helpful error message
            raise InstanceNotFoundError(
                f"Instance '{node_id}' not found. "
                f"The instance may have been terminated (training completed, spot preemption, or manual deletion). "
                f"Use --provision to create a new instance, or check 'rollouts monitor --runs --probe' for live instances."
            )

        logger.info("  GPU: %dx %s", instance.gpu_count, instance.gpu_type)
        logger.info("  Waiting for SSH...")
        ssh_ready = await instance.wait_until_ssh_ready(timeout=ssh_timeout)
        if not ssh_ready:
            raise RuntimeError(
                f"SSH not ready after {ssh_timeout}s for instance {node_id}. "
                f"Instance may still be starting up - try again in a minute."
            )

        key_path = broker.get_ssh_key_path(provider)
        if key_path is None:
            raise RuntimeError(f"No SSH key configured for provider {provider}")
        client = BifrostClient(instance.ssh_connection_string(), ssh_key_path=key_path)
        return client, instance

    # Mode 3: Provision new instance
    assert provision is not None  # Type narrowing

    # Build query: GPU type filter, plus optional provider filter
    query = broker.gpu_type.contains(provision.type)
    if provision.provider:
        query = query & (broker.provider == provision.provider)

    logger.info("Provisioning new instance (%dx %s)...", provision.count, provision.type)
    if provision.provider:
        logger.info("  Provider filter: %s", provision.provider)
    instance = await broker.create(
        query,
        image=provision.image,
        name=provision.name,
        gpu_count=provision.count,
        network_volume_id=provision.network_volume_id,
        datacenter_id=provision.datacenter_id,
        cloud_type=provision.cloud_type,
        container_disk_gb=provision.container_disk_gb,
        volume_disk_gb=provision.volume_disk_gb,
        exposed_ports=list(provision.exposed_ports) if provision.exposed_ports else None,
        enable_http_proxy=provision.enable_http_proxy,
        sort=lambda x: x.price_per_hour,
        min_cuda_version=provision.min_cuda,
    )
    if instance is None:
        raise RuntimeError("Failed to provision instance - no suitable GPU offers found")
    logger.info("  Instance ID: %s:%s", instance.provider, instance.id)
    logger.info("  GPU: %dx %s", instance.gpu_count, instance.gpu_type)

    logger.info("  Waiting for SSH...")
    ssh_ready = await instance.wait_until_ssh_ready(timeout=ssh_timeout)
    if not ssh_ready:
        raise RuntimeError(
            f"SSH not ready after {ssh_timeout}s. Instance may still be starting up. "
            f"Try again with --node-id {instance.provider}:{instance.id}"
        )

    key_path = broker.get_ssh_key_path(instance.provider)
    if key_path is None:
        raise RuntimeError(f"No SSH key configured for provider {instance.provider}")
    client = BifrostClient(instance.ssh_connection_string(), ssh_key_path=key_path)
    return client, instance
