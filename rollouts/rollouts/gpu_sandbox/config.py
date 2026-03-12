"""Configuration for GPU sandbox providers.

Each config type specifies how to provision and connect to sandboxes.
The SandboxPool uses these to create workers on demand.

Uses broker (~/research/broker) for GPU provisioning across providers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from broker.types import GPUInstance


@dataclass(frozen=True)
class SandboxConfig:
    """Base config for all sandbox types."""

    count: int = 1  # Number of sandboxes to provision


@dataclass(frozen=True)
class LocalSandboxConfig(SandboxConfig):
    """Run scoring in local subprocess (no remote GPU).

    Useful for testing or when training node has spare GPU capacity.
    """

    timeout_seconds: int = 120


@dataclass(frozen=True)
class BrokerSandboxConfig(SandboxConfig):
    """Provision sandboxes via broker (RunPod, Modal, Lambda, etc.).

    Broker handles the provisioning complexity - we just specify GPU requirements.
    See ~/research/broker for details.

    Example:
        # RunPod A100
        config = BrokerSandboxConfig(gpu_type="A100", provider="runpod", count=2)

        # Any provider with H100
        config = BrokerSandboxConfig(gpu_type="H100", max_price=3.0)
    """

    gpu_type: str = "A100"  # GPU type to search for (e.g., "A100", "H100")
    provider: str | None = None  # Specific provider or None for any
    max_price: float | None = None  # Max price per hour
    timeout_seconds: int = 600
    keep_alive: bool = False  # Keep instance running after pool.stop()
    # Exposed ports for the scoring worker server
    exposed_ports: tuple[int, ...] = (9200,)  # Default scoring worker port
    # Docker image (provider-specific, broker handles defaults)
    docker_image: str | None = None


@dataclass(frozen=True)
class ExistingInstanceConfig(SandboxConfig):
    """Use existing broker GPUInstance(s).

    For reusing already-provisioned instances or connecting to
    instances provisioned outside the pool.
    """

    instances: tuple[GPUInstance, ...] = ()  # Pre-provisioned instances
    timeout_seconds: int = 600


# Legacy aliases for backwards compatibility
ModalSandboxConfig = BrokerSandboxConfig  # Use provider="modal"
RunPodSandboxConfig = BrokerSandboxConfig  # Use provider="runpod"
SSHSandboxConfig = ExistingInstanceConfig  # Use with pre-connected instances

# Type alias for any sandbox config
AnySandboxConfig = LocalSandboxConfig | BrokerSandboxConfig | ExistingInstanceConfig


def serialize_sandbox_config(config: AnySandboxConfig) -> dict[str, object]:
    data = asdict(config)
    if isinstance(config, LocalSandboxConfig):
        data["kind"] = "local"
    elif isinstance(config, BrokerSandboxConfig):
        data["kind"] = "broker"
    elif isinstance(config, ExistingInstanceConfig):
        data["kind"] = "existing"
    else:
        raise TypeError(f"Unsupported sandbox config type: {type(config)!r}")
    return data


def deserialize_sandbox_config(data: dict[str, object]) -> AnySandboxConfig:
    payload = {k: v for k, v in data.items() if k != "kind"}
    kind = data.get("kind")
    if kind == "local":
        return LocalSandboxConfig(**payload)
    if kind == "broker":
        return BrokerSandboxConfig(**payload)
    if kind == "existing":
        return ExistingInstanceConfig(**payload)
    raise ValueError(f"Unknown sandbox config kind: {kind!r}")
