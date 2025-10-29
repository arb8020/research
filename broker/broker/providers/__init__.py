"""
GPU Provider implementations and protocol.

All providers must implement the ProviderProtocol to ensure consistent behavior.
"""

from typing import Protocol, Optional
import logging

logger = logging.getLogger(__name__)


class ProviderProtocol(Protocol):
    """Protocol that all GPU providers must implement.

    Tiger Style: Explicit contracts via protocols.
    Each method has clear pre/postconditions.
    """

    def wait_for_ssh_ready(self, instance, timeout: int) -> bool:
        """Wait until SSH is ready on instance.

        Args:
            instance: GPU instance to wait for (GPUInstance type)
            timeout: Maximum seconds to wait

        Returns:
            True if SSH ready, False if timeout/failure

        Provider-specific implementation should:
        1. Poll for SSH details (IP/port/username)
        2. Handle provider-specific quirks (proxy vs direct, etc)
        3. Test connectivity with simple command
        4. Update instance object with final details
        """
        ...

    def get_fresh_instance(self, instance_id: str, api_key: str):
        """Get fresh instance data from provider API.

        Args:
            instance_id: Instance identifier
            api_key: Provider API key

        Returns:
            Updated instance or None if not found (GPUInstance or None)

        Used by wait_for_ssh_ready to get latest SSH details.
        """
        ...


def get_provider_impl(provider_name: str):
    """Get provider implementation module.

    Args:
        provider_name: Provider name (e.g., "runpod", "primeintellect")

    Returns:
        Provider module implementing ProviderProtocol

    Raises:
        ValueError: If provider not supported
    """
    assert isinstance(provider_name, str), "provider_name must be string"
    assert len(provider_name) > 0, "provider_name cannot be empty"

    if provider_name == "runpod":
        from . import runpod
        return runpod
    elif provider_name == "primeintellect":
        from . import primeintellect
        return primeintellect
    elif provider_name == "lambdalabs":
        from . import lambdalabs
        return lambdalabs
    elif provider_name == "vast":
        from . import vast
        return vast
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")
