"""
GPU Broker Client - Main interface for GPU operations
"""

import os
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from .query import GPUQuery, QueryType
from .types import CloudType, GPUInstance, GPUOffer


logger = logging.getLogger(__name__)


class GPUClient:
    """Main client for GPU broker operations

    Handles configuration for API keys, SSH keys, and provides
    all GPU search, provisioning, and management functionality.

    Examples:
        # Initialize with credentials and SSH key
        client = GPUClient(
            credentials={"runpod": "your-runpod-api-key", "vast": "your-vast-api-key"},
            ssh_key_path="~/.ssh/id_ed25519"
        )

        # Search and provision
        offers = client.search(client.gpu_type.contains("A100"))
        instance = client.create(offers[0])
    """

    def __init__(self, credentials: Dict[str, str], ssh_key_path: str):
        """Initialize GPU broker client

        Args:
            credentials: Dict mapping provider name to API key
                        e.g., {"runpod": "key1", "vast": "key2"}
            ssh_key_path: Path to SSH private key (required)

        Raises:
            AssertionError: If parameters are invalid
        """
        # Assert inputs (Tiger Style: assert everything, fail fast)
        assert isinstance(credentials, dict), "credentials must be dict"
        assert len(credentials) > 0, "credentials dict cannot be empty"
        for provider_name, api_key in credentials.items():
            assert isinstance(provider_name, str), f"Provider name must be string: {provider_name}"
            assert isinstance(api_key, str), f"API key must be string for {provider_name}"
            assert len(api_key) > 0, f"API key cannot be empty for {provider_name}"

        assert isinstance(ssh_key_path, str), "ssh_key_path must be string"
        assert len(ssh_key_path) > 0, "ssh_key_path cannot be empty"

        self._credentials = credentials
        self._ssh_key_path = os.path.expanduser(ssh_key_path)

        # Validate SSH key (Tiger Style: assert everything, fail fast)
        assert os.path.exists(self._ssh_key_path), \
            f"SSH private key not found: {self._ssh_key_path}"

        assert os.access(self._ssh_key_path, os.R_OK), \
            f"SSH private key not readable: {self._ssh_key_path}"

        # Warn on bad permissions (non-blocking)
        stat_info = os.stat(self._ssh_key_path)
        if stat_info.st_mode & 0o077:
            logger.warning(
                f"SSH key has insecure permissions: {oct(stat_info.st_mode)[-3:]}. "
                f"Recommend: chmod 600 {self._ssh_key_path}"
            )

        # Assert reasonable file size
        assert stat_info.st_size < 10_000, \
            f"SSH key file suspiciously large ({stat_info.st_size} bytes): {self._ssh_key_path}"

        self._query = GPUQuery()

        # Assert output invariants
        assert self._credentials, "Failed to set credentials"
        assert self._ssh_key_path, "Failed to set ssh_key_path"
        assert self._query is not None, "Failed to initialize query"

    def get_ssh_key_path(self) -> str:
        """Get configured SSH key path"""
        return self._ssh_key_path

    # Query interface - expose as properties
    @property
    def gpu_type(self):
        """Query by GPU type: client.gpu_type.contains('A100')"""
        return self._query.gpu_type

    @property
    def price_per_hour(self):
        """Query by price: client.price_per_hour < 2.0"""
        return self._query.price_per_hour

    @property
    def memory_gb(self):
        """Query by memory: client.memory_gb > 24"""
        return self._query.memory_gb

    @property
    def vram_gb(self):
        """Query by GPU VRAM: client.vram_gb >= 8"""
        return self._query.vram_gb

    @property
    def cloud_type(self):
        """Query by cloud type: client.cloud_type == CloudType.SECURE"""
        return self._query.cloud_type

    @property
    def provider(self):
        """Query by provider: client.provider.in_(['runpod', 'vast'])"""
        return self._query.provider

    @property
    def cuda_version(self):
        """Query by CUDA version: client.cuda_version.contains('12.0')"""
        return self._query.cuda_version

    @property
    def manufacturer(self):
        """Query by GPU manufacturer: client.manufacturer == 'nvidia'"""
        return self._query.manufacturer

    # Main API methods
    def search(
        self,
        query: Optional[QueryType] = None,
        sort: Optional[Callable[[Any], Any]] = None,
        reverse: bool = False
    ) -> List[GPUOffer]:
        """Search for GPU offers

        Args:
            query: DSL query (e.g., client.provider.in_(['runpod']) & client.gpu_type.contains("A100"))
            sort: Sort function (e.g., lambda x: x.memory_gb / x.price_per_hour)
            reverse: Sort descending

        Returns:
            List of GPU offers (unsorted by default - user controls sorting)
        """
        # Import here to avoid circular dependency
        from . import api

        return api.search(
            query=query,
            sort=sort,
            reverse=reverse,
            credentials=self._credentials
        )

    def create(
        self,
        query: Union[QueryType, List[GPUOffer], GPUOffer],
        image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        name: Optional[str] = None,
        gpu_count: int = 1,
        exposed_ports: Optional[List[int]] = None,
        enable_http_proxy: bool = True,
        max_attempts: int = 3,
        **kwargs
    ) -> Optional['ClientGPUInstance']:
        """Create GPU instance

        Args:
            query: Query or offer(s) to provision from
            image: Docker image to use
            name: Instance name
            gpu_count: Number of GPUs
            exposed_ports: List of ports to expose via HTTP proxy
            enable_http_proxy: Enable provider's HTTP proxy for exposed ports
            max_attempts: Maximum provisioning attempts

        Returns:
            GPU instance with client configuration
        """
        from . import api

        instance = api.create(
            query=query,
            image=image,
            name=name,
            gpu_count=gpu_count,
            exposed_ports=exposed_ports,
            enable_http_proxy=enable_http_proxy,
            max_attempts=max_attempts,
            credentials=self._credentials,
            **kwargs
        )

        if instance:
            return ClientGPUInstance(instance, self)
        return None

    def get_instance(self, instance_id: str, provider: str) -> Optional['ClientGPUInstance']:
        """Get instance details

        Args:
            instance_id: Instance identifier
            provider: Provider name (e.g., 'runpod', 'vast')
        """
        from . import api

        instance = api.get_instance(instance_id, provider, credentials=self._credentials)
        if instance:
            return ClientGPUInstance(instance, self)
        return None

    def terminate_instance(self, instance_id: str, provider: str) -> bool:
        """Terminate instance

        Args:
            instance_id: Instance identifier
            provider: Provider name (e.g., 'runpod', 'vast')
        """
        from . import api

        return api.terminate_instance(instance_id, provider, credentials=self._credentials)

    def list_instances(self, provider: Optional[str] = None) -> List['ClientGPUInstance']:
        """List all user's instances

        Args:
            provider: Optional provider filter (e.g., 'runpod', 'vast')
        """
        from . import api

        instances = api.list_instances(provider, credentials=self._credentials)
        return [ClientGPUInstance(instance, self) for instance in instances]


class ClientGPUInstance:
    """GPU instance with client configuration

    Wraps GPUInstance to use client's SSH key configuration
    """

    def __init__(self, instance: GPUInstance, client: GPUClient):
        self._instance = instance
        self._client = client

    def __getattr__(self, name):
        """Delegate attribute access to wrapped instance"""
        return getattr(self._instance, name)

    def exec(self, command: str, ssh_key_path: Optional[str] = None, timeout: int = 30):
        """Execute command using client's SSH configuration (synchronous)"""
        if ssh_key_path is None:
            ssh_key_path = self._client.get_ssh_key_path()

        return self._instance.exec(command, ssh_key_path, timeout)

    async def aexec(self, command: str, ssh_key_path: Optional[str] = None, timeout: int = 30):
        """Execute command using client's SSH configuration (asynchronous)"""
        if ssh_key_path is None:
            ssh_key_path = self._client.get_ssh_key_path()

        return await self._instance.aexec(command, ssh_key_path, timeout)

    def wait_until_ready(self, timeout: int = 300) -> bool:
        """Wait until instance is running"""
        return self._instance.wait_until_ready(timeout)

    def wait_until_ssh_ready(self, timeout: int = 300) -> bool:
        """Wait until SSH is ready"""
        return self._instance.wait_until_ssh_ready(timeout)

    def refresh(self) -> 'ClientGPUInstance':
        """Refresh the wrapped instance with latest data"""
        updated_instance = self._client.get_instance(self._instance.id, self._instance.provider)
        if updated_instance:
            self._instance = updated_instance._instance
            return self
        else:
            raise ValueError(f"Could not refresh instance {self._instance.id}")

    def terminate(self) -> bool:
        """Terminate this instance"""
        return self._instance.terminate()
