"""
GPU Broker Client - Main interface for GPU operations
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union, cast

from shared.validation import validate_ssh_key_path
from .query import GPUQuery, QueryType
from .types import CloudType, GPUInstance, GPUOffer, InstanceStatus, ProviderCredentials
from .validation import validate_credentials


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

    def __init__(
        self,
        credentials: Union[ProviderCredentials, Dict[str, str]],
        ssh_key_path: Optional[str] = None,
        ssh_key_paths: Optional[Dict[str, str]] = None
    ):
        """Initialize GPU broker client

        Args:
            credentials: Provider credentials (ProviderCredentials or dict)
                        e.g., {"runpod": "key1", "vast": "key2"}
                        or ProviderCredentials(runpod="key1", vast="key2")
            ssh_key_path: Default SSH private key path for all providers (optional)
            ssh_key_paths: Per-provider SSH key paths (optional)
                          e.g., {"lambdalabs": "~/.ssh/lambda_key"}
                          Provider-specific keys override the default ssh_key_path

        Raises:
            AssertionError: If parameters are invalid
        """
        # Convert dict to ProviderCredentials if needed (backward compatibility)
        if isinstance(credentials, dict):
            credentials_dict = cast(Dict[str, str], credentials)
            validate_credentials(credentials_dict)  # Validate before conversion
            credentials = ProviderCredentials.from_dict(credentials_dict)

        # Assert credentials type after conversion
        assert isinstance(credentials, ProviderCredentials), \
            "credentials must be ProviderCredentials or dict"

        # Validate and store credentials (validation helper contains all assertions)
        self._credentials = credentials.to_dict()

        # Validate and store SSH key path if provided (validation helper contains all assertions)
        self._ssh_key_path = validate_ssh_key_path(ssh_key_path) if ssh_key_path else None

        # Validate and store per-provider SSH key paths
        self._ssh_key_paths: Dict[str, str] = {}
        if ssh_key_paths:
            assert isinstance(ssh_key_paths, dict), "ssh_key_paths must be dict"
            for provider, key_path in ssh_key_paths.items():
                assert isinstance(provider, str), f"provider must be string, got {type(provider)}"
                assert provider in self._credentials, f"Unknown provider in ssh_key_paths: {provider}"
                validated_path = validate_ssh_key_path(key_path)
                self._ssh_key_paths[provider] = validated_path

        # Initialize query interface
        self._query = GPUQuery()

        # Assert output invariants
        assert self._credentials, "Failed to set credentials"
        assert self._query is not None, "Failed to initialize query"
        assert isinstance(self._ssh_key_paths, dict), "Failed to set ssh_key_paths"

    def get_ssh_key_path(self, provider: Optional[str] = None) -> Optional[str]:
        """Get SSH key path for a specific provider with precedence.

        Args:
            provider: Provider name to get SSH key for (optional)

        Returns:
            SSH key path with precedence:
            1. Provider-specific key from ssh_key_paths (highest)
            2. Default ssh_key_path (fallback)
            3. None (no SSH key configured)
        """
        if provider and provider in self._ssh_key_paths:
            return self._ssh_key_paths[provider]
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
    def underlying_provider(self):
        """Query by underlying provider (for aggregators): client.underlying_provider == 'massedcompute'"""
        return self._query.underlying_provider

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
        image: str = "runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204",
        name: Optional[str] = None,
        gpu_count: int = 1,
        exposed_ports: Optional[List[int]] = None,
        enable_http_proxy: bool = True,
        n_offers: int = 3,
        cloud_type: Optional[Union[str, CloudType]] = None,
        sort: Optional[Callable[[Any], Any]] = None,
        reverse: bool = False,
        **kwargs
    ) -> Optional['ClientGPUInstance']:
        """Create GPU instance

        Args:
            query: Query or offer(s) to provision from. Can be:
                   - Single GPUOffer: Tries to provision that specific offer
                   - List of GPUOffers: Tries offers in order until one succeeds
                   - Query object: Searches for matching offers and tries them in order
            image: Docker image to use
            name: Instance name
            gpu_count: Number of GPUs
            exposed_ports: List of ports to expose via HTTP proxy
            enable_http_proxy: Enable provider's HTTP proxy for exposed ports
            n_offers: Number of offers to try before giving up.
                     If query is a list, tries up to this many offers from the list.
                     If query is a search filter, tries up to this many from search results.
            cloud_type: Cloud deployment type ("secure", "community", or CloudType enum)
            sort: Sort function for ordering offers (e.g., lambda x: x.price_per_hour)
            reverse: Sort descending

        Returns:
            GPU instance with client configuration
        """
        from . import api

        # Handle cloud_type parameter by adding it to the query
        if cloud_type is not None and not isinstance(query, (list, GPUOffer)):
            # Convert string to enum if needed
            if isinstance(cloud_type, str):
                cloud_enum = CloudType.SECURE if cloud_type.lower() == "secure" else CloudType.COMMUNITY
            else:
                cloud_enum = cloud_type

            # Add cloud_type to query
            if query is None:
                query = (self.cloud_type == cloud_enum)
            else:
                query = query & (self.cloud_type == cloud_enum)

        instance = api.create(
            query=query,
            image=image,
            name=name,
            gpu_count=gpu_count,
            exposed_ports=exposed_ports,
            enable_http_proxy=enable_http_proxy,
            n_offers=n_offers,
            sort=sort,
            reverse=reverse,
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

    # Explicit attribute forwarding for type safety
    @property
    def id(self) -> str:
        """Instance ID"""
        return self._instance.id

    @property
    def provider(self) -> str:
        """Provider name (e.g., 'runpod', 'vast')"""
        return self._instance.provider

    @property
    def status(self) -> InstanceStatus:
        """Instance status (InstanceStatus enum)"""
        return self._instance.status

    @property
    def gpu_type(self) -> str:
        """GPU type (e.g., 'NVIDIA RTX A4000')"""
        return self._instance.gpu_type

    @property
    def gpu_count(self) -> int:
        """Number of GPUs"""
        return self._instance.gpu_count

    @property
    def price_per_hour(self) -> float:
        """Price per hour in USD"""
        return self._instance.price_per_hour

    @property
    def name(self) -> Optional[str]:
        """Instance name"""
        return self._instance.name

    @property
    def public_ip(self) -> Optional[str]:
        """Public IP address"""
        return self._instance.public_ip

    @property
    def ssh_port(self) -> Optional[int]:
        """SSH port"""
        return self._instance.ssh_port

    @property
    def ssh_username(self) -> Optional[str]:
        """SSH username"""
        return self._instance.ssh_username

    @property
    def raw_data(self) -> Optional[Dict[str, Any]]:
        """Raw provider API data"""
        return self._instance.raw_data

    def ssh_connection_string(self) -> str:
        """Get SSH connection string (user@host:port)"""
        return self._instance.ssh_connection_string()

    def exec(self, command: str, ssh_key_path: Optional[str] = None, timeout: int = 30):
        """Execute command using client's SSH configuration (synchronous)"""
        if ssh_key_path is None:
            # Get provider-specific SSH key with fallback to default
            ssh_key_path = self._client.get_ssh_key_path(provider=self._instance.provider)

        return self._instance.exec(command, ssh_key_path, timeout)

    async def aexec(self, command: str, ssh_key_path: Optional[str] = None, timeout: int = 30):
        """Execute command using client's SSH configuration (asynchronous)"""
        if ssh_key_path is None:
            # Get provider-specific SSH key with fallback to default
            ssh_key_path = self._client.get_ssh_key_path(provider=self._instance.provider)

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
