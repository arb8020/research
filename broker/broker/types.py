"""
Core data types for GPU cloud operations
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .ssh_clients_compat import execute_command_sync, execute_command_async


class InstanceStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    STOPPED = "stopped"
    TERMINATED = "terminated"
    FAILED = "failed"


class GPUAvailability(str, Enum):
    """GPU availability status"""
    IMMEDIATE = "immediate"
    QUEUED = "queued"
    UNAVAILABLE = "unavailable"


class CloudType(str, Enum):
    """Cloud deployment type for GPU instances"""
    SECURE = "secure"
    COMMUNITY = "community" 
    ALL = "all"


@dataclass
class GPUOffer:
    """A GPU offer from a provider"""
    id: str
    provider: str
    gpu_type: str
    gpu_count: int
    vcpu: int
    memory_gb: int
    storage_gb: int
    price_per_hour: float
    availability_zone: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    
    # Additional fields for pandas-style queries
    vram_gb: Optional[int] = None
    region: Optional[str] = None
    availability: Optional[GPUAvailability] = None
    spot: bool = False
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    cloud_type: Optional[CloudType] = None
    manufacturer: Optional[str] = None


@dataclass
class GPUInstance:
    """A provisioned GPU instance with convenience methods"""
    id: str
    provider: str
    status: InstanceStatus
    gpu_type: str
    gpu_count: int
    price_per_hour: float
    name: Optional[str] = None
    public_ip: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_username: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None  # Store API key for internal API calls
    
    def exec(self, command: str, ssh_key_path: Optional[str] = None, timeout: int = 30) -> 'SSHResult':
        """Execute command via SSH using configured key (synchronous)"""
        self._validate_ssh_ready()
        key_content = self._load_ssh_key(ssh_key_path)
        
        exit_code, stdout, stderr = execute_command_sync(
            self, key_content, command, timeout=timeout
        )
        
        return SSHResult(
            success=exit_code == 0,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code
        )
    
    async def aexec(self, command: str, ssh_key_path: Optional[str] = None, timeout: int = 30) -> 'SSHResult':
        """Execute command via SSH using configured key (asynchronous)
        
        Args:
            command: Command to execute
            ssh_key_path: Path to SSH private key file
            timeout: Command timeout in seconds
            
        Returns:
            SSHResult with command output
            
        Example:
            # Single async command
            result = await instance.aexec("nvidia-smi")
            
            # Multiple commands in parallel
            results = await asyncio.gather(
                instance.aexec("nvidia-smi"),
                instance.aexec("df -h"),
                instance.aexec("ps aux")
            )
        """
        self._validate_ssh_ready()
        key_content = self._load_ssh_key(ssh_key_path)
        
        exit_code, stdout, stderr = await execute_command_async(
            self, key_content, command, timeout=timeout
        )
        
        return SSHResult(
            success=exit_code == 0,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code
        )
    
    def _validate_ssh_ready(self) -> None:
        """Validate that instance has SSH connection details available"""
        if not self.public_ip or not self.ssh_username:
            raise ValueError("Instance SSH details not available - may not be running yet")
    
    def _load_ssh_key(self, ssh_key_path: Optional[str]) -> Optional[str]:
        """Load SSH private key content from file path"""
        import os
        
        if not ssh_key_path:
            return None
        
        key_path = os.path.expanduser(ssh_key_path)
        try:
            with open(key_path) as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to load SSH key from {key_path}: {e}") from e
    
    
    def ssh_connection_string(self, ssh_key_path: Optional[str] = None, full_command: bool = False) -> str:
        """Get SSH connection string for use with bifrost or other tools.

        Args:
            ssh_key_path: Optional SSH key path to include in command
            full_command: If True, returns full 'ssh -p port user@host' command.
                         If False, returns 'user@host:port' format.

        Returns:
            SSH connection string in format:
            - If full_command=True: 'ssh -p <port> -i <key> <user>@<host>' (or without -i if no key)
            - If full_command=False: 'user@host:port'

        Raises:
            ValueError: If instance SSH details not available
        """
        self._validate_ssh_ready()

        if full_command:
            parts = ["ssh", "-p", str(self.ssh_port)]
            if ssh_key_path:
                parts.extend(["-i", ssh_key_path])
            parts.append(f"{self.ssh_username}@{self.public_ip}")
            return " ".join(parts)
        else:
            return f"{self.ssh_username}@{self.public_ip}:{self.ssh_port}"
    
    def terminate(self) -> bool:
        """Terminate this instance"""
        from .api import terminate_instance
        credentials = {self.provider: self.api_key} if self.api_key else None
        return terminate_instance(self.id, self.provider, credentials=credentials)
    
    def wait_until_ready(self, timeout: int = 600) -> bool:
        """Wait until instance status is RUNNING"""
        import time

        from .api import get_instance

        start_time = time.time()

        while time.time() - start_time < timeout:
            credentials = {self.provider: self.api_key} if self.api_key else None
            updated_instance = get_instance(self.id, self.provider, credentials=credentials)
            if not updated_instance:
                return False
                
            if updated_instance.status == InstanceStatus.RUNNING:
                # Update this instance with new details
                self.__dict__.update(updated_instance.__dict__)
                return True
            elif updated_instance.status in [InstanceStatus.FAILED, InstanceStatus.TERMINATED]:
                return False
                
            time.sleep(15)  # Check every 15 seconds
        
        return False  # Timeout
    
    def wait_until_ssh_ready(self, timeout: int = 600) -> bool:
        """Wait until instance is running AND SSH is ready for connections.

        Delegates to provider-specific implementation since SSH setup varies
        across providers (proxy vs direct, timing, authentication, etc).

        Args:
            timeout: Maximum seconds to wait (default: 600 = 5min)

        Returns:
            True if SSH ready, False if timeout/failure
        """
        import logging
        from .providers import get_provider_impl

        logger = logging.getLogger(__name__)

        # Assert preconditions (Tiger Style)
        assert isinstance(timeout, int) and timeout > 0, \
            f"timeout must be positive int, got {timeout}"
        assert self.provider, "Instance missing provider"
        assert self.api_key, "Instance missing API key"

        try:
            # Get provider-specific implementation
            provider = get_provider_impl(self.provider)

            # Delegate to provider
            result = provider.wait_for_ssh_ready(self, timeout)

            # Assert postconditions
            if result:
                assert self.public_ip, "SSH ready but no public_ip"
                assert self.ssh_port, "SSH ready but no ssh_port"
                assert self.ssh_username, "SSH ready but no ssh_username"

            return result

        except Exception as e:
            logger.error(f"wait_until_ssh_ready failed: {e}")
            return False
    
    def refresh(self) -> 'GPUInstance':
        """Refresh instance details from provider"""
        from .api import get_instance
        
        updated_instance = get_instance(self.id, self.provider)
        if updated_instance:
            self.__dict__.update(updated_instance.__dict__)
            return self
        else:
            raise ValueError(f"Could not refresh instance {self.id}")
    


@dataclass
class ProvisionRequest:
    """Request to provision a GPU instance"""
    gpu_type: Optional[str] = None
    gpu_count: int = 1
    image: str = "runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204"
    name: Optional[str] = None
    max_price_per_hour: Optional[float] = None
    provider: Optional[str] = None  # If None, search all providers
    spot_instance: bool = False
    ssh_startup_script: Optional[str] = None  # SSH key injection script
    container_disk_gb: Optional[int] = None  # Container disk size in GB (default: 50)
    volume_disk_gb: Optional[int] = None  # Volume disk size in GB (default: 0)
    memory_gb: Optional[int] = None  # System memory allocation in GB (default: provider minimum)
    # Port exposure configuration
    exposed_ports: Optional[List[int]] = None  # Ports to expose via HTTP proxy
    enable_http_proxy: bool = True  # Enable RunPod's HTTP proxy
    manufacturer: Optional[str] = None  # GPU manufacturer filter (e.g., "nvidia", "amd")
    # Jupyter configuration
    start_jupyter: bool = False  # Auto-start Jupyter Lab
    jupyter_password: Optional[str] = None  # Jupyter authentication token


@dataclass
class SSHResult:
    """Result of SSH command execution"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    command: Optional[str] = None


@dataclass
class SSHConfig:
    """SSH connection configuration"""
    hostname: str
    port: int
    username: str
    key_path: Optional[str] = None
    method: Optional[str] = None  # "direct" or "proxy"


# ============================================================================
# New Frozen Dataclasses (Type Improvements)
# ============================================================================


@dataclass(frozen=True)
class ProviderCredentials:
    """API credentials for cloud GPU providers.

    Supports multiple providers (RunPod, Prime Intellect, etc).
    Immutable to prevent accidental credential leaks.
    """
    runpod: str = ""
    primeintellect: str = ""
    # Add more providers as needed

    def __post_init__(self):
        # Tiger Style: assert at least one credential provided
        assert self.runpod or self.primeintellect, \
            "At least one provider credential required"

        # Validate credential format (basic length check)
        if self.runpod:
            assert len(self.runpod) > 10, \
                "RunPod API key appears invalid (too short)"

        if self.primeintellect:
            assert len(self.primeintellect) > 10, \
                "Prime Intellect API key appears invalid (too short)"

        # Assert output invariant
        assert self.runpod or self.primeintellect, "credentials validated"

    def get(self, provider: str) -> Optional[str]:
        """Get credential for specific provider."""
        if provider == "runpod":
            return self.runpod
        elif provider == "primeintellect":
            return self.primeintellect
        return None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dict for backward compatibility."""
        result = {}
        if self.runpod:
            result["runpod"] = self.runpod
        if self.primeintellect:
            result["primeintellect"] = self.primeintellect
        return result

    @classmethod
    def from_dict(cls, credentials: Dict[str, str]) -> 'ProviderCredentials':
        """Create from dict (for backward compatibility)."""
        return cls(
            runpod=credentials.get("runpod", ""),
            primeintellect=credentials.get("primeintellect", "")
        )
