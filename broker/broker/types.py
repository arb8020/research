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
    
    
    def ssh_connection_string(self) -> str:
        """Get SSH connection string for use with bifrost or other tools.
        
        Returns:
            SSH connection string in format: user@host:port
            
        Raises:
            ValueError: If instance SSH details not available
        """
        self._validate_ssh_ready()
        return f"{self.ssh_username}@{self.public_ip}:{self.ssh_port}"
    
    def terminate(self) -> bool:
        """Terminate this instance"""
        from .api import terminate_instance
        return terminate_instance(self.id, self.provider, api_key=self.api_key)
    
    def wait_until_ready(self, timeout: int = 300) -> bool:
        """Wait until instance status is RUNNING"""
        import time

        from .api import get_instance

        start_time = time.time()

        while time.time() - start_time < timeout:
            updated_instance = get_instance(self.id, self.provider, api_key=self.api_key)
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
    
    def wait_until_ssh_ready(self, timeout: int = 300) -> bool:
        """Wait until instance is running AND SSH is ready for connections"""
        import time
        
        start_time = time.time()
        
        # First wait for instance to be running
        if not self.wait_until_ready(timeout=min(timeout, 300)):
            return False
        
        # Wait for SSH assignment (direct or proxy)
        if not self._wait_for_ssh_assignment(start_time, timeout):
            return False
        
        # Test SSH connectivity
        return self._test_ssh_connectivity()
    
    def _wait_for_ssh_assignment(self, start_time: float, timeout: int) -> bool:
        """Wait for direct SSH to be assigned (not proxy)."""
        import time
        import logging

        logger = logging.getLogger(__name__)
        from .providers.runpod import get_instance_details

        logger.info("Waiting for direct SSH to be assigned...")
        logger.info("Note: This may take up to 10 minutes. Proxy SSH will be ignored.")

        while time.time() - start_time < timeout:
            # Get fresh data directly from API (like broker list does)
            fresh_instance = get_instance_details(self.id, api_key=self.api_key)

            if fresh_instance and fresh_instance.public_ip and fresh_instance.ssh_port:
                if fresh_instance.public_ip != "ssh.runpod.io":
                    # Update current instance with fresh data
                    self.public_ip = fresh_instance.public_ip
                    self.ssh_port = fresh_instance.ssh_port
                    self.ssh_username = fresh_instance.ssh_username
                    self.status = fresh_instance.status

                    logger.info(f"Direct SSH assigned: {self.public_ip}:{self.ssh_port}")
                    return True

            self._log_ssh_wait_status(start_time)
            time.sleep(10)

        elapsed_minutes = int((time.time() - start_time) / 60)
        logger.error(f"Timeout waiting for direct SSH after {elapsed_minutes} minutes")
        logger.error("Direct SSH was not assigned within the timeout period.")
        logger.error("The instance will be cleaned up automatically.")
        return False
    
    def _has_direct_ssh_details(self) -> bool:
        """Check if instance has direct SSH details (not proxy)."""
        return (
            self.public_ip and 
            self.ssh_port and 
            self.public_ip != "ssh.runpod.io"
        )
    
    def _log_ssh_wait_status(self, start_time: float) -> None:
        """Log current SSH wait status with timing info."""
        import time
        import logging

        logger = logging.getLogger(__name__)
        elapsed = int(time.time() - start_time)

        if self.public_ip and self.ssh_port:
            if self.public_ip == "ssh.runpod.io":
                logger.debug(f"Still waiting for direct SSH (currently proxy) - {elapsed}s elapsed...")
            else:
                logger.debug(f"SSH details available but not recognized as direct - {elapsed}s elapsed...")
        else:
            logger.debug(f"Still waiting for SSH details - {elapsed}s elapsed...")
    
    def _test_ssh_connectivity(self) -> bool:
        """Test SSH connectivity with a simple command."""
        import time
        import logging

        logger = logging.getLogger(__name__)

        logger.info("Got direct SSH! Waiting for SSH daemon to initialize...")
        time.sleep(30)  # SSH daemons typically need time to start

        try:
            result = self.exec("echo 'ssh_ready'", timeout=30)
            if result.success and "ssh_ready" in result.stdout:
                logger.info("SSH connectivity confirmed!")
                return True
            else:
                logger.warning(f"SSH test failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"SSH connection error: {e}")
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
    image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
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