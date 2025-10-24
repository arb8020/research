"""
Prime Intellect provider implementation
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from ..types import CloudType, GPUInstance, GPUOffer, InstanceStatus, ProvisionRequest

logger = logging.getLogger(__name__)

PRIME_API_BASE_URL = "https://api.primeintellect.ai/api/v1"


def _make_api_request(method: str, endpoint: str, data: Optional[Dict] = None, 
                     params: Optional[Dict] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Make a REST API request to Prime Intellect API

    Args:
        method: HTTP method (GET, POST, DELETE, etc.)
        endpoint: API endpoint (e.g., "/availability/")
        data: Optional request body data
        params: Optional query parameters
        api_key: Prime Intellect API key (required)
    """
    if not api_key:
        raise ValueError("Prime Intellect API key is required but was not provided")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    url = f"{PRIME_API_BASE_URL}{endpoint}"
    logger.debug("Prime Intellect API request %s %s (api key ...%s)", method, url, api_key[-4:] if api_key else "none")
    
    try:
        response = requests.request(
            method=method,
            url=url,
            json=data,
            params=params,
            headers=headers,
            timeout=(10, 30),  # Connect timeout, read timeout
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        logger.error("Prime Intellect API request timed out")
        raise
    except requests.RequestException as exc:
        logger.error(f"Prime Intellect API request failed: {exc}")
        raise

    # Handle empty responses (e.g., DELETE operations)
    if response.status_code == 204 or not response.content:
        return {}
        
    return response.json()


def search_gpu_offers(cuda_version: Optional[str] = None, manufacturer: Optional[str] = None,
                      memory_gb: Optional[int] = None, container_disk_gb: Optional[int] = None,
                      gpu_count: int = 1, api_key: Optional[str] = None) -> List[GPUOffer]:
    """Search for available GPU offers on Prime Intellect with optional filtering

    Note: Prime Intellect's availability API doesn't support gpu_count as a query parameter,
    but we filter the results client-side to match the requested gpu_count.
    """

    # Build query parameters for availability API
    params = {}

    # Add CUDA version filter if specified
    if cuda_version:
        params["cuda_version"] = cuda_version

    # Add memory filter if specified
    if memory_gb:
        params["min_memory"] = memory_gb

    try:
        data = _make_api_request("GET", "/availability/", params=params, api_key=api_key)
        offers = []

        # Prime Intellect API returns data grouped by GPU type
        for gpu_type, gpu_offers in data.items():
            # Skip CPU-only offers - we want GPU offers
            if gpu_type == "CPU_NODE":
                continue

            for offer in gpu_offers:
                # Filter by GPU count (client-side filtering)
                if offer.get("gpuCount", 1) != gpu_count:
                    continue

                # Filter by manufacturer if specified
                if manufacturer and offer.get("provider"):
                    # Note: Prime Intellect doesn't have manufacturer field, using provider as proxy
                    if manufacturer.lower() not in offer["provider"].lower():
                        continue
                
                # Determine cloud type from security field
                cloud_type = CloudType.SECURE if offer.get("security") == "secure_cloud" else CloudType.COMMUNITY

                # Extract pricing - prefer onDemand, fallback to communityPrice
                # Note: PrimeIntellect returns total node price for multi-GPU offers
                # We normalize to per-GPU pricing for consistency with other providers
                total_price = 0.0
                prices = offer.get("prices", {})
                if prices.get("onDemand"):
                    total_price = prices["onDemand"]
                elif prices.get("communityPrice"):
                    total_price = prices["communityPrice"]

                # Normalize to per-GPU pricing
                offer_gpu_count = offer.get("gpuCount", 1)
                price_per_hour = total_price / offer_gpu_count if offer_gpu_count > 0 else total_price

                # Create unique offer ID
                offer_id = f"prime-{offer.get('cloudId', 'unknown')}-{offer.get('dataCenter', 'unknown')}"
                
                gpu_offer = GPUOffer(
                    id=offer_id,
                    provider="primeintellect",
                    gpu_type=offer.get("gpuType", "unknown"),
                    gpu_count=offer.get("gpuCount", 1),
                    vcpu=offer.get("vcpu", {}).get("defaultCount", 0),
                    memory_gb=offer.get("memory", {}).get("defaultCount", 0),
                    vram_gb=offer.get("gpuMemory", 0),
                    storage_gb=offer.get("disk", {}).get("defaultCount", 0),
                    price_per_hour=price_per_hour,
                    availability_zone=offer.get("dataCenter", "unknown"),
                    cloud_type=cloud_type,
                    cuda_version=cuda_version,  # Pass through filter
                    manufacturer=offer.get("provider"),  # Use provider as manufacturer proxy
                    underlying_provider=offer.get("provider"),  # Extract underlying provider (e.g., massedcompute, hyperstack)
                    raw_data=offer
                )
                offers.append(gpu_offer)
        
        return offers
        
    except Exception as e:
        logger.error(f"Failed to search Prime Intellect GPU offers: {e}")
        return []


def provision_instance(request: ProvisionRequest, ssh_startup_script: Optional[str] = None,
                      api_key: Optional[str] = None) -> Optional[GPUInstance]:
    """Provision a GPU instance on Prime Intellect"""
    # Build the pod definition
    pod_data = {
        "name": request.name or f"prime-{request.gpu_type or 'auto'}-{int(time.time())}",
        "gpuCount": request.gpu_count,
        "image": request.image or "ubuntu_22_cuda_12",  # Default to Ubuntu with CUDA
    }
    
    # Build provider specification first - use the actual provider from the offer
    # For Prime Intellect, we need to specify the underlying provider (e.g., "runpod", "hyperstack")
    provider_data = {
        "type": "runpod"  # Default to runpod for now
    }
    
    # Add GPU type if specified
    if request.gpu_type:
        # For Prime Intellect, gpu_type should be the cloudId from the offer
        # The offer ID format is: "prime-{cloudId}-{dataCenter}"
        # Extract the cloudId from the offer ID if it's in our format
        if request.gpu_type.startswith("prime-"):
            parts = request.gpu_type.split("-")
            if len(parts) >= 3:
                pod_data["cloudId"] = parts[1]
                # Try to extract datacenter if available
                if len(parts) >= 3:
                    pod_data["dataCenterId"] = parts[2]
        else:
            # Direct cloudId provided
            pod_data["cloudId"] = request.gpu_type
    
    # Add resource specifications if provided
    if request.container_disk_gb:
        pod_data["diskSize"] = request.container_disk_gb
    if request.memory_gb:
        pod_data["memory"] = request.memory_gb
    
    # Add environment variables if startup script provided
    env_vars = []
    if ssh_startup_script:
        env_vars.append({
            "key": "STARTUP_SCRIPT",
            "value": ssh_startup_script
        })
    
    # Add Jupyter password if provided
    if request.jupyter_password:
        pod_data["jupyterPassword"] = request.jupyter_password
    
    if env_vars:
        pod_data["envVars"] = env_vars
    
    # Provider data already built above
    
    # Determine security level
    if not request.spot_instance:
        pod_data["security"] = "secure_cloud"
    else:
        pod_data["security"] = "community_cloud"
    
    # Build the complete request body
    request_body = {
        "pod": pod_data,
        "provider": provider_data
    }
    
    try:
        data = _make_api_request("POST", "/pods/", data=request_body, api_key=api_key)
        
        if not data:
            logger.error("No pod returned from Prime Intellect deployment")
            return None
        
        # Parse the response and create GPUInstance
        return _parse_pod_to_instance(data, api_key=api_key)
        
    except Exception as e:
        logger.error(f"Failed to provision Prime Intellect instance: {e}")
        return None


def get_instance_details(instance_id: str, api_key: Optional[str] = None) -> Optional[GPUInstance]:
    """Get details of a specific instance"""
    try:
        data = _make_api_request("GET", f"/pods/{instance_id}", api_key=api_key)
        
        if not data:
            return None
        
        return _parse_pod_to_instance(data, api_key=api_key)
        
    except Exception as e:
        logger.error(f"Failed to get Prime Intellect instance details: {e}")
        return None


def list_instances(api_key: Optional[str] = None) -> List[GPUInstance]:
    """List all user's instances"""
    try:
        data = _make_api_request("GET", "/pods/", api_key=api_key)
        
        instances = []
        # API might return a list directly or wrapped in a data field
        pods = data if isinstance(data, list) else data.get("pods", [])
        
        for pod in pods:
            try:
                instance = _parse_pod_to_instance(pod, api_key=api_key)
                if instance:
                    instances.append(instance)
            except Exception as e:
                logger.warning(f"Failed to parse pod {pod.get('id', 'unknown')}: {e}")
                continue
        
        return instances
        
    except Exception as e:
        logger.error(f"Failed to list Prime Intellect instances: {e}")
        return []


def terminate_instance(instance_id: str, api_key: Optional[str] = None) -> bool:
    """Terminate a Prime Intellect instance"""
    try:
        _make_api_request("DELETE", f"/pods/{instance_id}", api_key=api_key)
        logger.info(f"Successfully terminated Prime Intellect instance {instance_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to terminate Prime Intellect instance: {e}")
        return False


def _parse_pod_to_instance(pod: Dict[str, Any], api_key: Optional[str] = None) -> GPUInstance:
    """Parse a pod dictionary into a GPUInstance"""
    
    # Map Prime Intellect statuses to our enum
    status_map = {
        "PROVISIONING": InstanceStatus.PENDING,
        "RUNNING": InstanceStatus.RUNNING,
        "STOPPED": InstanceStatus.STOPPED,
        "TERMINATED": InstanceStatus.TERMINATED,
        "FAILED": InstanceStatus.FAILED
    }
    status = status_map.get(pod.get("status", ""), InstanceStatus.PENDING)
    
    # Extract SSH connection info
    ssh_connection = pod.get("sshConnection", "")
    public_ip = pod.get("ip", "")
    ssh_port = 22
    ssh_username = "root"
    
    # Parse SSH connection string if available (format might be "ssh root@ip -p port")
    if ssh_connection and "@" in ssh_connection:
        try:
            # Extract components from SSH connection string
            parts = ssh_connection.split()
            for i, part in enumerate(parts):
                if "@" in part:
                    ssh_username, ip_part = part.split("@", 1)
                    public_ip = ip_part
                elif part == "-p" and i + 1 < len(parts):
                    ssh_port = int(parts[i + 1])
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse SSH connection string: {ssh_connection}")
    
    # Extract GPU information
    gpu_type = pod.get("gpuName", pod.get("gpuType", "unknown"))
    gpu_count = pod.get("gpuCount", 1)
    
    # Extract pricing
    price_per_hour = pod.get("priceHr", 0.0)
    
    return GPUInstance(
        id=pod["id"],
        provider="primeintellect",
        status=status,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        name=pod.get("name"),
        price_per_hour=price_per_hour,
        public_ip=public_ip,
        ssh_port=ssh_port,
        ssh_username=ssh_username,
        raw_data=pod,
        api_key=api_key  # Store API key for instance methods
    )


def get_user_balance(api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get user balance and spending information from Prime Intellect"""

    # Note: This endpoint might not exist in Prime Intellect API
    # Returning None for now, can be implemented when endpoint is available
    try:
        # Placeholder - actual endpoint TBD
        # data = _make_api_request("GET", "/user/balance", api_key=api_key)

        return {
            "provider": "primeintellect",
            "current_balance": None,
            "message": "Balance endpoint not yet implemented for Prime Intellect"
        }

    except Exception as e:
        logger.error(f"Failed to get Prime Intellect user balance: {e}")
        return None


def wait_for_ssh_ready(instance, timeout: int = 300) -> bool:
    """Prime Intellect-specific SSH waiting implementation"""
    # Tiger Style: Assert preconditions
    assert instance.provider == "primeintellect"
    assert instance.api_key
    assert timeout > 0

    # Wait for RUNNING status
    if not _wait_until_running(instance, timeout):
        return False

    # Wait for SSH details to be populated
    if not _wait_for_ssh_assignment(instance, time.time(), timeout):
        return False

    # Test connectivity
    return _test_ssh_connectivity(instance)


def _wait_until_running(instance, timeout: int) -> bool:
    """Wait for instance to reach RUNNING status"""
    start_time = time.time()

    logger.info(f"Waiting for instance {instance.id} to reach RUNNING...")

    while time.time() - start_time < timeout:
        fresh = get_instance_details(instance.id, api_key=instance.api_key)
        if not fresh:
            logger.error("Instance disappeared")
            return False

        if fresh.status.value == "running":
            instance.__dict__.update(fresh.__dict__)
            logger.info(f"Instance {instance.id} is RUNNING")
            return True
        elif fresh.status.value in ["failed", "terminated"]:
            logger.error(f"Instance terminal state: {fresh.status}")
            return False

        time.sleep(15)

    logger.error(f"Timeout waiting for RUNNING after {timeout}s")
    return False


def _wait_for_ssh_assignment(instance, start_time: float, timeout: int) -> bool:
    """Wait for SSH details to be assigned"""
    logger.info("Waiting for SSH details...")
    next_log_time = start_time + 30  # Log at 30s, 60s, 90s, ...

    while time.time() - start_time < timeout:
        fresh = get_instance_details(instance.id, api_key=instance.api_key)

        if fresh and fresh.public_ip and fresh.ssh_port:
            # Update instance with SSH details
            instance.public_ip = fresh.public_ip
            instance.ssh_port = fresh.ssh_port
            instance.ssh_username = fresh.ssh_username
            instance.status = fresh.status

            elapsed = int(time.time() - start_time)
            logger.info(f"SSH ready: {instance.public_ip}:{instance.ssh_port} (took {elapsed}s)")
            return True

        # Log progress every 30s
        current_time = time.time()
        if current_time >= next_log_time:
            elapsed = int(current_time - start_time)
            logger.debug(f"Waiting for SSH details - {elapsed}s")
            next_log_time += 30  # Schedule next log

        time.sleep(10)

    elapsed_min = int((time.time() - start_time) / 60)
    logger.error(f"Timeout waiting for SSH after {elapsed_min} min")
    return False


def _test_ssh_connectivity(instance) -> bool:
    """Test SSH connectivity with echo command"""
    logger.info("SSH details ready! Waiting 30s for SSH daemon...")
    time.sleep(30)

    try:
        result = instance.exec("echo 'ssh_ready'", timeout=30)
        if result.success and "ssh_ready" in result.stdout:
            logger.info("SSH connectivity confirmed!")
            return True
        else:
            logger.warning(f"SSH test failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"SSH connection error: {e}")
        return False


def get_fresh_instance(instance_id: str, api_key: str):
    """Alias for get_instance_details (ProviderProtocol requirement)"""
    return get_instance_details(instance_id, api_key=api_key)