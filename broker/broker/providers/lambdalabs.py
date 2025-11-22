"""
Lambda Labs provider implementation
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from ..types import CloudType, GPUInstance, GPUOffer, InstanceStatus, ProvisionRequest
from shared.retry import retry

logger = logging.getLogger(__name__)

LAMBDA_API_BASE_URL = "https://cloud.lambdalabs.com/api/v1"


@retry(max_attempts=3, delay=1, backoff=2, exceptions=(requests.RequestException, requests.Timeout))
def _make_api_request(method: str, endpoint: str, data: Optional[Dict] = None,
                     params: Optional[Dict] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Make a REST API request to Lambda Labs API with automatic retries.

    Retries up to 3 times with exponential backoff (1s, 2s, 4s) on network errors.

    Args:
        method: HTTP method (GET, POST, DELETE, etc.)
        endpoint: API endpoint (e.g., "/instance-types")
        data: Optional request body data
        params: Optional query parameters
        api_key: Lambda Labs API key (required)
    """
    if not api_key:
        raise ValueError("Lambda Labs API key is required but was not provided")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    url = f"{LAMBDA_API_BASE_URL}{endpoint}"
    logger.debug("Lambda Labs API request %s %s (api key ...%s)", method, url, api_key[-4:] if api_key else "none")

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
        logger.error("Lambda Labs API request timed out")
        raise
    except requests.RequestException as exc:
        logger.error(f"Lambda Labs API request failed: {exc}")
        raise

    # Handle empty responses (e.g., DELETE operations)
    if response.status_code == 204 or not response.content:
        return {}

    return response.json()


def search_gpu_offers(cuda_version: Optional[str] = None, manufacturer: Optional[str] = None,
                      memory_gb: Optional[int] = None, container_disk_gb: Optional[int] = None,
                      gpu_count: int = 1, api_key: Optional[str] = None) -> List[GPUOffer]:
    """Search for available GPU offers on Lambda Labs with optional filtering"""

    try:
        response = _make_api_request("GET", "/instance-types", api_key=api_key)
        offers = []

        # Lambda Labs API returns: {"data": {"gpu_1x_a100": {...}, ...}}
        instance_types = response.get("data", {})

        for instance_type_name, type_data in instance_types.items():
            instance_type = type_data.get("instance_type", {})
            specs = instance_type.get("specs", {})
            regions = type_data.get("regions_with_capacity_available", [])

            # Skip if no capacity available
            if not regions:
                continue

            # Extract GPU info from name (e.g., "gpu_8x_h100_sxm5" -> 8 GPUs, H100)
            gpu_count_in_type = specs.get("gpus", 1)

            # Filter by GPU count
            if gpu_count_in_type != gpu_count:
                continue

            # Extract GPU model from description (e.g., "8x H100 (80 GB SXM5)" -> "H100")
            gpu_description = instance_type.get("gpu_description", "")
            gpu_type = gpu_description.split("(")[0].strip() if gpu_description else "unknown"

            # Extract VRAM from GPU description (e.g., "H100 (80 GB SXM5)" -> 80)
            vram_gb = 0
            if "(" in gpu_description and "GB" in gpu_description:
                try:
                    vram_str = gpu_description.split("(")[1].split("GB")[0].strip()
                    vram_gb = int(vram_str)
                except (ValueError, IndexError):
                    pass

            # Filter by manufacturer (Lambda Labs is all NVIDIA)
            if manufacturer and manufacturer.lower() != "nvidia":
                continue

            # Filter by memory
            memory_gib = specs.get("memory_gib", 0)
            if memory_gb and memory_gib < memory_gb:
                continue

            # Filter by storage
            storage_gib = specs.get("storage_gib", 0)
            if container_disk_gb and storage_gib < container_disk_gb:
                continue

            # Convert price from cents to dollars
            price_cents = instance_type.get("price_cents_per_hour", 0)
            price_per_hour = price_cents / 100.0

            # Normalize to per-GPU pricing
            price_per_gpu = price_per_hour / gpu_count_in_type if gpu_count_in_type > 0 else price_per_hour

            # Create offers for each region with availability
            for region in regions:
                region_name = region.get("name", "unknown")
                region_desc = region.get("description", "")

                # Create unique offer ID
                offer_id = f"lambda-{instance_type_name}-{region_name}"

                gpu_offer = GPUOffer(
                    id=offer_id,
                    provider="lambdalabs",
                    gpu_type=gpu_type,
                    gpu_count=gpu_count_in_type,
                    vcpu=specs.get("vcpus", 0),
                    memory_gb=memory_gib,
                    vram_gb=vram_gb,
                    storage_gb=storage_gib,
                    price_per_hour=price_per_gpu,
                    availability_zone=region_desc or region_name,
                    cloud_type=CloudType.SECURE,  # Lambda Labs is all secure cloud
                    spot=False,  # Lambda Labs doesn't have spot instances
                    cuda_version=cuda_version,  # Pass through filter
                    manufacturer="nvidia",
                    raw_data={
                        "instance_type_name": instance_type_name,
                        "region_name": region_name,
                        **type_data
                    }
                )
                offers.append(gpu_offer)

        return offers

    except Exception as e:
        logger.error(f"Failed to search Lambda Labs GPU offers: {e}")
        return []


def provision_instance(request: ProvisionRequest, ssh_startup_script: Optional[str] = None,
                      api_key: Optional[str] = None) -> Optional[GPUInstance]:
    """Provision a GPU instance on Lambda Labs"""

    # Parse instance type and region from offer ID
    # Format: "lambda-{instance_type_name}-{region_name}"
    instance_type_name = None
    region_name = None

    if request.gpu_type and request.gpu_type.startswith("lambda-"):
        parts = request.gpu_type.split("-", 2)  # Split into max 3 parts
        if len(parts) >= 3:
            instance_type_name = parts[1]
            region_name = parts[2]

    if not instance_type_name or not region_name:
        logger.error(f"Invalid offer ID format: {request.gpu_type}. Expected: lambda-<instance_type>-<region>")
        return None

    # Get SSH key ID from Lambda Labs
    # We need to use the SSH key that's already registered with Lambda Labs
    ssh_key_names = []
    try:
        ssh_keys_response = _make_api_request("GET", "/ssh-keys", api_key=api_key)
        ssh_keys = ssh_keys_response.get("data", [])
        if ssh_keys:
            # Use the first available SSH key
            ssh_key_names = [ssh_keys[0]["name"]]
            logger.info(f"using ssh key: {ssh_key_names[0]}")
        else:
            logger.warning("No SSH keys found in Lambda Labs account. Instance may not be accessible.")
    except Exception as e:
        logger.warning(f"Failed to fetch SSH keys: {e}")

    # Build launch request
    launch_data = {
        "region_name": region_name,
        "instance_type_name": instance_type_name,
        "ssh_key_names": ssh_key_names,
        "quantity": 1  # Lambda Labs only supports launching 1 instance at a time
    }

    # Add file system names if specified (optional)
    if hasattr(request, 'file_system_names') and request.file_system_names:
        launch_data["file_system_names"] = request.file_system_names

    # Add instance name if specified (Lambda Labs may not support this directly)
    # We'll store it in the raw_data for reference
    instance_name = request.name or f"lambda-{instance_type_name}-{int(time.time())}"

    try:
        response = _make_api_request("POST", "/instance-operations/launch", data=launch_data, api_key=api_key)

        if not response or "data" not in response:
            logger.error("No instance data returned from Lambda Labs launch")
            return None

        # Extract instance IDs from response
        # Lambda Labs returns: {"data": {"instance_ids": ["abc123"]}}
        data = response.get("data", {})
        instance_ids = data.get("instance_ids", [])

        if not instance_ids:
            logger.error("No instance IDs returned from Lambda Labs launch")
            return None

        instance_id = instance_ids[0]
        logger.info(f"lambda labs instance launched: {instance_id}")

        # Immediately fetch instance details to get full info
        # Wait a moment for instance to be queryable
        time.sleep(2)
        instance = get_instance_details(instance_id, api_key=api_key)

        if instance:
            # Update name
            instance.name = instance_name
            return instance

        # Fallback: create minimal instance object if we can't fetch details yet
        return GPUInstance(
            id=instance_id,
            provider="lambdalabs",
            status=InstanceStatus.PENDING,
            gpu_type=instance_type_name,
            gpu_count=request.gpu_count or 1,
            name=instance_name,
            price_per_hour=0.0,  # Will be filled in when details are fetched
            raw_data=response,
            api_key=api_key
        )

    except Exception as e:
        logger.error(f"Failed to provision Lambda Labs instance: {e}")
        return None


def get_instance_details(instance_id: str, api_key: Optional[str] = None) -> Optional[GPUInstance]:
    """Get details of a specific instance"""
    try:
        response = _make_api_request("GET", f"/instances/{instance_id}", api_key=api_key)

        if not response or "data" not in response:
            return None

        instance_data = response.get("data", {})
        return _parse_instance_to_gpu_instance(instance_data, api_key=api_key)

    except Exception as e:
        logger.error(f"Failed to get Lambda Labs instance details: {e}")
        return None


def list_instances(api_key: Optional[str] = None) -> List[GPUInstance]:
    """List all user's instances"""
    try:
        response = _make_api_request("GET", "/instances", api_key=api_key)

        instances = []
        # Lambda Labs returns: {"data": [instance1, instance2, ...]}
        instance_list = response.get("data", [])

        for instance_data in instance_list:
            try:
                instance = _parse_instance_to_gpu_instance(instance_data, api_key=api_key)
                if instance:
                    instances.append(instance)
            except Exception as e:
                logger.warning(f"Failed to parse instance {instance_data.get('id', 'unknown')}: {e}")
                continue

        return instances

    except Exception as e:
        logger.error(f"Failed to list Lambda Labs instances: {e}")
        return []


def terminate_instance(instance_id: str, api_key: Optional[str] = None) -> bool:
    """Terminate a Lambda Labs instance"""
    try:
        # Lambda Labs terminate endpoint takes a JSON body with instance_ids array
        terminate_data = {
            "instance_ids": [instance_id]
        }
        _make_api_request("POST", "/instance-operations/terminate", data=terminate_data, api_key=api_key)
        logger.info(f"successfully terminated lambda labs instance {instance_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to terminate Lambda Labs instance: {e}")
        return False


def _parse_instance_to_gpu_instance(instance_data: Dict[str, Any], api_key: Optional[str] = None) -> GPUInstance:
    """Parse a Lambda Labs instance dictionary into a GPUInstance"""

    # Map Lambda Labs statuses to our enum
    # Lambda Labs statuses: "active", "booting", "unhealthy", "terminated"
    status_map = {
        "booting": InstanceStatus.PENDING,
        "active": InstanceStatus.RUNNING,
        "unhealthy": InstanceStatus.FAILED,
        "terminated": InstanceStatus.TERMINATED,
    }
    status_str = instance_data.get("status", "booting")
    status = status_map.get(status_str, InstanceStatus.PENDING)

    # Extract instance info
    instance_id = instance_data.get("id", "")
    instance_type = instance_data.get("instance_type", {})

    # Get instance type name (e.g., "gpu_1x_h100_sxm5")
    instance_type_name = instance_type.get("name", "unknown")

    # Extract GPU info from description
    gpu_description = instance_type.get("gpu_description", "")
    gpu_type = gpu_description.split("(")[0].strip() if gpu_description else "unknown"

    # Extract specs
    specs = instance_type.get("specs", {})
    gpu_count = specs.get("gpus", 1)

    # Extract IP address
    public_ip = instance_data.get("ip", "")

    # Lambda Labs uses standard SSH (port 22, username ubuntu)
    ssh_port = 22
    ssh_username = "ubuntu"

    # Extract hostname (may be different from IP)
    hostname = instance_data.get("hostname", "")

    # Use hostname if IP not available yet
    if not public_ip and hostname:
        public_ip = hostname

    # Extract pricing
    price_cents = instance_type.get("price_cents_per_hour", 0)
    price_per_hour = price_cents / 100.0

    # Extract region
    region = instance_data.get("region", {})
    region_name = region.get("name", "unknown")

    # Extract name (Lambda Labs might not have a name field)
    name = instance_data.get("name", f"lambda-{instance_type_name}-{instance_id[:8]}")

    return GPUInstance(
        id=instance_id,
        provider="lambdalabs",
        status=status,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        name=name,
        price_per_hour=price_per_hour,
        public_ip=public_ip,
        ssh_port=ssh_port,
        ssh_username=ssh_username,
        raw_data=instance_data,
        api_key=api_key  # Store API key for instance methods
    )


def get_user_balance(api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get user balance and spending information from Lambda Labs"""

    # Note: Lambda Labs doesn't appear to have a balance endpoint in their public API
    # Returning a placeholder response
    try:
        return {
            "provider": "lambdalabs",
            "current_balance": None,
            "message": "Balance endpoint not available for Lambda Labs"
        }

    except Exception as e:
        logger.error(f"Failed to get Lambda Labs user balance: {e}")
        return None


def wait_for_ssh_ready(instance, timeout: int = 900) -> bool:
    """Lambda Labs-specific SSH waiting implementation"""
    # Assert preconditions
    assert instance.provider == "lambdalabs"
    assert instance.api_key
    assert timeout > 0

    # Wait for RUNNING status
    if not _wait_until_active(instance, timeout):
        return False

    # Wait for SSH details to be populated
    if not _wait_for_ssh_assignment(instance, time.time(), timeout):
        return False

    # Lambda Labs instances are ready for SSH once they're active and have an IP
    # We can't test SSH connectivity here because we don't have the SSH key path
    # (it's only available at the client level, not in the provider layer)
    # Give SSH daemon a moment to fully start
    logger.debug("instance is active with ssh details assigned. waiting 10s for ssh daemon...")
    time.sleep(10)
    logger.debug("ssh should be ready!")
    return True


def _wait_until_active(instance, timeout: int) -> bool:
    """Wait for instance to reach active status"""
    start_time = time.time()

    logger.debug(f"waiting for instance {instance.id} to reach active status...")

    while time.time() - start_time < timeout:
        fresh = get_instance_details(instance.id, api_key=instance.api_key)
        if not fresh:
            logger.error("Instance disappeared")
            return False

        if fresh.status.value == "running":
            instance.__dict__.update(fresh.__dict__)
            elapsed = int(time.time() - start_time)
            logger.debug(f"instance {instance.id} is active (took {elapsed}s)")
            return True
        elif fresh.status.value in ["failed", "terminated"]:
            logger.error(f"Instance terminal state: {fresh.status}")
            return False

        time.sleep(10)

    logger.error(f"Timeout waiting for active status after {timeout}s")
    return False


def _wait_for_ssh_assignment(instance, start_time: float, timeout: int) -> bool:
    """Wait for SSH details to be assigned"""
    logger.debug("waiting for ssh details...")
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
            logger.debug(f"ssh ready: {instance.public_ip}:{instance.ssh_port} (took {elapsed}s)")
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
    logger.debug("ssh details ready! waiting 15s for ssh daemon...")
    time.sleep(15)

    try:
        result = instance.exec("echo 'ssh_ready'", timeout=30)
        if result.success and "ssh_ready" in result.stdout:
            logger.debug("ssh connectivity confirmed!")
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
