"""
Vast.ai provider implementation
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from ..types import CloudType, GPUInstance, GPUOffer, InstanceStatus, ProvisionRequest
from shared.retry import retry

logger = logging.getLogger(__name__)

VAST_API_BASE_URL = "https://console.vast.ai/api/v0"


def _map_status(actual_status: str) -> InstanceStatus:
    """Map Vast.ai status string to our InstanceStatus enum.

    Why separate function: Status mapping logic used by multiple functions,
    extracting ensures consistency and makes updates easier.
    """
    status_map = {
        "running": InstanceStatus.RUNNING,
        "loading": InstanceStatus.PENDING,
        "exited": InstanceStatus.STOPPED,
        "offline": InstanceStatus.FAILED
    }
    # Default to PENDING for unknown statuses
    return status_map.get(actual_status, InstanceStatus.PENDING)


@retry(max_attempts=3, delay=1, backoff=2, exceptions=(requests.RequestException, requests.Timeout))
def _make_api_request(method: str, endpoint: str, data: Optional[Dict] = None,
                     params: Optional[Dict] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Make a REST API request to Vast.ai with automatic retries.

    Retries up to 3 times with exponential backoff (1s, 2s, 4s) on network errors.

    Tiger Style: Assert preconditions at function entry.
    """
    if not api_key:
        raise ValueError("Vast.ai API key is required but was not provided")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    url = f"{VAST_API_BASE_URL}{endpoint}"
    logger.debug("Vast.ai API request %s %s (api key ...%s)",
                 method, url, api_key[-4:] if api_key else "none")

    try:
        response = requests.request(
            method=method,
            url=url,
            json=data,
            params=params,
            headers=headers,
            timeout=(10, 30),  # (connect_timeout, read_timeout)
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        logger.error("Vast.ai API request timed out")
        raise
    except requests.RequestException as exc:
        logger.error(f"Vast.ai API request failed: {exc}")
        raise

    # Handle empty responses (e.g., DELETE operations)
    if response.status_code == 204 or not response.content:
        return {}

    return response.json()


def search_gpu_offers(cuda_version: Optional[str] = None, manufacturer: Optional[str] = None,
                      memory_gb: Optional[int] = None, container_disk_gb: Optional[int] = None,
                      gpu_count: int = 1, min_reliability: float = 0.95,
                      api_key: Optional[str] = None) -> List[GPUOffer]:
    """Search for available GPU offers on Vast.ai

    Args:
        cuda_version: CUDA version requirement (not used by Vast.ai API directly)
        manufacturer: GPU manufacturer filter (NVIDIA or AMD)
        memory_gb: Minimum system RAM in GB
        container_disk_gb: Minimum disk space in GB
        gpu_count: Number of GPUs required (default: 1)
        min_reliability: Minimum reliability score 0-1 (default: 0.95)
                        Set to 0 to disable reliability filtering
        api_key: Vast.ai API key

    Returns:
        List of GPUOffer objects sorted by price ascending

    Tiger Style: All parameters have explicit types and defaults
    """
    # Tiger Style: Assert preconditions on API inputs
    assert api_key is not None, "API key required"
    assert gpu_count > 0, f"gpu_count must be positive, got {gpu_count}"

    # Warn if reliability filtering is low
    if min_reliability > 0 and min_reliability < 0.9:
        logger.warning(f"Low reliability threshold {min_reliability} may result in unreliable hosts")

    # Build base query with sensible defaults (explicit, not implicit)
    query = {
        "verified": {"eq": True},      # Only verified hosts
        "external": {"eq": False},     # Exclude external hosts
        "rentable": {"eq": True},      # Only rentable machines
        "rented": {"eq": False},       # Only available machines
        "order": [["dph_total", "asc"]],  # Sort by price ascending
        "type": "on-demand",           # Default to on-demand (not spot)
        "allocated_storage": 10,       # Minimum storage allocation
    }

    # Add GPU count filter
    query["num_gpus"] = {"eq": gpu_count}

    # Add reliability filter if specified
    if min_reliability > 0:
        query["reliability2"] = {"gte": min_reliability}

    # Add memory filter if specified (convert GB to MB)
    if memory_gb:
        query["cpu_ram"] = {"gte": memory_gb * 1000}

    # Add disk space filter if specified
    if container_disk_gb:
        query["disk_space"] = {"gte": container_disk_gb}

    # Casey Muratori: Provide BOTH high-level AND low-level access
    # Query Vast.ai API (broad search), then filter in Python (precise control)
    response = _make_api_request("POST", "/bundles/", data=query, api_key=api_key)
    offers_raw = response.get("offers", [])

    # Post-process filtering (decoupled from API call)
    offers = []
    for offer_data in offers_raw:
        gpu_name = offer_data.get("gpu_name", "")

        # Tiger Style: Assert data validity from external API
        assert isinstance(gpu_name, str), f"gpu_name must be string, got {type(gpu_name)}"

        # Apply manufacturer filter (if specified)
        if manufacturer:
            mfr_lower = manufacturer.lower()
            gpu_name_lower = gpu_name.lower()

            # Vast.ai supports NVIDIA and AMD (as of May 2024)
            # NVIDIA: Most GPUs (RTX, GTX, A100, H100, V100, etc.)
            is_nvidia = any(x in gpu_name_lower for x in [
                "rtx", "gtx", "quadro", "tesla", "a100", "h100", "v100",
                "a6000", "a5000", "l40", "l4", "a10", "a30", "a40"
            ])

            # AMD: Radeon and Instinct lines
            is_amd = any(x in gpu_name_lower for x in [
                "radeon", "instinct", "rx ", "mi100", "mi200", "mi300"
            ])

            # Filter based on manufacturer
            if mfr_lower == "nvidia" and not is_nvidia:
                continue
            elif mfr_lower == "amd" and not is_amd:
                continue
            elif mfr_lower not in ["nvidia", "amd"]:
                # Intel, other manufacturers not yet supported by Vast.ai
                continue

        # Extract offer details
        offer_id = offer_data.get("id")
        num_gpus = offer_data.get("num_gpus", 1)
        dph_total = offer_data.get("dph_total", 0)

        # Tiger Style: Assert critical data
        assert offer_id is not None, f"Offer missing ID: {offer_data}"
        assert isinstance(offer_id, int), f"Offer ID must be int, got {type(offer_id)}"
        assert num_gpus > 0, f"Invalid num_gpus: {num_gpus}"
        assert dph_total > 0, f"Invalid price: {dph_total}"

        # Calculate per-GPU price (Vast.ai returns total price for all GPUs)
        price_per_gpu = dph_total / num_gpus

        # Determine cloud type based on offer type
        offer_type = query.get("type", "on-demand")
        cloud_type = CloudType.SECURE if offer_type == "on-demand" else CloudType.COMMUNITY

        # Determine manufacturer from GPU name
        gpu_name_lower = gpu_name.lower()
        if any(x in gpu_name_lower for x in ["rtx", "gtx", "quadro", "tesla", "a100", "h100", "v100", "a6000", "a5000", "l40", "l4", "a10", "a30", "a40"]):
            gpu_manufacturer = "NVIDIA"
        elif any(x in gpu_name_lower for x in ["radeon", "instinct", "rx ", "mi100", "mi200", "mi300"]):
            gpu_manufacturer = "AMD"
        else:
            gpu_manufacturer = "unknown"

        # Create GPUOffer with vast-{offer_id} format
        offer = GPUOffer(
            id=f"vast-{offer_id}",
            provider="vast",
            gpu_type=gpu_name,
            gpu_count=num_gpus,
            vcpu=offer_data.get("cpu_cores_effective", 0),
            memory_gb=offer_data.get("cpu_ram", 0) / 1000,  # MB to GB
            storage_gb=offer_data.get("disk_space", 0),
            price_per_hour=price_per_gpu,
            availability_zone=offer_data.get("geolocation"),
            vram_gb=offer_data.get("gpu_ram", 0) / 1000 if offer_data.get("gpu_ram") else None,  # MB to GB
            cloud_type=cloud_type,
            manufacturer=gpu_manufacturer,
            cuda_version=str(offer_data.get("cuda_max_good")) if offer_data.get("cuda_max_good") else None,
            driver_version=offer_data.get("driver_version"),
            raw_data=offer_data  # Store full offer data for provisioning
        )

        offers.append(offer)

    return offers


def provision_instance(request: ProvisionRequest, ssh_startup_script: Optional[str] = None,
                      api_key: Optional[str] = None) -> Optional[GPUInstance]:
    """Provision a GPU instance on Vast.ai

    Args:
        request: ProvisionRequest containing instance configuration
                 request.image: Docker image (default from ProvisionRequest is fine)
                               Vast.ai supports standard Docker Hub images
                 request.gpu_type: Offer ID in format "vast-{offer_id}"
        ssh_startup_script: Shell script to run on startup (optional)
                           Maps to Vast.ai's "onstart" field
        api_key: Vast.ai API key

    Returns:
        GPUInstance if successful, None if provisioning failed

    Tiger Style: Returns None for operating errors (offer unavailable),
                raises ValueError for programmer errors (invalid params)
    """
    # Extract offer_id from request.gpu_type
    # Format: vast-{offer_id}
    if not request.gpu_type or not request.gpu_type.startswith("vast-"):
        logger.error(f"Invalid gpu_type format: {request.gpu_type}")
        raise ValueError(f"gpu_type must be in format 'vast-{{offer_id}}', got {request.gpu_type}")

    try:
        offer_id = int(request.gpu_type.split("-", 1)[1])
    except (IndexError, ValueError) as e:
        logger.error(f"Failed to parse offer_id from gpu_type '{request.gpu_type}': {e}")
        raise ValueError(f"Invalid gpu_type format: {request.gpu_type}")

    # Get price from raw_data (stored in GPUOffer during search)
    # The raw_data should contain the original Vast.ai API response
    raw_data = request.raw_data if hasattr(request, 'raw_data') else None

    # If raw_data not available, we need to extract from stored offer data
    # For now, we'll use a reasonable max price - this should be improved
    # by passing price through ProvisionRequest
    price = raw_data.get("dph_total") if raw_data else 1.0

    # Build request body
    request_body = {
        "client_id": "me",
        "image": request.image,
        "price": price,  # Use exact price from offer
        "disk": request.container_disk_gb or 50,  # Default 50GB matches RunPod
        "label": request.name or f"vast-{offer_id}-{int(time.time())}",
        "onstart": ssh_startup_script or "",  # Startup script content
        "runtype": "ssh",  # Required for SSH access
        "image_login": "",
        "python_utf8": False,
        "lang_utf8": False,
        "use_jupyter_lab": False,
        "jupyter_dir": "/",
        "force": False,
        "cancel_unavail": False
    }

    try:
        data = _make_api_request("PUT", f"/asks/{offer_id}/", data=request_body, api_key=api_key)

        # Tiger Style: Assert response validity
        assert "success" in data, f"Vast.ai response missing 'success' field: {data}"

        if not data["success"]:
            # Expected operating error (offer unavailable, etc.)
            logger.warning(f"Vast.ai provisioning failed: {data.get('msg', 'Unknown error')}")
            return None

        assert "new_contract" in data, f"Successful response missing 'new_contract': {data}"
        instance_id = data["new_contract"]

        # Tiger Style: Assert postcondition
        assert isinstance(instance_id, int), f"instance_id must be int, got {type(instance_id)}"

        # Convert instance_id to string for consistency with other providers
        instance_id_str = str(instance_id)

        # Return GPUInstance with initial state (status will be updated by polling)
        return GPUInstance(
            id=instance_id_str,
            provider="vast",
            status=InstanceStatus.PENDING,
            gpu_type=request.gpu_type,
            gpu_count=request.gpu_count or 1,
            price_per_hour=price / (request.gpu_count or 1),  # Convert to per-GPU price
            name=request.name,
            api_key=api_key,  # Store for future API calls
            raw_data={"offer_id": offer_id, "provisioning_response": data}
        )

    except ValueError as e:
        # Programmer error - invalid parameters
        logger.error(f"Invalid parameters for Vast.ai provisioning: {e}")
        raise  # Re-raise programmer errors

    except Exception as e:
        # Unexpected error - treat as operating error
        logger.error(f"Unexpected error provisioning Vast.ai instance: {e}", exc_info=True)
        return None


def get_instance_details(instance_id: str, api_key: Optional[str] = None) -> Optional[GPUInstance]:
    """Get details of a specific Vast.ai instance

    Args:
        instance_id: Vast.ai instance ID (numeric string)
        api_key: Vast.ai API key

    Returns:
        GPUInstance if found, None otherwise
    """
    if not api_key:
        raise ValueError("Vast.ai API key is required")

    try:
        # List all instances and find the matching one
        data = _make_api_request("GET", "/instances/", params={"owner": "me"}, api_key=api_key)

        instances = data.get("instances", [])
        assert isinstance(instances, list), f"instances must be list, got {type(instances)}"

        # Find instance by ID
        for inst in instances:
            if str(inst.get("id")) == str(instance_id):
                return _parse_instance(inst, api_key)

        # Instance not found
        logger.warning(f"Vast.ai instance {instance_id} not found")
        return None

    except Exception as e:
        logger.error(f"Error getting Vast.ai instance details: {e}", exc_info=True)
        return None


def _parse_instance(inst: Dict[str, Any], api_key: str) -> GPUInstance:
    """Parse Vast.ai instance data into GPUInstance

    Helper function to convert raw API response to GPUInstance.
    Separated for reuse in get_instance_details and list_instances.
    """
    instance_id = inst.get("id")
    actual_status = inst.get("actual_status", "loading")

    # Extract SSH details
    ssh_host = inst.get("ssh_host")
    ssh_port = inst.get("ssh_port")

    # Vast.ai ALWAYS uses root as SSH username
    ssh_username = "root"

    # Map status
    status = _map_status(actual_status)

    # Create GPUInstance
    return GPUInstance(
        id=str(instance_id),
        provider="vast",
        status=status,
        gpu_type=inst.get("gpu_name", "unknown"),
        gpu_count=inst.get("num_gpus", 1),
        price_per_hour=inst.get("dph_total", 0) / inst.get("num_gpus", 1),
        name=inst.get("label"),
        public_ip=ssh_host,
        ssh_port=ssh_port,
        ssh_username=ssh_username,
        api_key=api_key,
        raw_data=inst
    )


def list_instances(api_key: Optional[str] = None) -> List[GPUInstance]:
    """List all user's Vast.ai instances

    Args:
        api_key: Vast.ai API key

    Returns:
        List of GPUInstance objects
    """
    if not api_key:
        raise ValueError("Vast.ai API key is required")

    try:
        data = _make_api_request("GET", "/instances/", params={"owner": "me"}, api_key=api_key)

        instances = data.get("instances", [])
        assert isinstance(instances, list), f"instances must be list, got {type(instances)}"

        return [_parse_instance(inst, api_key) for inst in instances]

    except Exception as e:
        logger.error(f"Error listing Vast.ai instances: {e}", exc_info=True)
        return []


def terminate_instance(instance_id: str, api_key: Optional[str] = None) -> bool:
    """Terminate a Vast.ai instance

    Args:
        instance_id: Vast.ai instance ID (numeric string)
        api_key: Vast.ai API key

    Returns:
        True if termination successful, False otherwise
    """
    if not api_key:
        raise ValueError("Vast.ai API key is required")

    try:
        data = _make_api_request("DELETE", f"/instances/{instance_id}/", api_key=api_key)

        # Check success field
        if data.get("success"):
            logger.info(f"Vast.ai instance {instance_id} terminated: {data.get('msg')}")
            return True
        else:
            logger.error(f"Failed to terminate Vast.ai instance {instance_id}: {data.get('msg')}")
            return False

    except Exception as e:
        logger.error(f"Error terminating Vast.ai instance {instance_id}: {e}", exc_info=True)
        return False


def wait_for_ssh_ready(instance, timeout: int = 300) -> bool:
    """Wait for Vast.ai instance SSH to be ready

    Args:
        instance: GPUInstance object
        timeout: Maximum seconds to wait (default: 300 = 5 minutes)

    Returns:
        True if SSH is ready, False if timeout or error

    Implementation follows RunPod pattern:
    1. Poll until actual_status == "running"
    2. Poll until SSH details (ssh_host, ssh_port) are populated
    3. Test SSH connectivity with simple command (echo 'ssh_ready')

    Note: Does NOT wait for onstart script completion - dependency installation
    happens separately via Bifrost after SSH is ready.
    """
    start_time = time.time()

    # Step 1: Wait for RUNNING status
    logger.info(f"Waiting for Vast.ai instance {instance.id} to reach RUNNING status...")
    while True:
        if time.time() - start_time > timeout:
            logger.error(f"Timeout waiting for instance {instance.id} to reach RUNNING status")
            return False

        # Refresh instance details
        fresh_instance = get_instance_details(instance.id, api_key=instance.api_key)
        if not fresh_instance:
            logger.error(f"Failed to get instance details for {instance.id}")
            return False

        if fresh_instance.status == InstanceStatus.RUNNING:
            logger.info(f"Instance {instance.id} is now RUNNING")
            instance.status = fresh_instance.status
            instance.public_ip = fresh_instance.public_ip
            instance.ssh_port = fresh_instance.ssh_port
            instance.ssh_username = fresh_instance.ssh_username
            break
        elif fresh_instance.status in [InstanceStatus.FAILED, InstanceStatus.STOPPED, InstanceStatus.TERMINATED]:
            logger.error(f"Instance {instance.id} entered terminal state: {fresh_instance.status}")
            return False

        logger.debug(f"Instance {instance.id} status: {fresh_instance.status}, waiting...")
        time.sleep(5)

    # Step 2: Wait for SSH details to be populated
    logger.info(f"Waiting for SSH details to be populated for instance {instance.id}...")
    while True:
        if time.time() - start_time > timeout:
            logger.error(f"Timeout waiting for SSH details for instance {instance.id}")
            return False

        # Refresh instance details
        fresh_instance = get_instance_details(instance.id, api_key=instance.api_key)
        if not fresh_instance:
            logger.error(f"Failed to get instance details for {instance.id}")
            return False

        if fresh_instance.public_ip and fresh_instance.ssh_port:
            logger.info(f"SSH details ready: {fresh_instance.public_ip}:{fresh_instance.ssh_port}")
            instance.public_ip = fresh_instance.public_ip
            instance.ssh_port = fresh_instance.ssh_port
            instance.ssh_username = fresh_instance.ssh_username
            break

        logger.debug(f"SSH details not yet available, waiting...")
        time.sleep(5)

    # Step 3: Test SSH connectivity
    # Wait 30s for SSH daemon to be ready
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
    """Alias for get_instance_details (ProviderProtocol requirement)

    Why alias: Some parts of broker use get_fresh_instance, others use
    get_instance_details. Providing both ensures compatibility.
    """
    return get_instance_details(instance_id, api_key=api_key)
