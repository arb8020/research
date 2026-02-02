"""
DigitalOcean AMD Developer Cloud GPU Droplet provider implementation.

Uses the AMD-specific API endpoint (api-amd.digitalocean.com) which provides
access to AMD MI300X GPUs at $1.99/hr.
"""

import logging
import time
from typing import Any

import httpx
import trio
from infra_utils.retry import async_retry

from ..types import CloudType, GPUInstance, GPUOffer, InstanceStatus, ProvisionRequest

logger = logging.getLogger(__name__)

# AMD Developer Cloud uses a different API endpoint
DIGITALOCEAN_AMD_API_BASE_URL = "https://api-amd.digitalocean.com/v2"

# GPU model name normalization mapping
GPU_MODEL_NAMES = {
    "amd_mi300x": "MI300X",
    "amd_mi325x": "MI325X",
}


@async_retry(
    max_attempts=3, delay=1, backoff=2, exceptions=(httpx.HTTPError, httpx.TimeoutException)
)
async def _make_api_request(
    method: str,
    endpoint: str,
    data: dict | None = None,
    params: dict | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Make a REST API request to DigitalOcean AMD API with automatic retries.

    Retries up to 3 times with exponential backoff (1s, 2s, 4s) on network errors.

    Args:
        method: HTTP method (GET, POST, DELETE, etc.)
        endpoint: API endpoint (e.g., "/sizes")
        data: Optional request body data
        params: Optional query parameters
        api_key: DigitalOcean AMD API key (required)
    """
    if not api_key:
        raise ValueError("DigitalOcean AMD API key is required but was not provided")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    url = f"{DIGITALOCEAN_AMD_API_BASE_URL}{endpoint}"
    logger.debug(
        "DigitalOcean AMD API request %s %s (api key ...%s)",
        method,
        url,
        api_key[-4:] if api_key else "none",
    )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10)) as client:
            response = await client.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=headers,
            )
            response.raise_for_status()
    except httpx.TimeoutException:
        logger.exception("DigitalOcean AMD API request timed out")
        raise
    except httpx.HTTPError as exc:
        logger.exception(f"DigitalOcean AMD API request failed: {exc}")
        raise

    # Handle empty responses (e.g., DELETE operations return 204)
    if response.status_code == 204 or not response.content:
        return {}

    return response.json()


async def search_gpu_offers(
    cuda_version: str | None = None,
    manufacturer: str | None = None,
    memory_gb: int | None = None,
    container_disk_gb: int | None = None,
    gpu_count: int = 1,
    api_key: str | None = None,
) -> list[GPUOffer]:
    """Search for available AMD GPU offers on DigitalOcean AMD Developer Cloud.

    Args:
        cuda_version: Not used (AMD GPUs use ROCm, not CUDA)
        manufacturer: Filter by GPU manufacturer (only "amd" supported here)
        memory_gb: Minimum system memory in GB
        container_disk_gb: Minimum disk size in GB
        gpu_count: Number of GPUs required
        api_key: DigitalOcean AMD API key

    Returns:
        List of available GPU offers
    """
    # Skip if explicitly requesting non-AMD GPUs
    if manufacturer and manufacturer.lower() != "amd":
        return []

    try:
        response = await _make_api_request(
            "GET", "/sizes", params={"per_page": 200}, api_key=api_key
        )
        offers = []

        sizes = response.get("sizes", [])

        for size in sizes:
            # Skip non-GPU sizes
            gpu_info = size.get("gpu_info")
            if not gpu_info:
                continue

            # Extract GPU details
            gpu_count_in_size = gpu_info.get("count", 1)
            gpu_model = gpu_info.get("model", "unknown")
            vram_info = gpu_info.get("vram", {})
            vram_gb = vram_info.get("amount", 0)

            # Only include AMD GPUs
            if not gpu_model.startswith("amd_"):
                continue

            # Filter by GPU count
            if gpu_count_in_size != gpu_count:
                continue

            # Filter by memory
            memory_mb = size.get("memory", 0)
            memory_gib = memory_mb // 1024
            if memory_gb and memory_gib < memory_gb:
                continue

            # Filter by disk
            disk_gb = size.get("disk", 0)
            if container_disk_gb and disk_gb < container_disk_gb:
                continue

            # Get regions with availability
            regions = size.get("regions", [])
            if not regions:
                # Size exists but no regions available
                continue

            # Normalize GPU type name
            gpu_type = GPU_MODEL_NAMES.get(gpu_model, gpu_model)

            # Extract pricing (per-GPU price for consistency with other providers)
            price_per_hour = size.get("price_hourly", 0.0)
            price_per_gpu = (
                price_per_hour / gpu_count_in_size if gpu_count_in_size > 0 else price_per_hour
            )

            # Create offer for each region
            for region in regions:
                offer_id = f"do-amd-{size['slug']}-{region}"

                gpu_offer = GPUOffer(
                    id=offer_id,
                    provider="digitalocean_amd",
                    gpu_type=gpu_type,
                    gpu_count=gpu_count_in_size,
                    vcpu=size.get("vcpus", 0),
                    memory_gb=memory_gib,
                    vram_gb=vram_gb,
                    storage_gb=disk_gb,
                    price_per_hour=price_per_gpu,
                    availability_zone=region,
                    cloud_type=CloudType.SECURE,
                    spot=False,  # DigitalOcean doesn't have spot GPU instances
                    manufacturer="amd",
                    raw_data={
                        "slug": size["slug"],
                        "region": region,
                        "description": size.get("description", ""),
                        **size,
                    },
                )
                offers.append(gpu_offer)

        return offers

    except Exception as e:
        logger.exception(f"Failed to search DigitalOcean AMD GPU offers: {e}")
        return []


async def provision_instance(
    request: ProvisionRequest, ssh_startup_script: str | None = None, api_key: str | None = None
) -> GPUInstance | None:
    """Provision an AMD GPU Droplet on DigitalOcean AMD Developer Cloud.

    Args:
        request: Provision request with GPU type, name, etc.
        ssh_startup_script: Optional cloud-init user data script
        api_key: DigitalOcean AMD API key

    Returns:
        GPUInstance if successful, None otherwise
    """
    # Parse size slug and region from offer ID
    # Format: "do-amd-{slug}-{region}"
    size_slug = None
    region = None

    if request.gpu_type and request.gpu_type.startswith("do-amd-"):
        # Remove "do-amd-" prefix and split off region (last part)
        parts = request.gpu_type[7:].rsplit("-", 1)
        if len(parts) == 2:
            size_slug = parts[0]
            region = parts[1]

    if not size_slug or not region:
        logger.error(
            f"Invalid offer ID format: {request.gpu_type}. Expected: do-amd-<slug>-<region>"
        )
        return None

    # Get SSH key IDs from DigitalOcean account
    ssh_key_ids = []
    try:
        ssh_keys_response = await _make_api_request("GET", "/account/keys", api_key=api_key)
        ssh_keys = ssh_keys_response.get("ssh_keys", [])
        if ssh_keys:
            # Use all available SSH keys
            ssh_key_ids = [key["id"] for key in ssh_keys]
            logger.info(f"using {len(ssh_key_ids)} ssh key(s)")
        else:
            logger.warning(
                "No SSH keys found in DigitalOcean AMD account. Instance may not be accessible."
            )
    except Exception as e:
        logger.warning(f"Failed to fetch SSH keys: {e}")

    # Generate instance name
    instance_name = request.name or f"amd-gpu-{size_slug}-{int(time.time())}"

    # Use AMD AI/ML Ready image by default (has ROCm pre-installed)
    image = "gpu-amd-base"
    if request.image and not request.image.startswith("runpod/"):
        image = request.image

    create_data = {
        "name": instance_name,
        "region": region,
        "size": size_slug,
        "image": image,
        "ssh_keys": ssh_key_ids,
        "backups": False,
        "ipv6": True,
        "monitoring": True,
    }

    # Add user_data (cloud-init) if provided
    if ssh_startup_script:
        create_data["user_data"] = ssh_startup_script

    try:
        response = await _make_api_request("POST", "/droplets", data=create_data, api_key=api_key)

        if not response or "droplet" not in response:
            logger.error(f"No droplet data returned from DigitalOcean AMD: {response}")
            return None

        droplet = response["droplet"]
        droplet_id = str(droplet["id"])
        logger.info(f"digitalocean amd droplet created: {droplet_id}")

        # Wait a moment for droplet to be queryable
        await trio.sleep(2)

        # Fetch full details
        instance = await get_instance_details(droplet_id, api_key=api_key)
        if instance:
            return instance

        # Fallback: create minimal instance object
        return GPUInstance(
            id=droplet_id,
            provider="digitalocean_amd",
            status=InstanceStatus.PENDING,
            gpu_type=size_slug,
            gpu_count=request.gpu_count or 1,
            name=instance_name,
            price_per_hour=0.0,
            raw_data=response,
            api_key=api_key,
        )

    except httpx.HTTPStatusError as e:
        # Check for specific error messages
        error_msg = str(e)
        if hasattr(e, "response") and e.response is not None:
            try:
                error_data = e.response.json()
                error_msg = error_data.get("message", str(e))
            except Exception:
                pass

        if "droplet limit" in error_msg.lower():
            logger.exception(
                "AMD GPU Droplet limit reached. Contact DigitalOcean sales or request a limit increase."
            )
        else:
            logger.exception(f"Failed to provision DigitalOcean AMD droplet: {error_msg}")
        return None

    except Exception as e:
        logger.exception(f"Failed to provision DigitalOcean AMD droplet: {e}")
        return None


async def get_instance_details(instance_id: str, api_key: str | None = None) -> GPUInstance | None:
    """Get details of a specific AMD Droplet.

    Args:
        instance_id: Droplet ID
        api_key: DigitalOcean AMD API key

    Returns:
        GPUInstance if found, None otherwise
    """
    try:
        response = await _make_api_request("GET", f"/droplets/{instance_id}", api_key=api_key)

        if not response or "droplet" not in response:
            return None

        droplet = response["droplet"]
        return _parse_droplet_to_gpu_instance(droplet, api_key=api_key)

    except Exception as e:
        logger.exception(f"Failed to get DigitalOcean AMD droplet details: {e}")
        return None


async def list_instances(api_key: str | None = None) -> list[GPUInstance]:
    """List all user's AMD GPU Droplets.

    Args:
        api_key: DigitalOcean AMD API key

    Returns:
        List of GPU instances
    """
    try:
        response = await _make_api_request(
            "GET", "/droplets", params={"per_page": 200}, api_key=api_key
        )

        instances = []
        droplets = response.get("droplets", [])

        for droplet in droplets:
            try:
                # Check if it's a GPU droplet by checking size slug
                size_slug = droplet.get("size_slug", "")
                if not size_slug.startswith("gpu-"):
                    continue

                instance = _parse_droplet_to_gpu_instance(droplet, api_key=api_key)
                if instance:
                    instances.append(instance)
            except Exception as e:
                logger.warning(f"Failed to parse droplet {droplet.get('id', 'unknown')}: {e}")
                continue

        return instances

    except Exception as e:
        logger.exception(f"Failed to list DigitalOcean AMD droplets: {e}")
        return []


async def terminate_instance(instance_id: str, api_key: str | None = None) -> bool:
    """Terminate (delete) a DigitalOcean AMD Droplet.

    Args:
        instance_id: Droplet ID
        api_key: DigitalOcean AMD API key

    Returns:
        True if successful, False otherwise
    """
    try:
        await _make_api_request("DELETE", f"/droplets/{instance_id}", api_key=api_key)
        logger.info(f"successfully terminated digitalocean amd droplet {instance_id}")
        return True

    except Exception as e:
        logger.exception(f"Failed to terminate DigitalOcean AMD droplet: {e}")
        return False


def _parse_droplet_to_gpu_instance(
    droplet: dict[str, Any], api_key: str | None = None
) -> GPUInstance:
    """Parse a DigitalOcean AMD Droplet dictionary into a GPUInstance.

    Args:
        droplet: Droplet data from API
        api_key: API key to store for instance methods

    Returns:
        GPUInstance object
    """
    # Map DigitalOcean statuses to our enum
    # DO statuses: "new", "active", "off", "archive"
    status_map = {
        "new": InstanceStatus.PENDING,
        "active": InstanceStatus.RUNNING,
        "off": InstanceStatus.STOPPED,
        "archive": InstanceStatus.TERMINATED,
    }
    status_str = droplet.get("status", "new")
    status = status_map.get(status_str, InstanceStatus.PENDING)

    # Extract instance info
    droplet_id = str(droplet.get("id", ""))
    name = droplet.get("name", "")

    # Extract GPU info from size
    size = droplet.get("size", {})
    gpu_info = size.get("gpu_info", {})
    gpu_count = gpu_info.get("count", 1)
    gpu_model = gpu_info.get("model", "unknown")
    gpu_type = GPU_MODEL_NAMES.get(gpu_model, gpu_model)

    # Extract public IP (first public IPv4)
    public_ip = None
    networks = droplet.get("networks", {})
    v4_networks = networks.get("v4", [])
    for net in v4_networks:
        if net.get("type") == "public":
            public_ip = net.get("ip_address")
            break

    # DigitalOcean uses standard SSH (port 22, username root)
    ssh_port = 22
    ssh_username = "root"

    # Extract pricing from size
    price_hourly = size.get("price_hourly", 0.0)
    price_per_gpu = price_hourly / gpu_count if gpu_count > 0 else price_hourly

    return GPUInstance(
        id=droplet_id,
        provider="digitalocean_amd",
        status=status,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        name=name,
        price_per_hour=price_per_gpu,
        public_ip=public_ip,
        ssh_port=ssh_port,
        ssh_username=ssh_username,
        raw_data=droplet,
        api_key=api_key,
    )


async def wait_for_ssh_ready(instance, timeout: int = 900) -> bool:
    """DigitalOcean AMD-specific SSH waiting implementation.

    Args:
        instance: GPUInstance to wait for
        timeout: Maximum seconds to wait

    Returns:
        True if SSH ready, False if timeout/failure
    """
    # Assert preconditions
    assert instance.provider == "digitalocean_amd"
    assert instance.api_key
    assert timeout > 0

    start_time = time.time()

    # Wait for RUNNING status and IP assignment
    logger.debug(f"waiting for amd droplet {instance.id} to become active...")

    while time.time() - start_time < timeout:
        fresh = await get_instance_details(instance.id, api_key=instance.api_key)
        if not fresh:
            logger.error("AMD Droplet disappeared")
            return False

        if fresh.status == InstanceStatus.RUNNING and fresh.public_ip:
            # Update instance with fresh data
            instance.public_ip = fresh.public_ip
            instance.ssh_port = fresh.ssh_port
            instance.ssh_username = fresh.ssh_username
            instance.status = fresh.status

            elapsed = int(time.time() - start_time)
            logger.debug(f"amd droplet active with IP {fresh.public_ip} (took {elapsed}s)")

            # Wait for SSH daemon to start (DigitalOcean is usually fast)
            logger.debug("waiting 15s for ssh daemon to initialize...")
            await trio.sleep(15)
            return True

        elif fresh.status in [InstanceStatus.FAILED, InstanceStatus.TERMINATED]:
            logger.error(f"AMD Droplet terminal state: {fresh.status}")
            return False

        await trio.sleep(10)

    logger.error(f"Timeout waiting for AMD droplet after {timeout}s")
    return False


async def get_fresh_instance(instance_id: str, api_key: str):
    """Alias for get_instance_details (ProviderProtocol requirement)."""
    return await get_instance_details(instance_id, api_key=api_key)
