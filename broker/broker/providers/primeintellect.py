"""
Prime Intellect provider implementation
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import trio
from infra_utils.retry import async_retry

from ..client import AccountError
from ..types import CloudType, GPUInstance, GPUOffer, InstanceStatus, ProvisionRequest

logger = logging.getLogger(__name__)

PRIME_API_BASE_URL = "https://api.primeintellect.ai/api/v1"

# Spinup time cache from dashboard (populated by scripts/prime_dashboard_auth.py)
SPINUP_CACHE_FILE = Path.home() / ".prime" / "spinup_times_cache.json"
_spinup_cache: dict[str, float] | None = None


def _load_spinup_cache() -> dict[str, float]:
    """Load spinup times cache. Returns {provider: avg_seconds}."""
    global _spinup_cache
    if _spinup_cache is not None:
        return _spinup_cache

    if not SPINUP_CACHE_FILE.exists():
        _spinup_cache = {}
        return _spinup_cache

    try:
        data = json.loads(SPINUP_CACHE_FILE.read_text())
        stats = data.get("stats", {})
        # Extract just provider -> avg_seconds mapping
        _spinup_cache = {
            provider: info["avg_seconds"]
            for provider, info in stats.items()
            if "avg_seconds" in info
        }
        return _spinup_cache
    except (json.JSONDecodeError, KeyError, TypeError):
        _spinup_cache = {}
        return _spinup_cache


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
    """Make a REST API request to Prime Intellect API with automatic retries.

    Retries up to 3 times with exponential backoff (1s, 2s, 4s) on network errors.

    Args:
        method: HTTP method (GET, POST, DELETE, etc.)
        endpoint: API endpoint (e.g., "/availability/")
        data: Optional request body data
        params: Optional query parameters
        api_key: Prime Intellect API key (required)
    """
    if not api_key:
        raise ValueError("Prime Intellect API key is required but was not provided")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    url = f"{PRIME_API_BASE_URL}{endpoint}"
    logger.debug(
        "Prime Intellect API request %s %s (api key ...%s)",
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
        logger.exception("Prime Intellect API request timed out")
        raise
    except httpx.HTTPError as exc:
        # 401/403 = auth/account error, not retryable
        if hasattr(exc, "response") and exc.response is not None:
            if exc.response.status_code in (401, 403):
                key_hint = api_key[-4:] if api_key else "none"
                raise AccountError(
                    "Prime Intellect API key invalid or unauthorized",
                    provider="primeintellect",
                    key_hint=key_hint,
                ) from exc
            # Log full response body for debugging
            logger.error(
                f"Prime Intellect API request failed: {exc} - Response: {exc.response.text}"
            )
        else:
            logger.error(f"Prime Intellect API request failed: {exc}")  # noqa: TRY400 — re-raising, don't want duplicate traceback
        raise

    # Handle empty responses (e.g., DELETE operations)
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
        data = await _make_api_request("GET", "/availability/", params=params, api_key=api_key)
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
                cloud_type = (
                    CloudType.SECURE
                    if offer.get("security") == "secure_cloud"
                    else CloudType.COMMUNITY
                )

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
                price_per_hour = (
                    total_price / offer_gpu_count if offer_gpu_count > 0 else total_price
                )

                # Create unique offer ID
                offer_id = (
                    f"prime-{offer.get('cloudId', 'unknown')}-{offer.get('dataCenter', 'unknown')}"
                )

                # Look up estimated spinup time from cache
                underlying = offer.get("provider")
                spinup_cache = _load_spinup_cache()
                estimated_spinup = spinup_cache.get(underlying) if underlying else None

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
                    underlying_provider=underlying,
                    estimated_spinup_seconds=estimated_spinup,
                    raw_data=offer,
                )
                offers.append(gpu_offer)

        return offers

    except AccountError:
        raise
    except Exception as e:
        logger.exception(f"Failed to search Prime Intellect GPU offers: {e}")
        return []


async def provision_instance(
    request: ProvisionRequest, ssh_startup_script: str | None = None, api_key: str | None = None
) -> GPUInstance | None:
    """Provision a GPU instance on Prime Intellect"""
    raw = request.raw_data or {}

    # Extract required fields from raw offer data
    cloud_id = raw.get("cloudId")
    gpu_type = raw.get("gpuType")
    socket = raw.get("socket")
    data_center = raw.get("dataCenter")
    provider_type = raw.get("provider")  # e.g., "datacrunch", "hyperstack"
    security = raw.get("security", "secure_cloud")

    if not all([cloud_id, gpu_type, socket, data_center, provider_type]):
        logger.error(
            f"Missing required fields from offer raw_data: cloudId={cloud_id}, "
            f"gpuType={gpu_type}, socket={socket}, dataCenter={data_center}, provider={provider_type}"
        )
        return None

    # Valid Prime Intellect images (not Docker images like RunPod)
    VALID_IMAGES = {
        "ubuntu_22_cuda_12",
        "cuda_12_1_pytorch_2_2",
        "cuda_11_8_pytorch_2_1",
        "cuda_12_1_pytorch_2_3",
        "cuda_12_1_pytorch_2_4",
        "cuda_12_4_pytorch_2_4",
        "cuda_12_4_pytorch_2_5",
        "cuda_12_4_pytorch_2_6",
        "cuda_12_6_pytorch_2_7",
        "stable_diffusion",
        "axolotl",
        "bittensor",
        "hivemind",
        "petals_llama",
        "vllm_llama_8b",
        "vllm_llama_70b",
        "vllm_llama_405b",
        "custom_template",
        "flux",
        "prime_rl",
    }

    # Use requested image if valid, otherwise default to ubuntu_22_cuda_12
    image = request.image if request.image in VALID_IMAGES else "ubuntu_22_cuda_12"

    # Build the pod definition with all required fields
    pod_data = {
        "name": request.name or f"prime-{gpu_type}-{int(time.time())}",
        "cloudId": cloud_id,
        "gpuType": gpu_type,
        "socket": socket,
        "dataCenterId": data_center,
        "gpuCount": request.gpu_count,
        "image": image,
        "security": security,
    }

    # Build provider specification
    provider_data = {"type": provider_type}

    # Add resource specifications if provided
    if request.container_disk_gb:
        pod_data["diskSize"] = request.container_disk_gb
    if request.memory_gb:
        pod_data["memory"] = request.memory_gb

    # Add environment variables if startup script provided
    env_vars = []
    if ssh_startup_script:
        env_vars.append({"key": "STARTUP_SCRIPT", "value": ssh_startup_script})

    # Add Jupyter password if provided
    if request.jupyter_password:
        pod_data["jupyterPassword"] = request.jupyter_password

    if env_vars:
        pod_data["envVars"] = env_vars

    # Build the complete request body
    request_body = {"pod": pod_data, "provider": provider_data}

    # Attach disks if specified in raw_data
    # Can be a single disk ID string or list of disk IDs
    disk_ids = raw.get("disk_ids") or request.raw_data.get("disk_ids") if request.raw_data else None
    if disk_ids:
        if isinstance(disk_ids, str):
            disk_ids = [disk_ids]
        request_body["disks"] = disk_ids
        logger.info(f"Attaching disks: {disk_ids}")

    # Look up estimated spinup time for this provider
    spinup_cache = _load_spinup_cache()
    estimated_spinup = spinup_cache.get(provider_type)

    try:
        data = await _make_api_request("POST", "/pods/", data=request_body, api_key=api_key)

        if not data:
            logger.error("No pod returned from Prime Intellect deployment")
            return None

        # Parse the response and create GPUInstance
        return _parse_pod_to_instance(
            data, api_key=api_key, estimated_spinup_seconds=estimated_spinup
        )

    except AccountError:
        raise
    except Exception as e:
        logger.exception(f"Failed to provision Prime Intellect instance: {e}")
        return None


async def get_instance_details(instance_id: str, api_key: str | None = None) -> GPUInstance | None:
    """Get details of a specific instance"""
    try:
        data = await _make_api_request("GET", f"/pods/{instance_id}", api_key=api_key)

        if not data:
            return None

        return _parse_pod_to_instance(data, api_key=api_key)

    except Exception as e:
        logger.exception(f"Failed to get Prime Intellect instance details: {e}")
        return None


async def list_instances(api_key: str | None = None) -> list[GPUInstance]:
    """List all user's instances"""
    try:
        data = await _make_api_request("GET", "/pods/", api_key=api_key)

        instances = []
        # API returns {"data": [...], "total_count": N, ...}
        pods = data if isinstance(data, list) else data.get("data", [])

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
        logger.exception(f"Failed to list Prime Intellect instances: {e}")
        return []


async def terminate_instance(instance_id: str, api_key: str | None = None) -> bool:
    """Terminate a Prime Intellect instance"""
    try:
        await _make_api_request("DELETE", f"/pods/{instance_id}", api_key=api_key)
        logger.info(f"successfully terminated prime intellect instance {instance_id}")
        return True

    except Exception as e:
        logger.exception(f"Failed to terminate Prime Intellect instance: {e}")
        return False


def _parse_pod_to_instance(
    pod: dict[str, Any],
    api_key: str | None = None,
    estimated_spinup_seconds: float | None = None,
) -> GPUInstance:
    """Parse a pod dictionary into a GPUInstance"""

    # Map Prime Intellect statuses to our enum
    status_map = {
        "PROVISIONING": InstanceStatus.PENDING,
        "RUNNING": InstanceStatus.RUNNING,
        "STOPPED": InstanceStatus.STOPPED,
        "TERMINATED": InstanceStatus.TERMINATED,
        "FAILED": InstanceStatus.FAILED,
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
        api_key=api_key,  # Store API key for instance methods
        estimated_spinup_seconds=estimated_spinup_seconds,
    )


async def get_user_balance(api_key: str | None = None) -> dict[str, Any] | None:
    """Get user balance and spending information from Prime Intellect"""

    # Note: This endpoint might not exist in Prime Intellect API
    # Returning None for now, can be implemented when endpoint is available
    try:
        # Placeholder - actual endpoint TBD
        # data = await _make_api_request("GET", "/user/balance", api_key=api_key)

        return {
            "provider": "primeintellect",
            "current_balance": None,
            "message": "Balance endpoint not yet implemented for Prime Intellect",
        }

    except Exception as e:
        logger.exception(f"Failed to get Prime Intellect user balance: {e}")
        return None


async def wait_for_ssh_ready(instance, timeout: int = 300) -> bool:
    """Prime Intellect-specific SSH waiting implementation"""
    # Tiger Style: Assert preconditions
    assert instance.provider == "primeintellect"
    assert instance.api_key
    assert timeout > 0

    # Wait for RUNNING status
    if not await _wait_until_running(instance, timeout):
        return False

    # Wait for SSH details to be populated
    if not await _wait_for_ssh_assignment(instance, time.time(), timeout):
        return False

    # Test connectivity
    return await _test_ssh_connectivity(instance)


async def _wait_until_running(instance, timeout: int) -> bool:
    """Wait for instance to reach RUNNING status"""
    start_time = time.time()

    logger.debug(f"waiting for instance {instance.id} to reach running...")

    while time.time() - start_time < timeout:
        fresh = await get_instance_details(instance.id, api_key=instance.api_key)
        if not fresh:
            logger.error("Instance disappeared")
            return False

        if fresh.status.value == "running":
            instance.__dict__.update(fresh.__dict__)
            logger.info(f"instance {instance.id} is running")
            return True
        elif fresh.status.value in ["failed", "terminated"]:
            logger.error(f"Instance terminal state: {fresh.status}")
            return False

        await trio.sleep(15)

    logger.error(f"Timeout waiting for RUNNING after {timeout}s")
    return False


async def _wait_for_ssh_assignment(instance, start_time: float, timeout: int) -> bool:
    """Wait for SSH details to be assigned"""
    logger.debug("waiting for ssh details...")
    next_log_time = start_time + 30  # Log at 30s, 60s, 90s, ...

    while time.time() - start_time < timeout:
        fresh = await get_instance_details(instance.id, api_key=instance.api_key)

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

        await trio.sleep(10)

    elapsed_min = int((time.time() - start_time) / 60)
    logger.error(f"Timeout waiting for SSH after {elapsed_min} min")
    return False


async def _test_ssh_connectivity(instance) -> bool:
    """Test SSH connectivity with echo command"""
    logger.debug("ssh details ready! waiting 30s for ssh daemon...")
    await trio.sleep(30)

    try:
        result = instance.exec("echo 'ssh_ready'", timeout=30)
        if result.success and "ssh_ready" in result.stdout:
            logger.debug("ssh connectivity confirmed!")
            return True
        else:
            logger.warning(f"SSH test failed: {result.stderr}")
            return False
    except Exception as e:
        logger.exception(f"SSH connection error: {e}")
        return False


async def get_fresh_instance(instance_id: str, api_key: str):
    """Alias for get_instance_details (ProviderProtocol requirement)"""
    return await get_instance_details(instance_id, api_key=api_key)


# =============================================================================
# Disk Operations
# =============================================================================


@dataclass
class DiskOffer:
    """Available disk configuration from Prime Intellect."""

    provider: str  # e.g., "hyperstack", "runpod"
    datacenter_id: str
    datacenter_name: str
    min_size_gb: int
    max_size_gb: int
    price_per_gb_hour: float  # $/GB/hour

    @property
    def price_per_gb_month(self) -> float:
        """Price per GB per month (assuming 720 hours)."""
        return self.price_per_gb_hour * 720


@dataclass
class Disk:
    """A Prime Intellect persistent disk."""

    id: str
    name: str | None
    size_gb: int
    status: str  # PROVISIONING, PENDING, ACTIVE, STOPPED, DELETING, TERMINATED
    provider: str
    datacenter_id: str
    datacenter_name: str | None
    price_per_gb_hour: float
    created_at: str | None = None
    raw_data: dict[str, Any] | None = None

    @property
    def is_ready(self) -> bool:
        """Check if disk is ready to attach."""
        return self.status == "ACTIVE"

    @property
    def monthly_cost(self) -> float:
        """Estimated monthly cost."""
        return self.size_gb * self.price_per_gb_hour * 720


async def list_disk_availability(api_key: str | None = None) -> list[DiskOffer]:
    """List available disk configurations (providers, datacenters, pricing).

    Returns list of DiskOffer with pricing and size constraints.
    """
    try:
        data = await _make_api_request("GET", "/availability/disks", api_key=api_key)
        offers = []

        # API returns {"items": [...], "totalCount": N}
        for item in data.get("items", []):
            spec = item.get("spec", {})
            provider = item.get("provider", "unknown")  # string, not object
            datacenter = item.get("dataCenter")  # string or null

            offers.append(
                DiskOffer(
                    provider=provider,
                    datacenter_id=datacenter or "",
                    datacenter_name=datacenter or "(default)",
                    min_size_gb=spec.get("minCount", 0),
                    max_size_gb=spec.get("maxCount", 10000),
                    price_per_gb_hour=spec.get("pricePerUnit", 0.0),
                )
            )

        return offers

    except Exception as e:
        logger.exception(f"Failed to list disk availability: {e}")
        return []


async def create_disk(
    size_gb: int,
    provider: str,
    datacenter_id: str,
    name: str | None = None,
    api_key: str | None = None,
) -> Disk | None:
    """Create a persistent disk.

    Args:
        size_gb: Disk size in GB
        provider: Provider type (e.g., "hyperstack", "runpod")
        datacenter_id: Datacenter ID from availability response
        name: Optional human-readable name
        api_key: Prime Intellect API key

    Returns:
        Disk object if successful, None otherwise
    """
    payload = {
        "disk": {
            "size": size_gb,
            "dataCenterId": datacenter_id,
        },
        "provider": {
            "type": provider,
        },
    }
    if name:
        payload["disk"]["name"] = name

    try:
        data = await _make_api_request("POST", "/disks/", data=payload, api_key=api_key)
        disk_data = data.get("data", data)
        logger.info(f"Created disk {disk_data.get('id')} ({size_gb}GB on {provider})")
        return _parse_disk(disk_data)

    except Exception as e:
        logger.exception(f"Failed to create disk: {e}")
        return None


async def list_disks(api_key: str | None = None) -> list[Disk]:
    """List all user's disks."""
    try:
        data = await _make_api_request("GET", "/disks/", api_key=api_key)
        disks = []

        for item in data.get("data", []):
            disk = _parse_disk(item)
            if disk:
                disks.append(disk)

        return disks

    except Exception as e:
        logger.exception(f"Failed to list disks: {e}")
        return []


async def get_disk(disk_id: str, api_key: str | None = None) -> Disk | None:
    """Get details of a specific disk."""
    try:
        data = await _make_api_request("GET", f"/disks/{disk_id}", api_key=api_key)
        return _parse_disk(data.get("data", data))

    except Exception as e:
        logger.exception(f"Failed to get disk {disk_id}: {e}")
        return None


async def delete_disk(disk_id: str, api_key: str | None = None) -> bool:
    """Delete a disk. WARNING: This is irreversible and all data will be lost."""
    try:
        await _make_api_request("DELETE", f"/disks/{disk_id}", api_key=api_key)
        logger.info(f"Deleted disk {disk_id}")
        return True

    except Exception as e:
        logger.exception(f"Failed to delete disk {disk_id}: {e}")
        return False


async def wait_for_disk_ready(
    disk_id: str,
    timeout: int = 300,
    api_key: str | None = None,
) -> Disk | None:
    """Wait for disk to become ACTIVE.

    Args:
        disk_id: Disk ID
        timeout: Max seconds to wait
        api_key: API key

    Returns:
        Disk if ready, None if timeout or error
    """
    start = time.time()
    while time.time() - start < timeout:
        disk = await get_disk(disk_id, api_key=api_key)
        if disk is None:
            return None
        if disk.is_ready:
            return disk
        if disk.status in ("TERMINATED", "FAILED"):
            logger.error(f"Disk {disk_id} entered terminal state: {disk.status}")
            return None

        await trio.sleep(5)

    logger.error(f"Timeout waiting for disk {disk_id} to become ready")
    return None


def _parse_disk(data: dict[str, Any]) -> Disk | None:
    """Parse disk API response into Disk object."""
    if not data:
        return None

    info = data.get("info", {})
    size_gb = data.get("size", 0)
    price_hr = data.get("priceHr", 0.0)
    # Convert total price/hr to per-GB price/hr
    price_per_gb_hr = price_hr / size_gb if size_gb > 0 else 0.0

    return Disk(
        id=data.get("id", ""),
        name=data.get("name"),
        size_gb=size_gb,
        status=data.get("status", "UNKNOWN"),
        provider=data.get("providerType", "unknown"),
        datacenter_id=info.get("dataCenterId", ""),
        datacenter_name=info.get("dataCenterId"),  # No separate name field
        price_per_gb_hour=price_per_gb_hr,
        created_at=data.get("createdAt"),
        raw_data=data,
    )
