"""
RunPod provider implementation
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from ..types import CloudType, GPUInstance, GPUOffer, InstanceStatus, ProvisionRequest

logger = logging.getLogger(__name__)

RUNPOD_API_URL = "https://api.runpod.io/graphql"

# GraphQL query for pod details - extracted to module level to avoid duplication
# Why separate constant: This 45-line query appears in multiple functions,
# extracting it reduces code size and ensures consistency across API calls
_POD_DETAILS_QUERY = """
query pod($input: PodFilter!) {
    pod(input: $input) {
        id
        name
        machineId
        imageName
        env
        machineId
        machine {
            podHostId
            gpuType {
                displayName
                manufacturer
                memoryInGb
            }
        }
        desiredStatus
        lastStatusChange
        gpuCount
        vcpuCount
        memoryInGb
        costPerHr
        containerDiskInGb
        volumeInGb
        ports
        runtime {
            uptimeInSeconds
            ports {
                ip
                isIpPublic
                privatePort
                publicPort
                type
            }
            gpus {
                id
                gpuUtilPercent
                memoryUtilPercent
            }
        }
    }
}
"""


def _is_direct_ssh_port(port: Dict[str, Any]) -> bool:
    """Check if port configuration represents direct SSH access.

    Why three conditions: RunPod exposes multiple ports, but only one
    with privatePort=22, isIpPublic=True, type=tcp is direct SSH.
    Proxy SSH uses different mechanism (podHostId).

    Tiger Style: Split compound condition for debuggability.
    Each condition is checked separately so failures are traceable.
    """
    # Tiger Style: Assert precondition
    assert isinstance(port, dict), f"Port must be dict, got {type(port)}"

    # Check each condition separately (easier to debug than compound condition)
    if port.get("privatePort") != 22:
        return False
    if not port.get("isIpPublic"):
        return False
    if port.get("type") != "tcp":
        return False

    return True


def _extract_direct_ssh(pod: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """Extract direct SSH connection info from pod runtime.

    Returns (ip, port) or None if direct SSH not available.

    Why runtime.ports: RunPod provisions direct IP 5-10 minutes after
    instance starts. Until then, runtime.ports is empty or missing.
    """
    runtime = pod.get("runtime")
    if not runtime:
        return None

    ports = runtime.get("ports")
    if not ports:
        return None

    # Tiger Style: Assert the type we expect
    assert isinstance(ports, list), f"runtime.ports must be list, got {type(ports)}"

    # Find SSH port
    for port in ports:
        if _is_direct_ssh_port(port):
            ip = port.get("ip")
            public_port = port.get("publicPort")

            # Tiger Style: Assert postconditions - validate extracted data
            # Why assert here: port passed _is_direct_ssh_port() but could still have missing fields
            assert ip, f"Direct SSH port found but missing IP: {port}"
            assert public_port, f"Direct SSH port found but missing publicPort: {port}"
            assert isinstance(public_port, int), f"publicPort must be int, got {type(public_port)}"
            assert 0 < public_port < 65536, f"Invalid port number: {public_port}"

            return (ip, public_port)

    return None


def _extract_proxy_ssh(pod: Dict[str, Any]) -> Optional[str]:
    """Extract proxy SSH podHostId from pod machine.

    Returns podHostId (username for ssh.runpod.io) or None.

    Why proxy fallback: RunPod always provides proxy SSH via ssh.runpod.io,
    even when direct IP not yet assigned. Uses podHostId as SSH username.
    """
    machine = pod.get("machine")
    if not machine:
        return None

    pod_host_id = machine.get("podHostId")
    if not pod_host_id:
        return None

    # Tiger Style: Assert postcondition
    assert isinstance(pod_host_id, str), f"podHostId must be string, got {type(pod_host_id)}"
    assert pod_host_id.strip(), "podHostId cannot be empty string"

    return pod_host_id


def _extract_ssh_info(pod: Dict[str, Any]) -> Optional[Tuple[str, int, str]]:
    """Extract SSH connection info with fallback strategy.

    Returns (public_ip, ssh_port, ssh_username) or None if no SSH method available.

    Why two-method fallback: Direct SSH is faster but takes 5-10 min to provision.
    Proxy SSH is always available immediately but slower. Try direct first, fall back to proxy.
    """
    # Tiger Style: Assert preconditions
    assert isinstance(pod, dict), f"Pod must be dict, got {type(pod)}"
    assert "id" in pod, "Pod missing required 'id' field"

    # Method 1: Try direct SSH (preferred for performance)
    direct_ssh = _extract_direct_ssh(pod)
    if direct_ssh:
        ip, port = direct_ssh
        return (ip, port, "root")  # Direct SSH always uses root

    # Method 2: Fallback to proxy SSH
    pod_host_id = _extract_proxy_ssh(pod)
    if pod_host_id:
        # Proxy SSH: connect to ssh.runpod.io using podHostId as username
        return ("ssh.runpod.io", 22, pod_host_id)

    # No SSH method available - pod may still be provisioning
    return None


def _extract_gpu_type(pod: Dict[str, Any]) -> str:
    """Extract GPU type name from pod machine details.

    Returns GPU display name or "unknown" if not available.
    """
    # Navigate nested structure safely
    machine = pod.get("machine")
    if not machine:
        return "unknown"

    gpu_type_obj = machine.get("gpuType")
    if not gpu_type_obj:
        return "unknown"

    display_name = gpu_type_obj.get("displayName")
    if not display_name:
        return "unknown"

    return display_name


def _map_status(desired_status: str) -> InstanceStatus:
    """Map RunPod status string to our InstanceStatus enum.

    Why separate function: Status mapping logic used by multiple functions,
    extracting ensures consistency and makes updates easier.
    """
    status_map = {
        "RUNNING": InstanceStatus.RUNNING,
        "PENDING": InstanceStatus.PENDING,
        "STOPPED": InstanceStatus.STOPPED,
        "TERMINATED": InstanceStatus.TERMINATED,
        "FAILED": InstanceStatus.FAILED
    }
    # Default to PENDING for unknown statuses
    return status_map.get(desired_status, InstanceStatus.PENDING)


def _build_ports_string(exposed_ports: Optional[List[int]], enable_http_proxy: bool) -> str:
    """Build ports string for RunPod instance configuration.
    
    Args:
        exposed_ports: List of ports to expose (e.g., [8000] for vLLM)
        enable_http_proxy: Whether to enable HTTP proxy for exposed ports
        
    Returns:
        Ports string in RunPod format (e.g., "22/tcp,8000/http")
    """
    ports = ["22/tcp"]  # Always include SSH
    
    if exposed_ports and enable_http_proxy:
        for port in exposed_ports:
            ports.append(f"{port}/http")
    elif exposed_ports:
        # TCP only if HTTP proxy disabled
        for port in exposed_ports:
            ports.append(f"{port}/tcp")
            
    return ",".join(ports)


def _make_graphql_request(query: str, variables: Optional[Dict] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Make a GraphQL request to RunPod API

    Args:
        query: GraphQL query string
        variables: Optional GraphQL variables
        api_key: RunPod API key (required)
    """
    if not api_key:
        raise ValueError("RunPod API key is required but was not provided")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    logger.debug("RunPod GraphQL request %s (api key ...%s)", query.split("\n")[0][:60], api_key[-4:] if api_key else "none")
    
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    try:
        response = requests.post(
            RUNPOD_API_URL,
            json=payload,
            headers=headers,
            timeout=(10, 30),
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        logger.error("RunPod GraphQL request timed out")
        raise
    except requests.RequestException as exc:
        logger.error(f"RunPod GraphQL request failed: {exc}")
        raise

    data = response.json()
    if "errors" in data:
        raise Exception(f"GraphQL errors: {data['errors']}")
    
    return data["data"]


def search_gpu_offers(cuda_version: Optional[str] = None, manufacturer: Optional[str] = None,
                      memory_gb: Optional[int] = None, container_disk_gb: Optional[int] = None,
                      gpu_count: int = 1, api_key: Optional[str] = None) -> List[GPUOffer]:
    """Search for available GPU offers on RunPod with optional CUDA version and manufacturer filtering

    Note: Queries secure and community clouds separately to get accurate stock availability
    """
    offers = []

    # Query secure and community clouds separately to get accurate availability data
    # This is necessary because availability differs between cloud types
    for cloud_type_filter in [True, False]:  # True = secure, False = community
        cloud_name = "secure" if cloud_type_filter else "community"

        # Build lowestPrice input with cloud type filter
        lowest_price_params = [f"gpuCount: {gpu_count}", f"secureCloud: {str(cloud_type_filter).lower()}"]
        if cuda_version:
            lowest_price_params.append(f'cudaVersion: "{cuda_version}"')
        lowest_price_input = f"{{ {', '.join(lowest_price_params)} }}"

        # Query to get available GPU types for this cloud type
        query = f"""
        query {{
            gpuTypes {{
                id
                displayName
                memoryInGb
                manufacturer
                secureCloud
                communityCloud
                lowestPrice(input: {lowest_price_input}) {{
                    minimumBidPrice
                    uninterruptablePrice
                    stockStatus
                    maxUnreservedGpuCount
                    availableGpuCounts
                }}
            }}
        }}
        """

        try:
            data = _make_graphql_request(query, api_key=api_key)

            for gpu_type in data.get("gpuTypes", []):
                # Filter by manufacturer if specified
                if manufacturer and gpu_type.get("manufacturer"):
                    gpu_manufacturer = gpu_type["manufacturer"].lower()
                    if manufacturer.lower() not in gpu_manufacturer:
                        continue

                # Get pricing info for this cloud type
                # Note: RunPod's lowestPrice returns total price for requested gpu_count
                # We normalize to per-GPU pricing for consistency
                price_info = gpu_type.get("lowestPrice", {})

                # Check if this cloud type is available and has pricing
                if cloud_type_filter:  # Secure cloud
                    if not gpu_type.get("secureCloud") or not price_info.get("uninterruptablePrice"):
                        continue
                    total_price = price_info["uninterruptablePrice"]
                    cloud_type = CloudType.SECURE
                    offer_id = f"runpod-{gpu_type['id']}-secure"
                    spot = False
                else:  # Community cloud
                    if not gpu_type.get("communityCloud") or not price_info.get("minimumBidPrice"):
                        continue
                    total_price = price_info["minimumBidPrice"]
                    cloud_type = CloudType.COMMUNITY
                    offer_id = f"runpod-{gpu_type['id']}-community-spot"
                    spot = True

                gpu_vram = gpu_type.get("memoryInGb", 0)
                per_gpu_price = total_price / gpu_count if gpu_count > 0 else total_price

                # Extract availability information
                max_gpu_count = price_info.get("maxUnreservedGpuCount")
                available_counts = price_info.get("availableGpuCounts", [])
                stock_status = price_info.get("stockStatus")

                offers.append(GPUOffer(
                    id=offer_id,
                    provider="runpod",
                    gpu_type=gpu_type["displayName"],
                    gpu_count=gpu_count,
                    vcpu=0,  # Not specified in this query
                    memory_gb=gpu_vram,  # For backward compatibility - this is actually GPU VRAM
                    vram_gb=gpu_vram,    # Properly populate VRAM field
                    storage_gb=0,  # Not specified in this query
                    price_per_hour=per_gpu_price,  # Normalized per-GPU price
                    availability_zone=f"{cloud_name}-cloud",
                    cloud_type=cloud_type,
                    spot=spot,
                    cuda_version=cuda_version,
                    manufacturer=gpu_type.get("manufacturer"),
                    max_gpu_count=max_gpu_count,
                    available_gpu_counts=available_counts,
                    stock_status=stock_status,
                    raw_data=gpu_type
                ))

        except Exception as e:
            logger.error(f"Failed to query RunPod {cloud_name} cloud: {e}")
            continue

    return offers


def provision_instance(request: ProvisionRequest, ssh_startup_script: Optional[str] = None, api_key: Optional[str] = None) -> Optional[GPUInstance]:
    """Provision a GPU instance on RunPod"""
    # First, we need to find a suitable GPU type
    # For now, let's implement a simple approach using podFindAndDeployOnDemand
    
    mutation = """
    mutation podFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            machineId
            machine {
                podHostId
            }
        }
    }
    """
    
    # Build input for the mutation
    env_vars = []
    
    # Add SSH startup script if provided
    if ssh_startup_script:
        env_vars.append({
            "key": "RUNPOD_STARTUP_SCRIPT",
            "value": ssh_startup_script
        })
    
    # Add Jupyter password if provided
    if request.jupyter_password:
        env_vars.append({
            "key": "JUPYTER_PASSWORD",
            "value": request.jupyter_password
        })
    
    pod_input = {
        "gpuCount": request.gpu_count,
        "imageName": request.image,
        "cloudType": "SECURE" if not request.spot_instance else "COMMUNITY",
        "name": request.name or f"gpus-{request.gpu_type or 'auto'}-{int(time.time())}",
        "supportPublicIp": True,  # Required for SSH access
        "containerDiskInGb": request.container_disk_gb or 50,  # Default 50GB for ML workloads, configurable
        "minVcpuCount": 1,  # Required minimum CPU
        "minMemoryInGb": request.memory_gb or 4,  # Use requested memory or default 4GB minimum
        "ports": _build_ports_string(request.exposed_ports, request.enable_http_proxy),
        "startSsh": True,  # This enables SSH daemon
        "startJupyter": request.start_jupyter,  # Auto-start Jupyter Lab
        "env": env_vars
    }

    # Add volume configuration if requested
    # IMPORTANT: volumeMountPath is REQUIRED when volumeInGb > 0
    # Without it, RunPod fails with "invalid mount config for type bind field target must not be empty"
    if request.volume_disk_gb and request.volume_disk_gb > 0:
        pod_input["volumeInGb"] = request.volume_disk_gb
        pod_input["volumeMountPath"] = "/workspace"  # RunPod default mount point

    # Add GPU type if specified
    if request.gpu_type:
        # Use the GPU type ID directly - it should already be the full RunPod ID
        pod_input["gpuTypeId"] = request.gpu_type
    
    variables = {"input": pod_input}
    
    try:
        data = _make_graphql_request(mutation, variables, api_key=api_key)
        pod_data = data.get("podFindAndDeployOnDemand")
        
        if not pod_data:
            logger.error("No pod returned from deployment")
            return None
        
        # Return basic instance info - we'll need another query to get full details
        return GPUInstance(
            id=pod_data["id"],
            provider="runpod",
            status=InstanceStatus.PENDING,
            gpu_type=request.gpu_type or "auto-selected",
            gpu_count=request.gpu_count,
            name=request.name,  # Fix: Use name from request
            price_per_hour=0.0,  # We'll get this from a separate query
            raw_data=pod_data,
            api_key=api_key  # Store API key for instance methods
        )
        
    except Exception as e:
        logger.error(f"Failed to provision RunPod instance: {e}")
        return None


def get_instance_details_enhanced(instance_id: str, api_key: Optional[str] = None) -> Optional[dict]:
    """Get comprehensive details of a specific instance with all available fields"""
    query = """
    query pod($input: PodFilter!) {
        pod(input: $input) {
            id
            name
            machineId
            imageName
            env
            desiredStatus
            lastStatusChange
            gpuCount
            vcpuCount
            memoryInGb
            costPerHr
            containerDiskInGb
            volumeInGb
            ports
            runtime {
                uptimeInSeconds
                ports {
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                    type
                }
                gpus {
                    id
                    gpuUtilPercent
                    memoryUtilPercent
                }
            }
            machine {
                podHostId
                gpuType {
                    displayName
                    manufacturer
                    memoryInGb
                }
            }
        }
    }
    """
    
    variables = {"input": {"podId": instance_id}}
    
    try:
        response = _make_graphql_request(query, variables, api_key=api_key)
        pod_data = response.get("pod")
        
        if not pod_data:
            return None
        
        return pod_data
        
    except Exception as e:
        logger.error(f"Failed to get RunPod instance details: {e}")
        return None


def get_instance_details(instance_id: str, api_key: Optional[str] = None) -> Optional[GPUInstance]:
    """Get details of a specific instance.

    Returns None if instance not found, raises on malformed data.

    Tiger Style: Function split into small helpers (SSH extraction, GPU extraction, status mapping).
    This brings line count from 113 to ~40 lines while adding proper assertions.
    """
    # Tiger Style: Assert preconditions
    assert instance_id, "instance_id cannot be empty"
    assert api_key, "api_key is required for RunPod API access"

    variables = {"input": {"podId": instance_id}}

    try:
        data = _make_graphql_request(_POD_DETAILS_QUERY, variables, api_key=api_key)
        pod = data.get("pod")

        if not pod:
            return None

        # Tiger Style: Assert critical fields exist before using them
        assert "id" in pod, f"Pod missing required 'id' field: {pod.keys()}"

        # Map status (delegated to helper)
        status = _map_status(pod.get("desiredStatus", ""))

        # Extract SSH info (delegated to helper with explicit fallback strategy)
        ssh_info = _extract_ssh_info(pod)
        if ssh_info:
            public_ip, ssh_port, ssh_username = ssh_info
        else:
            # SSH not available yet - pod may still be provisioning
            logger.debug(f"Instance {instance_id} has no SSH access yet (still provisioning)")
            public_ip, ssh_port, ssh_username = None, None, None

        # Extract GPU type (delegated to helper)
        gpu_type = _extract_gpu_type(pod)

        return GPUInstance(
            id=pod["id"],
            provider="runpod",
            status=status,
            gpu_type=gpu_type,
            gpu_count=pod.get("gpuCount", 0),
            name=pod.get("name"),
            price_per_hour=pod.get("costPerHr", 0.0),
            public_ip=public_ip,
            ssh_port=ssh_port,
            ssh_username=ssh_username,
            raw_data=pod,
            api_key=api_key
        )

    except Exception as e:
        logger.error(f"Failed to get RunPod instance details: {e}")
        return None


def _parse_pod_to_instance(pod: Dict[str, Any], api_key: Optional[str] = None) -> GPUInstance:
    """Parse a pod dictionary into a GPUInstance.

    Tiger Style: Refactored to use shared helpers, eliminating 40+ lines of duplication.
    Now has 2+ assertions and delegates SSH/GPU extraction to tested functions.
    """
    # Tiger Style: Assert preconditions
    assert isinstance(pod, dict), f"Pod must be dict, got {type(pod)}"
    assert "id" in pod, f"Pod missing required 'id' field: {pod.keys()}"

    # Map status (delegated to helper - ensures consistency with get_instance_details)
    status = _map_status(pod.get("desiredStatus", ""))

    # Extract SSH info (delegated to helper - same logic as get_instance_details)
    ssh_info = _extract_ssh_info(pod)
    if ssh_info:
        public_ip, ssh_port, ssh_username = ssh_info
    else:
        # SSH not available yet
        public_ip, ssh_port, ssh_username = None, None, None

    # Extract GPU type (delegated to helper)
    gpu_type = _extract_gpu_type(pod)

    return GPUInstance(
        id=pod["id"],
        provider="runpod",
        status=status,
        gpu_type=gpu_type,
        gpu_count=pod.get("gpuCount", 0),
        name=pod.get("name"),
        price_per_hour=pod.get("costPerHr", 0.0),
        public_ip=public_ip,
        ssh_port=ssh_port,
        ssh_username=ssh_username,
        raw_data=pod,
        api_key=api_key
    )


def list_instances(api_key: Optional[str] = None) -> List[GPUInstance]:
    """List all user's instances"""
    query = """
    query {
        myself {
            pods {
                id
                name
                machineId
                imageName
                env
                machineId
                machine {
                    podHostId
                    gpuType {
                        displayName
                        manufacturer
                        memoryInGb
                    }
                }
                desiredStatus
                lastStatusChange
                gpuCount
                vcpuCount
                memoryInGb
                costPerHr
                containerDiskInGb
                volumeInGb
                ports
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
            }
        }
    }
    """
    
    try:
        data = _make_graphql_request(query, api_key=api_key)
        pods = data.get("myself", {}).get("pods", [])
        
        instances = []
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
        logger.error(f"Failed to list RunPod instances: {e}")
        return []


def get_user_balance(api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get user balance and spending information from RunPod"""
    query = """
    query {
        myself {
            clientBalance
            currentSpendPerHr
            clientLifetimeSpend
            spendLimit
            creditAlertThreshold
            referralEarnings
        }
    }
    """
    
    try:
        data = _make_graphql_request(query, api_key=api_key)
        user_data = data.get("myself")
        
        if not user_data:
            return None
        
        # Convert to more readable format
        balance_info = {
            "provider": "runpod",
            "current_balance": user_data.get("clientBalance", 0),
            "current_spend_per_hour": user_data.get("currentSpendPerHr", 0),
            "lifetime_spend": user_data.get("clientLifetimeSpend", 0),
            "spend_limit": user_data.get("spendLimit"),
            "credit_alert_threshold": user_data.get("creditAlertThreshold"),
            "referral_earnings": user_data.get("referralEarnings", 0),
            "raw_data": user_data
        }
        
        return balance_info
        
    except Exception as e:
        logger.error(f"Failed to get RunPod user balance: {e}")
        return None


def terminate_instance(instance_id: str, api_key: Optional[str] = None) -> bool:
    """Terminate a RunPod instance"""
    # Use simple schema - RunPod API might return different types
    mutation = """
    mutation podTerminate($input: PodTerminateInput!) {
        podTerminate(input: $input)
    }
    """
    
    variables = {"input": {"podId": instance_id}}
    
    try:
        data = _make_graphql_request(mutation, variables, api_key=api_key)
        result = data.get("podTerminate")
        logger.info(f"RunPod terminate response: {result} (type: {type(result)})")
        logger.info(f"Expected instance_id: {instance_id}")
        
        # RunPod's podTerminate API behavior:
        # - Returns null/None on SUCCESSFUL termination
        # - Would throw exception on failure (handled in except block)
        # This is the opposite of typical API patterns
        if result is None:
            logger.info("Terminate succeeded (RunPod returns null on success)")
            return True
        else:
            logger.info(f"Unexpected terminate response: {result}")
            # Non-null response might still indicate success
            return True
        
    except Exception as e:
        logger.error(f"Failed to terminate RunPod instance: {e}")
        return False


def wait_for_ssh_ready(instance, timeout: int = 600) -> bool:
    """RunPod-specific SSH waiting implementation"""
    # Tiger Style: Assert preconditions
    assert instance.provider == "runpod"
    assert instance.api_key
    assert timeout > 0

    # Wait for RUNNING status
    if not _wait_until_running(instance, timeout):
        return False

    # Wait for direct SSH (RunPod-specific)
    if not _wait_for_direct_ssh_assignment(instance, time.time(), timeout):
        return False

    # Test connectivity
    return _test_ssh_connectivity(instance)


def _wait_until_running(instance, timeout: int) -> bool:
    """Wait for instance to reach RUNNING status (≤70 lines)"""
    import time
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


def _wait_for_direct_ssh_assignment(instance, start_time: float, timeout: int) -> bool:
    """Wait for direct SSH (not proxy) - RunPod specific (≤70 lines)"""
    import time

    logger.info("Waiting for direct SSH (may take 5-10 min, proxy ignored)")
    next_log_time = start_time + 30  # Log at 30s, 60s, 90s, ...

    while time.time() - start_time < timeout:
        fresh = get_instance_details(instance.id, api_key=instance.api_key)

        if fresh and _has_direct_ssh(fresh):
            # Update instance with SSH details
            instance.public_ip = fresh.public_ip
            instance.ssh_port = fresh.ssh_port
            instance.ssh_username = fresh.ssh_username
            instance.status = fresh.status

            elapsed = int(time.time() - start_time)
            logger.info(f"Direct SSH: {instance.public_ip}:{instance.ssh_port} (took {elapsed}s)")
            return True

        # Log progress every 30s
        current_time = time.time()
        if current_time >= next_log_time:
            elapsed = int(current_time - start_time)
            _log_ssh_wait_status(instance, elapsed)
            next_log_time += 30  # Schedule next log

        time.sleep(10)

    elapsed_min = int((time.time() - start_time) / 60)
    logger.error(f"Timeout waiting for direct SSH after {elapsed_min} min")
    return False


def _has_direct_ssh(instance) -> bool:
    """Check if instance has direct SSH (not proxy)"""
    return (
        instance.public_ip and
        instance.ssh_port and
        instance.public_ip != "ssh.runpod.io"  # RunPod proxy hostname
    )


def _log_ssh_wait_status(instance, elapsed_seconds: int) -> None:
    """Log SSH wait status with timing"""
    if instance.public_ip and instance.ssh_port:
        if instance.public_ip == "ssh.runpod.io":
            logger.debug(f"Waiting for direct SSH (proxy now) - {elapsed_seconds}s")
        else:
            logger.debug(f"SSH details available - {elapsed_seconds}s")
    else:
        logger.debug(f"Waiting for SSH details - {elapsed_seconds}s")


def _test_ssh_connectivity(instance) -> bool:
    """Test SSH connectivity with echo command (≤70 lines)"""
    import time

    logger.info("Direct SSH ready! Waiting 30s for SSH daemon...")
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


def get_pod_logs(instance_id: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get pod logs and runtime information for debugging

    Args:
        instance_id: The RunPod pod ID
        api_key: RunPod API key

    Returns:
        Dictionary containing pod status, runtime info, and telemetry data
    """
    query = """
    query pod($input: PodFilter!) {
        pod(input: $input) {
            id
            name
            imageName
            desiredStatus
            lastStatusChange
            env
            runtime {
                uptimeInSeconds
                ports {
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                    type
                }
                gpus {
                    id
                    gpuUtilPercent
                    memoryUtilPercent
                }
            }
            latestTelemetry {
                cpuUtilization
                memoryUtilization
                averageGpuMetrics {
                    percentUtilization
                    memoryUtilization
                    temperature
                }
            }
        }
    }
    """

    variables = {"input": {"podId": instance_id}}

    try:
        data = _make_graphql_request(query, variables, api_key=api_key)
        pod_data = data.get("pod")

        if not pod_data:
            logger.error(f"No pod found with ID: {instance_id}")
            return None

        # Format the data for easier reading
        result = {
            "id": pod_data.get("id"),
            "name": pod_data.get("name"),
            "image": pod_data.get("imageName"),
            "status": pod_data.get("desiredStatus"),
            "last_status_change": pod_data.get("lastStatusChange"),
            "environment": pod_data.get("env", []),
            "runtime": pod_data.get("runtime"),
            "telemetry": pod_data.get("latestTelemetry"),
        }

        return result

    except Exception as e:
        logger.error(f"Failed to get RunPod pod logs: {e}")
        return None
