"""
Vast.ai provider implementation (STUB)

This is a stub implementation for future Vast.ai support.
All methods raise NotImplementedError.
"""

import logging
from typing import Any, Dict, List, Optional

from ..types import GPUInstance, GPUOffer, ProvisionRequest

logger = logging.getLogger(__name__)


def search_gpu_offers(
    cuda_version: Optional[str] = None,
    manufacturer: Optional[str] = None,
    memory_gb: Optional[int] = None,
    container_disk_gb: Optional[int] = None,
    api_key: Optional[str] = None
) -> List[GPUOffer]:
    """Search for available GPU offers on Vast.ai

    Args:
        cuda_version: CUDA version filter
        manufacturer: GPU manufacturer filter
        memory_gb: Minimum memory in GB
        container_disk_gb: Minimum container disk in GB
        api_key: Vast.ai API key

    Returns:
        List of GPU offers

    Raises:
        NotImplementedError: Vast.ai provider not yet implemented
    """
    raise NotImplementedError(
        "Vast.ai provider not yet implemented. "
        "Currently only RunPod is supported."
    )


def provision_instance(
    request: ProvisionRequest,
    ssh_startup_script: Optional[str] = None,
    api_key: Optional[str] = None
) -> Optional[GPUInstance]:
    """Provision a GPU instance on Vast.ai

    Args:
        request: Provisioning request details
        ssh_startup_script: SSH key injection script
        api_key: Vast.ai API key

    Returns:
        Provisioned GPU instance

    Raises:
        NotImplementedError: Vast.ai provider not yet implemented
    """
    raise NotImplementedError(
        "Vast.ai provider not yet implemented. "
        "Currently only RunPod is supported."
    )


def get_instance_details(
    instance_id: str,
    api_key: Optional[str] = None
) -> Optional[GPUInstance]:
    """Get details of a specific Vast.ai instance

    Args:
        instance_id: Instance identifier
        api_key: Vast.ai API key

    Returns:
        GPU instance details

    Raises:
        NotImplementedError: Vast.ai provider not yet implemented
    """
    raise NotImplementedError(
        "Vast.ai provider not yet implemented. "
        "Currently only RunPod is supported."
    )


def list_instances(api_key: Optional[str] = None) -> List[GPUInstance]:
    """List all user's Vast.ai instances

    Args:
        api_key: Vast.ai API key

    Returns:
        List of GPU instances

    Raises:
        NotImplementedError: Vast.ai provider not yet implemented
    """
    raise NotImplementedError(
        "Vast.ai provider not yet implemented. "
        "Currently only RunPod is supported."
    )


def terminate_instance(
    instance_id: str,
    api_key: Optional[str] = None
) -> bool:
    """Terminate a Vast.ai instance

    Args:
        instance_id: Instance identifier
        api_key: Vast.ai API key

    Returns:
        True if termination successful

    Raises:
        NotImplementedError: Vast.ai provider not yet implemented
    """
    raise NotImplementedError(
        "Vast.ai provider not yet implemented. "
        "Currently only RunPod is supported."
    )
