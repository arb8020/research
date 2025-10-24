"""
Main functional API for GPU cloud operations
"""

import logging
from typing import Any, Callable, List, Optional, Union

from .providers import runpod, primeintellect
from .query import QueryType
from .types import GPUInstance, GPUOffer, ProvisionRequest

logger = logging.getLogger(__name__)


def search(
    query: Optional[QueryType] = None,
    # Legacy parameters for backward compatibility
    gpu_type: Optional[str] = None,
    max_price_per_hour: Optional[float] = None,
    provider: Optional[str] = None,
    cuda_version: Optional[str] = None,
    manufacturer: Optional[str] = None,
    # Resource requirements for accurate pricing
    memory_gb: Optional[int] = None,
    container_disk_gb: Optional[int] = None,
    gpu_count: int = 1,
    # New sorting parameters
    sort: Optional[Callable[[Any], Any]] = None,
    reverse: bool = False,
    # API credentials (dict mapping provider to API key)
    credentials: Optional[dict] = None
) -> List[GPUOffer]:
    """
    Search for available GPU offers across providers

    Args:
        query: Pandas-style query (e.g., gpus.gpu_type.contains("A100") & gpus.price_per_hour < 2.0)
        gpu_type: Legacy - Specific GPU type to search for
        max_price_per_hour: Legacy - Maximum price per hour filter
        provider: Specific provider to search (default: all providers)
        cuda_version: Filter by specific CUDA version (e.g., "12.0", "11.8")
        manufacturer: Filter by GPU manufacturer (e.g., "nvidia", "amd")
        memory_gb: Minimum memory in GB
        container_disk_gb: Container disk size in GB
        gpu_count: Number of GPUs to search for (default: 1). Affects pricing and availability.
        sort: Callable to extract sort key (e.g., lambda x: x.memory_gb/x.price_per_hour)
        reverse: Sort in descending order (default: False)
        credentials: Dict mapping provider name to API key (e.g., {"runpod": "key1"})

    Returns:
        List of available GPU offers
    """
    offers = []

    # Get all offers from providers
    # Only search providers that have credentials available
    if provider is None or provider == "runpod":
        api_key = credentials.get("runpod") if credentials else None
        if api_key:  # Only search if we have credentials
            runpod_offers = runpod.search_gpu_offers(cuda_version=cuda_version, manufacturer=manufacturer,
                                                     memory_gb=memory_gb, container_disk_gb=container_disk_gb,
                                                     gpu_count=gpu_count, api_key=api_key)
            offers.extend(runpod_offers)

    if provider is None or provider == "primeintellect":
        api_key = credentials.get("primeintellect") if credentials else None
        if api_key:  # Only search if we have credentials
            prime_offers = primeintellect.search_gpu_offers(cuda_version=cuda_version, manufacturer=manufacturer,
                                                            memory_gb=memory_gb, container_disk_gb=container_disk_gb,
                                                            gpu_count=gpu_count, api_key=api_key)
            offers.extend(prime_offers)
    
    # Apply pandas-style query if provided
    if query is not None:
        offers = [offer for offer in offers if query.evaluate(offer)]
    else:
        # Legacy filtering for backward compatibility
        if gpu_type:
            offers = [o for o in offers if gpu_type.lower() in o.gpu_type.lower()]
        
        if max_price_per_hour:
            offers = [o for o in offers if o.price_per_hour <= max_price_per_hour]
    
    # Sort by specified key or default to price
    if sort is not None:
        offers.sort(key=sort, reverse=reverse)
    else:
        # Default: sort by price (cheapest first)
        offers.sort(key=lambda x: x.price_per_hour, reverse=reverse)
    
    return offers


def get_instance(instance_id: str, provider: str, credentials: Optional[dict] = None) -> Optional[GPUInstance]:
    """
    Get details of a specific instance

    Args:
        instance_id: Instance ID
        provider: Provider name (e.g., 'runpod', 'vast')
        credentials: Dict mapping provider name to API key

    Returns:
        Instance details or None if not found
    """
    # Get API key for this provider
    api_key = credentials.get(provider) if credentials else None

    if provider == "runpod":
        instance = runpod.get_instance_details(instance_id, api_key=api_key)
        if instance:
            return instance
    elif provider == "primeintellect":
        instance = primeintellect.get_instance_details(instance_id, api_key=api_key)
        if instance:
            return instance

    return None


def terminate_instance(instance_id: str, provider: str, credentials: Optional[dict] = None) -> bool:
    """
    Terminate a GPU instance

    Args:
        instance_id: Instance ID
        provider: Provider name (e.g., 'runpod', 'vast')
        credentials: Dict mapping provider name to API key

    Returns:
        True if termination was successful
    """
    # Get API key for this provider
    api_key = credentials.get(provider) if credentials else None

    if provider == "runpod":
        if runpod.terminate_instance(instance_id, api_key=api_key):
            return True
    elif provider == "primeintellect":
        if primeintellect.terminate_instance(instance_id, api_key=api_key):
            return True

    return False


def create(
    query: Union[QueryType, List[GPUOffer], GPUOffer] = None,
    image: str = "runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204",
    name: Optional[str] = None,
    # Search parameters for when query is a filter
    gpu_type: Optional[str] = None,
    max_price_per_hour: Optional[float] = None,
    provider: Optional[str] = None,
    cuda_version: Optional[str] = None,
    manufacturer: Optional[str] = None,
    gpu_count: int = 1,
    sort: Optional[Callable[[Any], Any]] = None,
    reverse: bool = False,
    # Port exposure configuration
    exposed_ports: Optional[List[int]] = None,
    enable_http_proxy: bool = True,
    # Jupyter configuration
    start_jupyter: bool = False,
    jupyter_password: Optional[str] = None,
    # Offer selection parameters
    n_offers: int = 3,
    # API credentials
    credentials: Optional[dict] = None,
    **kwargs
) -> Optional[GPUInstance]:
    """
    Provision GPU using pandas-style query, search results, or specific offer
    
    Args:
        query: Query object, list of offers, or single offer (if None, searches all)
        image: Docker image to use
        name: Optional instance name
        gpu_type: Filter by GPU type (used if query is None)
        max_price_per_hour: Filter by max price (used if query is None)
        provider: Filter by provider (used if query is None)
        cuda_version: Filter by CUDA version (used if query is None)
        manufacturer: Filter by GPU manufacturer (used if query is None)
        sort: Sort key function (used if query is None)
        reverse: Sort order (used if query is None)
        n_offers: Number of offers to try from the list before giving up.
                 When a single offer is provided, wraps it in a list (tries once).
                 When a list is provided, tries offers in order until one succeeds.
        **kwargs: Additional provisioning parameters

    Returns:
        Provisioned GPU instance or None if failed

    Examples:
        # Provision best value GPU (memory/price ratio)
        create(sort=lambda x: x.memory_gb/x.price_per_hour, reverse=True)

        # Provision cheapest A100
        create(gpus.gpu_type.contains("A100"))

        # Try top 5 cheapest RTX 4090s (falls back if first is unavailable)
        offers = search(gpu_type="RTX 4090")
        create(offers[:5], n_offers=5)
    """
    # Handle different input types
    if isinstance(query, GPUOffer):
        # Single offer provided
        suitable_offers = [query]
    elif isinstance(query, list):
        # List of offers provided
        if not query:
            raise ValueError("No GPU offers provided")
        suitable_offers = query
    else:
        # Query object or None - search for suitable offers
        # Extract memory and disk from kwargs to pass to search
        memory_gb = kwargs.get('memory_gb')
        container_disk_gb = kwargs.get('container_disk_gb')

        suitable_offers = search(
            query=query,
            gpu_type=gpu_type,
            max_price_per_hour=max_price_per_hour,
            provider=provider,
            cuda_version=cuda_version,
            manufacturer=manufacturer,
            memory_gb=memory_gb,
            container_disk_gb=container_disk_gb,
            gpu_count=gpu_count,
            sort=sort,
            reverse=reverse,
            credentials=credentials
        )
        if not suitable_offers:
            logger.info("No GPUs found matching criteria")
            return None
    
    # Try to provision from the top offers
    last_error = None
    offers_tried = 0
    total_offers = min(len(suitable_offers), n_offers)

    for offer in suitable_offers[:n_offers]:
        offers_tried += 1
        logger.info(f"Trying offer {offers_tried}/{total_offers}: {offer.gpu_type} at ${offer.price_per_hour:.3f}/hr")
        
        try:
            # Create provision request using this offer
            # For RunPod, use the full GPU ID from raw_data, not the display name
            gpu_type_id = offer.gpu_type
            if offer.provider == "runpod" and offer.raw_data:
                gpu_type_id = offer.raw_data.get("id", offer.gpu_type)
            
            request = ProvisionRequest(
                gpu_type=gpu_type_id,
                gpu_count=gpu_count,
                image=image,
                name=name,
                provider=offer.provider,
                spot_instance=offer.spot,
                exposed_ports=exposed_ports,
                enable_http_proxy=enable_http_proxy,
                start_jupyter=start_jupyter,
                jupyter_password=jupyter_password,
                manufacturer=manufacturer,
                **kwargs
            )
            
            # Provision using the appropriate provider
            if offer.provider == "runpod":
                api_key = credentials.get("runpod") if credentials else None
                instance = runpod.provision_instance(request, request.ssh_startup_script, api_key=api_key)
                if instance:
                    logger.info(f"✅ Successfully provisioned GPU instance: {instance.id}")
                    logger.info(f"   GPU: {instance.gpu_type} x{instance.gpu_count}")
                    logger.info(f"   Provider: {offer.provider}")
                    logger.info(f"   Expected price: ${offer.total_price(instance.gpu_count):.3f}/hr")
                    return instance
                else:
                    logger.warning(f"Provisioning returned None for {offer.gpu_type}")
                    last_error = "Provisioning returned None"
            elif offer.provider == "primeintellect":
                api_key = credentials.get("primeintellect") if credentials else None
                instance = primeintellect.provision_instance(request, request.ssh_startup_script, api_key=api_key)
                if instance:
                    logger.info(f"✅ Successfully provisioned GPU instance: {instance.id}")
                    logger.info(f"   GPU: {instance.gpu_type} x{instance.gpu_count}")
                    logger.info(f"   Provider: {offer.provider}")
                    logger.info(f"   Expected price: ${offer.total_price(instance.gpu_count):.3f}/hr")
                    return instance
                else:
                    logger.warning(f"Provisioning returned None for {offer.gpu_type}")
                    last_error = "Provisioning returned None"
            else:
                raise ValueError(f"Unsupported provider: {offer.provider}")
                
        except Exception as e:
            logger.warning(f"Provisioning failed for {offer.gpu_type}: {e}")
            last_error = str(e)
            continue
    
    # All offers failed
    logger.error(f"Failed to provision after trying {offers_tried} offer(s)")
    if last_error:
        logger.error(f"Last error: {last_error}")
    return None


def list_instances(provider: Optional[str] = None, credentials: Optional[dict] = None) -> List[GPUInstance]:
    """
    List all user's instances across providers

    Args:
        provider: Specific provider to list from (default: all providers)
        credentials: Dict mapping provider name to API key

    Returns:
        List of user's GPU instances
    """
    instances = []

    # Get instances from providers
    # Only list instances for providers that have credentials available
    if provider is None or provider == "runpod":
        api_key = credentials.get("runpod") if credentials else None
        if api_key:  # Only list if we have credentials
            runpod_instances = runpod.list_instances(api_key=api_key)
            instances.extend(runpod_instances)

    if provider is None or provider == "primeintellect":
        api_key = credentials.get("primeintellect") if credentials else None
        if api_key:  # Only list if we have credentials
            prime_instances = primeintellect.list_instances(api_key=api_key)
            instances.extend(prime_instances)

    return instances