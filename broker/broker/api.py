"""
Main functional API for GPU cloud operations
"""

import logging
from typing import Any, Callable, List, Optional, Union, cast

from .providers import runpod, primeintellect, lambdalabs
from .query import QueryType
from .types import GPUInstance, GPUOffer, ProvisionRequest, ProvisionResult, ProvisionAttempt, ProviderModule

logger = logging.getLogger(__name__)

# Provider registry to eliminate duplication (Casey Muratori pattern)
# cast() asserts modules implement Protocol (structural typing, verified at usage)
PROVIDER_MODULES: dict[str, ProviderModule] = {
    "runpod": cast(ProviderModule, runpod),
    "primeintellect": cast(ProviderModule, primeintellect),
    "lambdalabs": cast(ProviderModule, lambdalabs),
}


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

    if provider is None or provider == "lambdalabs":
        api_key = credentials.get("lambdalabs") if credentials else None
        if api_key:  # Only search if we have credentials
            lambda_offers = lambdalabs.search_gpu_offers(cuda_version=cuda_version, manufacturer=manufacturer,
                                                         memory_gb=memory_gb, container_disk_gb=container_disk_gb,
                                                         gpu_count=gpu_count, api_key=api_key)
            offers.extend(lambda_offers)
    
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
    elif provider == "lambdalabs":
        instance = lambdalabs.get_instance_details(instance_id, api_key=api_key)
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
    elif provider == "lambdalabs":
        if lambdalabs.terminate_instance(instance_id, api_key=api_key):
            return True

    return False




def _normalize_query_input(
    query, gpu_type, max_price_per_hour, provider, cuda_version,
    manufacturer, gpu_count, sort, reverse, credentials, kwargs
) -> List[GPUOffer]:
    """Normalize query input to list of GPUOffers.

    Tiger Style: Extracted to keep create() under 70 lines.
    """
    # Handle different input types
    if isinstance(query, GPUOffer):
        # Single offer provided
        return [query]
    elif isinstance(query, list):
        # List of offers provided
        assert len(query) > 0, "Offer list cannot be empty"
        return query
    else:
        # Query object or None - search for suitable offers
        memory_gb = kwargs.get('memory_gb')
        container_disk_gb = kwargs.get('container_disk_gb')

        offers = search(
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
        return offers


def _try_provision_with_fallback(
    suitable_offers, n_offers, image, name, gpu_count,
    exposed_ports, enable_http_proxy, start_jupyter,
    jupyter_password, manufacturer, template_id, credentials, kwargs
) -> ProvisionResult:
    """Try provisioning from top N offers with automatic fallback.

    Tiger Style: Extracted to keep create() under 70 lines.
    Returns ProvisionResult with detailed attempt tracking.
    """
    attempts = []
    total_offers = min(len(suitable_offers), n_offers)

    for i, offer in enumerate(suitable_offers[:n_offers], 1):
        logger.info(f"Trying offer {i}/{total_offers}: {offer.gpu_type} at ${offer.price_per_hour:.3f}/hr")

        # Create provision request using this offer
        request = _build_provision_request(
            offer, image, name, gpu_count, exposed_ports,
            enable_http_proxy, start_jupyter, jupyter_password,
            manufacturer, template_id, kwargs
        )

        # Try provisioning from this offer
        attempt = _try_provision_from_offer(offer, request, request.ssh_startup_script, credentials)
        attempts.append(attempt)

        # Check if successful
        if attempt.error is None:
            # Success! Get the instance from the provider
            api_key = credentials.get(offer.provider)
            provider_module = PROVIDER_MODULES[offer.provider]
            instance = provider_module.provision_instance(request, request.ssh_startup_script, api_key=api_key)

            # Tiger Style: Assert postcondition
            assert instance is not None, "Successful attempt but no instance returned"

            logger.info(f"✅ Successfully provisioned GPU instance: {instance.id}")
            logger.info(f"   GPU: {instance.gpu_type} x{instance.gpu_count}")
            logger.info(f"   Provider: {offer.provider}")
            logger.info(f"   Expected price: ${offer.total_price(instance.gpu_count):.3f}/hr")

            return ProvisionResult(
                success=True,
                instance=instance,
                attempts=attempts
            )

    # All offers failed - categorize the failure
    return _categorize_failure(attempts, total_offers)


def _build_provision_request(
    offer, image, name, gpu_count, exposed_ports,
    enable_http_proxy, start_jupyter, jupyter_password,
    manufacturer, template_id, kwargs
) -> ProvisionRequest:
    """Build ProvisionRequest from offer and parameters.

    Tiger Style: Extracted to reduce complexity.
    """
    # For RunPod, use the full GPU ID from raw_data, not the display name
    gpu_type_id = offer.gpu_type
    if offer.provider == "runpod" and offer.raw_data:
        gpu_type_id = offer.raw_data.get("id", offer.gpu_type)
    elif offer.provider == "lambdalabs":
        # For Lambda Labs, use the offer ID which contains instance type and region
        gpu_type_id = offer.id

    return ProvisionRequest(
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
        template_id=template_id,
        **kwargs
    )


def _categorize_failure(attempts: List[ProvisionAttempt], total_offers: int) -> ProvisionResult:
    """Categorize provisioning failure based on attempt errors.

    Tiger Style: Extracted for clarity.
    """
    # Count error categories
    unavailable_count = sum(1 for a in attempts if a.error_category == "unavailable")
    credential_errors = sum(1 for a in attempts if a.error_category == "credentials")
    network_errors = sum(1 for a in attempts if a.error_category == "network")

    # Build error summary
    error_parts = []
    if unavailable_count == len(attempts):
        error_parts.append(f"All {len(attempts)} offers unavailable (capacity issue)")
    else:
        error_parts.append(f"Failed after trying {len(attempts)} offer(s)")

    if credential_errors > 0:
        error_parts.append(f"{credential_errors} credential error(s)")
    if network_errors > 0:
        error_parts.append(f"{network_errors} network error(s)")

    error_summary = "; ".join(error_parts)
    logger.error(error_summary)

    return ProvisionResult(
        success=False,
        attempts=attempts,
        error_summary=error_summary,
        all_unavailable=(unavailable_count == len(attempts)),
        credential_error=(credential_errors > 0),
        network_error=(network_errors > 0)
    )


def _try_provision_from_offer(
    offer: GPUOffer,
    request: ProvisionRequest,
    ssh_startup_script: Optional[str],
    credentials: dict
) -> ProvisionAttempt:
    """Try to provision from a single offer with explicit error categorization.

    Returns ProvisionAttempt with success/failure details.
    Tiger Style: Split from create() to keep functions under 70 lines.
    """
    # Tiger Style: Assert preconditions
    assert isinstance(offer, GPUOffer), f"offer must be GPUOffer, got {type(offer)}"
    assert offer.provider in PROVIDER_MODULES, f"Unsupported provider: {offer.provider}"
    assert isinstance(credentials, dict), f"credentials must be dict, got {type(credentials)}"

    # Validate API key exists for this provider
    api_key = credentials.get(offer.provider)
    assert api_key is not None, \
        f"Offer from {offer.provider} but no API key in credentials - search should have filtered this"
    assert isinstance(api_key, str) and len(api_key) > 0, \
        f"Invalid API key for {offer.provider}: must be non-empty string"

    # Get provider module
    provider_module = PROVIDER_MODULES[offer.provider]

    try:
        # Call provider's provision_instance
        instance = provider_module.provision_instance(request, ssh_startup_script, api_key=api_key)

        # Tiger Style: Assert postcondition - provider must return GPUInstance or None
        assert instance is None or isinstance(instance, GPUInstance), \
            f"{offer.provider}.provision_instance returned invalid type: {type(instance)}"

        if instance:
            # Success - validate instance has required fields
            assert instance.id, f"Instance missing ID: {instance}"
            assert instance.provider == offer.provider, \
                f"Instance provider mismatch: expected {offer.provider}, got {instance.provider}"

            return ProvisionAttempt(
                offer_id=offer.id,
                gpu_type=offer.gpu_type,
                provider=offer.provider,
                price_per_hour=offer.price_per_hour,
                error=None,
                error_category=None
            )
        else:
            # Provider returned None - offer unavailable (expected operating error)
            return ProvisionAttempt(
                offer_id=offer.id,
                gpu_type=offer.gpu_type,
                provider=offer.provider,
                price_per_hour=offer.price_per_hour,
                error="Offer unavailable",
                error_category="unavailable"
            )

    except ValueError as e:
        # Programmer error - invalid request parameters or credentials
        logger.warning(f"Credential/validation error for {offer.provider}: {e}")
        return ProvisionAttempt(
            offer_id=offer.id,
            gpu_type=offer.gpu_type,
            provider=offer.provider,
            price_per_hour=offer.price_per_hour,
            error=str(e),
            error_category="credentials"
        )

    except Exception as e:
        # Import here to avoid circular dependency
        import requests

        # Check if it's a network error
        if isinstance(e, (requests.Timeout, requests.RequestException)):
            logger.warning(f"Network error for {offer.provider}: {e}")
            return ProvisionAttempt(
                offer_id=offer.id,
                gpu_type=offer.gpu_type,
                provider=offer.provider,
                price_per_hour=offer.price_per_hour,
                error=str(e),
                error_category="network"
            )
        else:
            # Unknown error - log with full context
            logger.error(f"Unexpected error provisioning {offer.provider} {offer.gpu_type}: {e}", exc_info=True)
            return ProvisionAttempt(
                offer_id=offer.id,
                gpu_type=offer.gpu_type,
                provider=offer.provider,
                price_per_hour=offer.price_per_hour,
                error=str(e),
                error_category="unknown"
            )

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
    # RunPod-specific: Template support
    template_id: Optional[str] = None,
    # Offer selection parameters
    n_offers: int = 3,
    # API credentials
    credentials: Optional[dict] = None,
    **kwargs
) -> ProvisionResult:
    """
    Provision GPU using pandas-style query, search results, or specific offer.

    Returns ProvisionResult with detailed attempt tracking instead of Optional[GPUInstance].
    This allows callers to distinguish between different failure modes.

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
        n_offers: Number of offers to try from the list before giving up
        credentials: Dict mapping provider name to API key
        **kwargs: Additional provisioning parameters

    Returns:
        ProvisionResult with success/failure details and attempt log

    Examples:
        # Provision best value GPU (memory/price ratio)
        result = create(sort=lambda x: x.memory_gb/x.price_per_hour, reverse=True)
        if result.success:
            print(f"Provisioned: {result.instance.id}")
        else:
            print(f"Failed: {result.error_summary}")
    """
    # Tiger Style: Assert preconditions
    assert credentials is not None, "credentials dict required"
    assert isinstance(credentials, dict), f"credentials must be dict, got {type(credentials)}"
    assert len(credentials) > 0, "credentials dict cannot be empty"
    assert n_offers > 0, f"n_offers must be positive, got {n_offers}"
    assert gpu_count > 0, f"gpu_count must be positive, got {gpu_count}"

    # Normalize input to list of offers
    suitable_offers = _normalize_query_input(
        query, gpu_type, max_price_per_hour, provider, cuda_version,
        manufacturer, gpu_count, sort, reverse, credentials, kwargs
    )

    # Tiger Style: Assert postcondition from normalization
    assert isinstance(suitable_offers, list), f"_normalize_query_input must return list"

    # Check if search returned empty
    if not suitable_offers:
        return ProvisionResult(
            success=False,
            error_summary="No GPU offers found matching criteria",
            no_offers_found=True
        )

    # Try provisioning from top N offers
    return _try_provision_with_fallback(
        suitable_offers, n_offers, image, name, gpu_count,
        exposed_ports, enable_http_proxy, start_jupyter,
        jupyter_password, manufacturer, template_id, credentials, kwargs
    )


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

    if provider is None or provider == "lambdalabs":
        api_key = credentials.get("lambdalabs") if credentials else None
        if api_key:  # Only list if we have credentials
            lambda_instances = lambdalabs.list_instances(api_key=api_key)
            instances.extend(lambda_instances)

    return instances