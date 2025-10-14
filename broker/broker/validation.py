"""
Validation utilities for GPU broker operations.

All validators follow Tiger Style: assert everything, fail fast.
Each validator returns the validated/normalized value on success.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def validate_credentials(credentials: Dict[str, str]) -> Dict[str, str]:
    """Validate provider credentials dictionary (Tiger Style: assert everything).

    Args:
        credentials: Dict mapping provider name to API key

    Returns:
        Validated credentials dict

    Raises:
        AssertionError: If validation fails

    Validates:
        - Credentials is non-empty dict
        - All keys are valid provider names
        - All values are non-empty strings
        - API keys meet minimum length requirements
    """
    # Assert input type
    assert isinstance(credentials, dict), "credentials must be dict"
    assert len(credentials) > 0, "credentials cannot be empty"

    # Validate each provider credential
    valid_providers = {"runpod", "vast", "lambda"}
    for provider, api_key in credentials.items():
        # Assert provider is known
        assert isinstance(provider, str), f"Provider name must be string, got {type(provider)}"
        assert provider in valid_providers, \
            f"Unknown provider: {provider}. Valid providers: {valid_providers}"

        # Assert API key format
        assert isinstance(api_key, str), \
            f"API key for {provider} must be string, got {type(api_key)}"
        assert len(api_key) > 10, \
            f"{provider} API key appears invalid (too short: {len(api_key)} chars)"

    # Assert output invariant
    assert len(credentials) > 0, "Validated credentials"
    return credentials


def validate_gpu_count(gpu_count: int) -> int:
    """Validate GPU count parameter (Tiger Style: assert everything).

    Args:
        gpu_count: Number of GPUs to provision

    Returns:
        Validated GPU count

    Raises:
        AssertionError: If validation fails
    """
    # Assert input type
    assert isinstance(gpu_count, int), f"gpu_count must be int, got {type(gpu_count)}"

    # Assert reasonable range
    assert 1 <= gpu_count <= 8, \
        f"gpu_count must be 1-8, got {gpu_count}"

    # Warn about multi-GPU costs
    if gpu_count > 1:
        logger.info(f"Requesting {gpu_count} GPUs - cost will scale linearly")

    # Assert output invariant
    assert gpu_count > 0, "Validated gpu_count"
    return gpu_count


def validate_price(price: float, max_price: float = 10.0) -> float:
    """Validate price parameter (Tiger Style: assert everything).

    Args:
        price: Price per hour in USD
        max_price: Maximum reasonable price (default: $10/hr)

    Returns:
        Validated price

    Raises:
        AssertionError: If validation fails
    """
    # Assert input type
    assert isinstance(price, (int, float)), f"price must be numeric, got {type(price)}"
    assert isinstance(max_price, (int, float)), "max_price must be numeric"

    # Convert to float
    price = float(price)
    max_price = float(max_price)

    # Assert reasonable range
    assert price >= 0, f"price cannot be negative, got {price}"
    assert price <= max_price, \
        f"price exceeds maximum ${max_price}/hr, got ${price}/hr"

    # Warn about expensive instances
    if price > 5.0:
        logger.warning(f"High cost GPU: ${price:.2f}/hr")

    # Assert output invariant
    assert price >= 0, "Validated price"
    return price


def validate_memory_gb(memory_gb: int, min_memory: int = 1, max_memory: int = 512) -> int:
    """Validate memory size in GB (Tiger Style: assert everything).

    Args:
        memory_gb: Memory size in GB
        min_memory: Minimum memory (default: 1GB)
        max_memory: Maximum memory (default: 512GB)

    Returns:
        Validated memory size

    Raises:
        AssertionError: If validation fails
    """
    # Assert input type
    assert isinstance(memory_gb, int), f"memory_gb must be int, got {type(memory_gb)}"
    assert isinstance(min_memory, int), "min_memory must be int"
    assert isinstance(max_memory, int), "max_memory must be int"

    # Assert reasonable range
    assert min_memory <= memory_gb <= max_memory, \
        f"memory_gb must be {min_memory}-{max_memory}GB, got {memory_gb}GB"

    # Assert output invariant
    assert memory_gb > 0, "Validated memory_gb"
    return memory_gb


def validate_disk_gb(disk_gb: int, min_disk: int = 0, max_disk: int = 2000) -> int:
    """Validate disk size in GB (Tiger Style: assert everything).

    Args:
        disk_gb: Disk size in GB
        min_disk: Minimum disk (default: 0GB = no volume)
        max_disk: Maximum disk (default: 2000GB = 2TB)

    Returns:
        Validated disk size

    Raises:
        AssertionError: If validation fails
    """
    # Assert input type
    assert isinstance(disk_gb, int), f"disk_gb must be int, got {type(disk_gb)}"
    assert isinstance(min_disk, int), "min_disk must be int"
    assert isinstance(max_disk, int), "max_disk must be int"

    # Assert reasonable range
    assert min_disk <= disk_gb <= max_disk, \
        f"disk_gb must be {min_disk}-{max_disk}GB, got {disk_gb}GB"

    # Warn about large volumes
    if disk_gb > 500:
        logger.warning(f"Large volume requested: {disk_gb}GB")

    # Assert output invariant
    assert disk_gb >= 0, "Validated disk_gb"
    return disk_gb


def validate_instance_name(name: str) -> str:
    """Validate instance name (Tiger Style: assert everything).

    Args:
        name: Instance name to validate

    Returns:
        Validated instance name

    Raises:
        AssertionError: If validation fails
    """
    # Assert input type
    assert isinstance(name, str), f"name must be string, got {type(name)}"
    assert len(name) > 0, "name cannot be empty"
    assert len(name) <= 64, f"name too long: {len(name)} chars (max 64)"

    # Assert reasonable format (alphanumeric, hyphens, underscores)
    valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    assert all(c in valid_chars for c in name), \
        f"name contains invalid characters: {name}"

    # Assert doesn't start/end with hyphen
    assert not name.startswith("-") and not name.endswith("-"), \
        f"name cannot start/end with hyphen: {name}"

    # Assert output invariant
    assert len(name) > 0, "Validated instance name"
    return name
