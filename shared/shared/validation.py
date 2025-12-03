"""
Shared validation utilities for research components.

All validators follow Tiger Style: assert everything, fail fast.
Each validator returns the validated/normalized value on success.
"""

import logging
import os

logger = logging.getLogger(__name__)


def validate_ssh_key_path(path: str) -> str:
    """Validate SSH private key path (Tiger Style: assert everything).

    Args:
        path: SSH private key file path (can contain ~)

    Returns:
        Expanded absolute path to validated SSH key

    Raises:
        AssertionError: If validation fails

    Validates:
        - Path is non-empty string
        - File exists and is readable
        - File permissions are secure (warns if not)
        - File size is reasonable (<10KB)
    """
    # Assert input type and format
    assert isinstance(path, str), "ssh_key_path must be string"
    assert len(path) > 0, "ssh_key_path cannot be empty"

    # Expand and validate path
    expanded_path = os.path.expanduser(path)
    assert os.path.exists(expanded_path), \
        f"SSH private key not found: {expanded_path}"
    assert os.access(expanded_path, os.R_OK), \
        f"SSH private key not readable: {expanded_path}"

    # Check permissions (non-blocking warning)
    _check_ssh_key_permissions(expanded_path)

    # Assert reasonable file size
    stat_info = os.stat(expanded_path)
    assert stat_info.st_size < 10_000, \
        f"SSH key file suspiciously large ({stat_info.st_size} bytes): {expanded_path}"
    assert stat_info.st_size > 0, \
        f"SSH key file is empty: {expanded_path}"

    # Assert output invariant
    assert len(expanded_path) > 0, "Validated SSH key path"
    return expanded_path


def _check_ssh_key_permissions(key_path: str) -> None:
    """Check SSH key permissions and warn if insecure (Tiger Style helper).

    SSH keys should be 600 (owner read/write only).
    This is a warning, not a hard failure, since some systems may have different requirements.
    """
    stat_info = os.stat(key_path)
    perms = stat_info.st_mode & 0o777

    if perms & 0o077:  # Group or other has permissions
        logger.warning(
            f"SSH key has insecure permissions: {oct(perms)[-3:]}. "
            f"Recommend: chmod 600 {key_path}"
        )


def validate_timeout(timeout: int, min_value: int = 1, max_value: int = 3600) -> int:
    """Validate timeout parameter (Tiger Style: assert everything).

    Args:
        timeout: Timeout value in seconds
        min_value: Minimum allowed timeout (default: 1)
        max_value: Maximum allowed timeout (default: 3600 = 1 hour)

    Returns:
        Validated timeout value

    Raises:
        AssertionError: If validation fails
    """
    # Assert input type
    assert isinstance(timeout, int), f"timeout must be int, got {type(timeout)}"
    assert isinstance(min_value, int), "min_value must be int"
    assert isinstance(max_value, int), "max_value must be int"
    assert min_value < max_value, "min_value must be less than max_value"

    # Assert timeout range
    assert timeout >= min_value, \
        f"timeout must be >= {min_value}, got {timeout}"
    assert timeout <= max_value, \
        f"timeout must be <= {max_value}, got {timeout}"

    # Assert output invariant
    assert timeout > 0, "Validated timeout"
    return timeout


def validate_port(port: int) -> int:
    """Validate network port number (Tiger Style: assert everything).

    Args:
        port: Port number to validate

    Returns:
        Validated port number

    Raises:
        AssertionError: If validation fails
    """
    # Assert input type
    assert isinstance(port, int), f"port must be int, got {type(port)}"

    # Assert port range (1-65535 for TCP/UDP)
    assert 1 <= port <= 65535, \
        f"port must be 1-65535, got {port}"

    # Warn about privileged ports
    if port < 1024:
        logger.warning(f"Using privileged port {port} (< 1024)")

    # Assert output invariant
    assert port > 0, "Validated port"
    return port


def validate_hostname(hostname: str) -> str:
    """Validate hostname or IP address (Tiger Style: assert everything).

    Args:
        hostname: Hostname or IP address to validate

    Returns:
        Validated hostname

    Raises:
        AssertionError: If validation fails

    Note: This is a basic validation, not RFC-compliant DNS validation.
    """
    # Assert input type
    assert isinstance(hostname, str), f"hostname must be string, got {type(hostname)}"
    assert len(hostname) > 0, "hostname cannot be empty"
    assert len(hostname) <= 255, f"hostname too long: {len(hostname)} chars"

    # Assert no dangerous characters
    dangerous_chars = set(" ;|&`$(){}[]<>\"'\\")
    assert not any(c in dangerous_chars for c in hostname), \
        f"hostname contains dangerous characters: {hostname}"

    # Assert reasonable format (alphanumeric, dots, hyphens)
    valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
    assert all(c in valid_chars for c in hostname), \
        f"hostname contains invalid characters: {hostname}"

    # Assert output invariant
    assert len(hostname) > 0, "Validated hostname"
    return hostname


def validate_username(username: str) -> str:
    """Validate SSH username (Tiger Style: assert everything).

    Args:
        username: SSH username to validate

    Returns:
        Validated username

    Raises:
        AssertionError: If validation fails
    """
    # Assert input type
    assert isinstance(username, str), f"username must be string, got {type(username)}"
    assert len(username) > 0, "username cannot be empty"
    assert len(username) <= 32, f"username too long: {len(username)} chars"

    # Assert reasonable format (alphanumeric, underscore, hyphen)
    valid_chars = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    assert all(c in valid_chars for c in username.lower()), \
        f"username contains invalid characters: {username}"

    # Assert output invariant
    assert len(username) > 0, "Validated username"
    return username
