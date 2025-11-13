"""
Shared utilities and foundations for research components.

This module contains common functionality that is used across
broker, bifrost, and other components to reduce code duplication
and ensure consistent behavior.
"""

__version__ = "0.1.0"

# Export main SSH foundation classes
from .ssh_foundation import SSHConnectionInfo, UniversalSSHClient, secure_temp_ssh_key

# Export logging utilities
from .logging_config import setup_logging
from .print_interceptor import intercept_prints, PrintToLogger

__all__ = [
    # SSH
    "SSHConnectionInfo",
    "UniversalSSHClient",
    "secure_temp_ssh_key",
    # Logging
    "setup_logging",
    "intercept_prints",
    "PrintToLogger",
]