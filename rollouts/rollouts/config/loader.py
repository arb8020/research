"""Config loading utilities.

Tiger Style: Simple, explicit, no magic.
Load Python files as modules and extract the 'config' variable.
"""

import importlib.util
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def load_config_from_file(config_path: str | Path, config_name: str = "config") -> Any:
    """Load config from Python file.

    Tiger Style:
    - Explicit about what it does (loads Python file as module)
    - Returns the config object directly
    - No hidden state or global variables

    Standard pattern across all projects:
    1. Config files define a 'config' variable
    2. This function imports the file and returns that variable

    Args:
        config_path: Path to .py file containing config
        config_name: Name of variable to extract (default: "config")

    Returns:
        The config object from the file

    Raises:
        ValueError: If file can't be loaded or config_name doesn't exist

    Example:
        # configs/01_baseline.py contains:
        # config = MyConfig(param=value)

        # Load it:
        cfg = load_config_from_file("configs/01_baseline.py")
        # cfg is now the MyConfig instance

    Usage pattern in entrypoints:
        import sys
        from ..config import load_config_from_file

        config = load_config_from_file(sys.argv[1])
        # Use config...
    """
    config_path = Path(config_path)

    # Load Python file as module
    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract config variable
    if not hasattr(module, config_name):
        raise ValueError(f"Config file {config_path} must define '{config_name}' variable")

    return getattr(module, config_name)


def validate_config_protocol(config: Any, protocol_class: type) -> list[str]:
    """Validate that a config satisfies a protocol.

    Tiger Style: Explicit validation with list of errors.
    Returns empty list if valid, list of error messages if invalid.

    Args:
        config: Config instance to validate
        protocol_class: Protocol class to check against

    Returns:
        List of error messages (empty if valid)

    Example:
        from ..config import HasModelConfig, validate_config_protocol

        errors = validate_config_protocol(my_config, HasModelConfig)
        if errors:
            for err in errors:
                print(f"Config error: {err}")
            sys.exit(1)
    """
    errors = []

    # Check if config satisfies protocol
    if not isinstance(config, protocol_class):
        errors.append(f"Config does not satisfy {protocol_class.__name__} protocol")

        # Try to provide helpful error messages
        # Check for required methods
        if hasattr(protocol_class, "__annotations__"):
            for attr_name, _attr_type in protocol_class.__annotations__.items():
                if not hasattr(config, attr_name):
                    errors.append(f"Missing required attribute: {attr_name}")

    return errors
