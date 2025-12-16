"""Preset loader utilities.

Loads agent presets from Python files following experiment_config.md patterns.
"""

import importlib.util
import sys
from pathlib import Path

from .base_preset import AgentPresetConfig


def load_preset(preset_name: str, preset_dir: Path | None = None) -> AgentPresetConfig:
    """Load an agent preset by name or from a file.

    Args:
        preset_name: Either:
            - Preset module name (e.g., "fast_coder_01_01" or "fast_coder")
            - Path to a preset file (e.g., "~/my-presets/custom.py")
        preset_dir: Directory to search for presets (default: rollouts/agent_presets/)

    Returns:
        AgentPresetConfig instance

    Raises:
        FileNotFoundError: If preset file not found
        AttributeError: If preset file doesn't export 'config'
        ValueError: If config is not an AgentPresetConfig

    Example:
        >>> preset = load_preset("fast_coder_01_01")
        >>> preset = load_preset("fast_coder")  # Finds fast_coder_XX_XX.py
        >>> preset = load_preset("~/my-presets/custom.py")
    """
    # Determine if preset_name is a file path
    preset_path = Path(preset_name).expanduser()

    if preset_path.exists() and preset_path.is_file():
        # Load from explicit file path
        return _load_preset_file(preset_path)

    # Otherwise treat as module name in preset_dir
    if preset_dir is None:
        # Default to rollouts/agent_presets/ directory
        preset_dir = Path(__file__).parent

    # Try to find preset file
    # First try exact match
    preset_file = preset_dir / f"{preset_name}.py"

    if preset_file.exists():
        return _load_preset_file(preset_file)

    # If not found, try fuzzy match (find first file starting with preset_name)
    # E.g., "fast_coder" matches "fast_coder_01_01.py"
    for file in preset_dir.glob(f"{preset_name}*.py"):
        if file.name == "__init__.py" or file.name in ("base_preset.py", "loader.py"):
            continue
        return _load_preset_file(file)

    raise FileNotFoundError(
        f"Preset '{preset_name}' not found in {preset_dir}. "
        f"Available presets: {', '.join(list_presets(preset_dir))}"
    )


def _load_preset_file(path: Path) -> AgentPresetConfig:
    """Load preset from Python file.

    Expects file to export 'config' variable of type AgentPresetConfig.
    """
    # Load module from file
    spec = importlib.util.spec_from_file_location(f"preset_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load preset from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Extract config
    if not hasattr(module, "config"):
        raise AttributeError(
            f"Preset file {path} must export 'config' variable.\n"
            f"Example: config = AgentPresetConfig(name='my_preset', ...)"
        )

    config = module.config

    if not isinstance(config, AgentPresetConfig):
        raise ValueError(f"Preset config must be AgentPresetConfig, got {type(config)}")

    return config


def list_presets(preset_dir: Path | None = None) -> list[str]:
    """List available preset names.

    Args:
        preset_dir: Directory to search (default: rollouts/agent_presets/)

    Returns:
        List of preset names (without .py extension)
    """
    if preset_dir is None:
        preset_dir = Path(__file__).parent

    presets = []
    for file in preset_dir.glob("*.py"):
        if file.name in ("__init__.py", "base_preset.py", "loader.py"):
            continue
        presets.append(file.stem)

    return sorted(presets)
