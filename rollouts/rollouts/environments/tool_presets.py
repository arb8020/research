"""
Tool presets for coding environment.

Defines different tool configurations that can be swapped at runtime.
Inspired by pi-mono's --tools flag.
"""

from pathlib import Path
from typing import Protocol
from ..dtypes import Tool


class ToolPreset(Protocol):
    """Protocol for tool preset environments."""

    def get_name(self) -> str:
        """Return preset name identifier."""
        ...

    def get_description(self) -> str:
        """Return human-readable description."""
        ...

    def get_tools(self) -> list[Tool]:
        """Return tools for this preset."""
        ...


class LocalFilesystemEnvironmentFull:
    """Full coding environment with read, write, edit, bash tools."""

    def __init__(self, working_dir: Path):
        from .coding import LocalFilesystemEnvironment
        self._env = LocalFilesystemEnvironment(working_dir=working_dir)

    def get_name(self) -> str:
        return "coding"

    def get_description(self) -> str:
        return "Full access: read, write, edit, bash"

    def get_tools(self) -> list[Tool]:
        return self._env.get_tools()

    def __getattr__(self, name):
        """Delegate all other methods to the underlying environment."""
        return getattr(self._env, name)


class LocalFilesystemEnvironmentReadOnly:
    """Read-only coding environment with just read tool."""

    def __init__(self, working_dir: Path):
        from .coding import LocalFilesystemEnvironment
        self._env = LocalFilesystemEnvironment(working_dir=working_dir)

    def get_name(self) -> str:
        return "coding_readonly"

    def get_description(self) -> str:
        return "Read-only: read"

    def get_tools(self) -> list[Tool]:
        # Return only the read tool (first one)
        all_tools = self._env.get_tools()
        return [t for t in all_tools if t.function.name == "read"]

    def __getattr__(self, name):
        """Delegate all other methods to the underlying environment."""
        return getattr(self._env, name)


class LocalFilesystemEnvironmentNoWrite:
    """Coding environment without write tool (read, edit, bash only)."""

    def __init__(self, working_dir: Path):
        from .coding import LocalFilesystemEnvironment
        self._env = LocalFilesystemEnvironment(working_dir=working_dir)

    def get_name(self) -> str:
        return "coding_no_write"

    def get_description(self) -> str:
        return "No write: read, edit, bash (can modify existing files but not create new ones)"

    def get_tools(self) -> list[Tool]:
        all_tools = self._env.get_tools()
        return [t for t in all_tools if t.function.name != "write"]

    def __getattr__(self, name):
        """Delegate all other methods to the underlying environment."""
        return getattr(self._env, name)


# Registry of available presets
TOOL_PRESETS = {
    "full": LocalFilesystemEnvironmentFull,
    "readonly": LocalFilesystemEnvironmentReadOnly,
    "no-write": LocalFilesystemEnvironmentNoWrite,
}


def get_preset_names() -> list[str]:
    """Get list of available preset names."""
    return list(TOOL_PRESETS.keys())


def create_preset(name: str, working_dir: Path):
    """Create a tool preset environment.

    Args:
        name: Preset name (e.g., "full", "readonly", "no-write")
        working_dir: Working directory for the environment

    Returns:
        Environment instance with the specified tool preset

    Raises:
        ValueError: If preset name is unknown
    """
    if name not in TOOL_PRESETS:
        available = ", ".join(get_preset_names())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    preset_class = TOOL_PRESETS[name]
    return preset_class(working_dir=working_dir)
