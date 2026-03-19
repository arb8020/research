"""Top-level namespace for the rollouts subsystems.

Prefer explicit imports from the focused packages:
- ``rollouts.core`` for the shared schema
- ``rollouts.agents`` for live agent execution
- ``rollouts.eval`` for evaluation utilities
"""

from importlib import import_module
from typing import Any

__all__ = [
    "agents",
    "core",
    "eval",
]

__version__ = "0.4.0"


def __getattr__(name: str) -> Any:
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
