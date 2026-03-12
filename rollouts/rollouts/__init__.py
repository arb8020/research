"""Top-level namespace for the rollouts subsystems.

Prefer explicit imports from the focused packages:
- ``rollouts.core`` for the shared schema
- ``rollouts.agents`` for live agent execution
- ``rollouts.eval`` for evaluation utilities
"""

from . import agents, core, eval

__all__ = [
    "agents",
    "core",
    "eval",
]

__version__ = "0.4.0"
