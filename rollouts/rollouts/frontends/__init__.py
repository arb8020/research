"""Frontends for rollouts.

This package provides pluggable frontend implementations for the interactive
agent loop. All frontends implement the Frontend protocol, allowing them to
be used interchangeably.

Available frontends:
- NoneFrontend: Simple stdout printing (no TUI)
- TUIFrontend: Python-native terminal UI
- TextualFrontend: Rich Textual-based TUI (coming soon)
- IPCFrontend: Bridge for external processes (Go/TS)

Usage:
    from rollouts.frontends import run_interactive, NoneFrontend, TUIFrontend

    # Simple stdout mode
    frontend = NoneFrontend()
    states = await run_interactive(trajectory, endpoint, frontend=frontend)

    # Full TUI mode
    frontend = TUIFrontend(theme="dark")
    states = await run_interactive(trajectory, endpoint, frontend=frontend)
"""

from .json_frontend import JsonFrontend
from .none import NoneFrontend
from .protocol import Frontend, FrontendWithStatus
from .runner import InteractiveRunner, run_interactive
from .textual_frontend import TextualFrontend
from .tui_frontend import TUIFrontend

__all__ = [
    # Protocol
    "Frontend",
    "FrontendWithStatus",
    # Runner
    "InteractiveRunner",
    "run_interactive",
    # Implementations
    "JsonFrontend",
    "NoneFrontend",
    "TUIFrontend",
    "TextualFrontend",
]
