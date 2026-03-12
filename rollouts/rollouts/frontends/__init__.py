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
    from ..frontends import run_interactive, NoneFrontend, TUIFrontend

    # Simple stdout mode
    frontend = NoneFrontend()
    states = await run_interactive(trajectory, endpoint, frontend=frontend)

    # Full TUI mode
    frontend = TUIFrontend(theme="dark")
    states = await run_interactive(trajectory, endpoint, frontend=frontend)
"""

from .headless_json import HeadlessJsonFrontend
from .json_frontend import JsonFrontend
from .minimal import MinimalFrontend
from .none import NoneFrontend
from .protocol import (
    Frontend,
    FrontendWithStatus,
    InputExit,
    InputInterrupt,
    InputResult,
    SlashCommand,
    UserMessage,
)
from .runner import InteractiveRunner, RunnerConfig, run_interactive
from .textual_frontend import TextualFrontend
from .tui_frontend import TUIFrontend

__all__ = [
    # Protocol
    "Frontend",
    "FrontendWithStatus",
    "InputResult",
    "UserMessage",
    "SlashCommand",
    "InputExit",
    "InputInterrupt",
    # Runner
    "InteractiveRunner",
    "RunnerConfig",
    "run_interactive",
    # Implementations
    "HeadlessJsonFrontend",
    "JsonFrontend",
    "MinimalFrontend",
    "NoneFrontend",
    "TUIFrontend",
    "TextualFrontend",
]
