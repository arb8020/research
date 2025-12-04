"""
TUI - Terminal User Interface with differential rendering.

Ported from pi-mono/packages/tui.
"""

from .terminal import Terminal, ProcessTerminal
from .tui import Component, Container, TUI
from .utils import visible_width, wrap_text_with_ansi, truncate_to_width, apply_background_to_line
from .components import (
    Text,
    Spacer,
    Markdown,
    DefaultMarkdownTheme,
    Loader,
    UserMessage,
    AssistantMessage,
    ToolExecution,
    Input,
)
from .agent_renderer import AgentRenderer

__all__ = [
    # Terminal
    "Terminal",
    "ProcessTerminal",
    # TUI core
    "Component",
    "Container",
    "TUI",
    # Utils
    "visible_width",
    "wrap_text_with_ansi",
    "truncate_to_width",
    "apply_background_to_line",
    # Components
    "Text",
    "Spacer",
    "Markdown",
    "DefaultMarkdownTheme",
    "Loader",
    "UserMessage",
    "AssistantMessage",
    "ToolExecution",
    "Input",
    # Agent integration
    "AgentRenderer",
]
