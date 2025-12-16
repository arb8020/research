"""
TUI - Terminal User Interface with differential rendering.

Ported from pi-mono/packages/tui.
"""

from .agent_renderer import AgentRenderer
from .components import (
    AssistantMessage,
    DefaultMarkdownTheme,
    Input,
    LoaderContainer,
    Markdown,
    Spacer,
    Text,
    ToolExecution,
    UserMessage,
)
from .terminal import ProcessTerminal, Terminal
from .theme import DARK_THEME, MINIMAL_THEME, SOFT_DARK_THEME, Theme
from .tui import TUI, Component, Container
from .utils import apply_background_to_line, truncate_to_width, visible_width, wrap_text_with_ansi

__all__ = [
    # Terminal
    "Terminal",
    "ProcessTerminal",
    # TUI core
    "Component",
    "Container",
    "TUI",
    # Themes
    "Theme",
    "DARK_THEME",
    "SOFT_DARK_THEME",
    "MINIMAL_THEME",
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
    "UserMessage",
    "AssistantMessage",
    "ToolExecution",
    "Input",
    "LoaderContainer",
    # Agent integration
    "AgentRenderer",
]
