"""pytui - Elm-style Python TUI framework.

Zero deps. Differential rendering. Pure functions.

Quick start:
    from dataclasses import dataclass, replace
    from pytui import App, Cmd, KeyPress

    @dataclass(frozen=True)
    class Model:
        count: int = 0

    def update(model, msg):
        match msg:
            case KeyPress(key="q"):   return model, Cmd.quit()
            case KeyPress(key="j"):   return replace(model, count=model.count + 1), Cmd.none()
        return model, Cmd.none()

    def view(model, width, height):
        return [f"Count: {model.count}", "", "j: up  q: quit"]

    App(init=(Model(), Cmd.none()), update=update, view=view).run()
"""

# Elm core
from .app import App, Cmd, KeyPress, Resize, Sub

# Renderer
from .renderer import RenderState, diff_render

# Terminal
from .terminal import Terminal, TerminalProtocol

# Text utilities
from .text import (
    AnsiCode,
    AnsiCodeTracker,
    apply_background_to_line,
    extract_ansi_code,
    strip_terminal_control_sequences,
    truncate_to_width,
    visible_width,
    wrap_text_with_ansi,
)

# Spinner
from .spinner import Spinner

# Theme
from .theme import (
    DARK_THEME,
    MINIMAL_THEME,
    RESET,
    ROUNDED_THEME,
    SOFT_DARK_THEME,
    MinimalTheme,
    RoundedTheme,
    Theme,
    ToolDisplayMode,
    hex_to_bg,
    hex_to_fg,
    hex_to_rgb,
)

__all__ = [
    # Elm core
    "App",
    "Cmd",
    "Sub",
    "KeyPress",
    "Resize",
    # Terminal
    "Terminal",
    "TerminalProtocol",
    # Renderer
    "RenderState",
    "diff_render",
    # Text
    "visible_width",
    "truncate_to_width",
    "wrap_text_with_ansi",
    "apply_background_to_line",
    "extract_ansi_code",
    "strip_terminal_control_sequences",
    "AnsiCode",
    "AnsiCodeTracker",
    # Spinner
    "Spinner",
    # Theme
    "Theme",
    "MinimalTheme",
    "RoundedTheme",
    "DARK_THEME",
    "SOFT_DARK_THEME",
    "MINIMAL_THEME",
    "ROUNDED_THEME",
    "RESET",
    "ToolDisplayMode",
    "hex_to_fg",
    "hex_to_bg",
    "hex_to_rgb",
]
