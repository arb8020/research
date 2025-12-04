"""
Theme system for TUI - pi-mono inspired colors with true-color ANSI support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_to_fg(hex_color: str) -> str:
    """Convert hex color to ANSI true-color foreground escape."""
    r, g, b = hex_to_rgb(hex_color)
    return f"\x1b[38;2;{r};{g};{b}m"


def hex_to_bg(hex_color: str) -> str:
    """Convert hex color to ANSI true-color background escape."""
    r, g, b = hex_to_rgb(hex_color)
    return f"\x1b[48;2;{r};{g};{b}m"


RESET = "\x1b[0m"


@dataclass
class Theme:
    """TUI color theme - matches pi-mono dark theme."""

    # Core UI
    accent: str = "#8abeb7"
    border: str = "#808080"
    muted: str = "#666666"
    dim: str = "#505050"
    text: str = "#cccccc"

    # Message backgrounds
    user_message_bg: str = "#343541"
    tool_pending_bg: str = "#282832"
    tool_success_bg: str = "#283228"
    tool_error_bg: str = "#3c2828"

    # Markdown
    md_heading: str = "#f0c674"  # Golden
    md_link: str = "#81a2be"
    md_link_url: str = "#666666"
    md_code: str = "#ffff00"
    md_code_block: str = "#ffff00"
    md_code_border: str = "#666666"
    md_quote: str = "#cccccc"
    md_quote_border: str = "#00d7ff"
    md_hr: str = "#666666"
    md_list_bullet: str = "#00d7ff"

    # Thinking intensity levels (gray → purple gradient)
    thinking_minimal: str = "#6e6e6e"
    thinking_low: str = "#5f87af"
    thinking_medium: str = "#81a2be"
    thinking_high: str = "#b294bb"

    # Helper methods for common operations
    def fg(self, hex_color: str) -> Callable[[str], str]:
        """Return a function that applies foreground color to text."""
        prefix = hex_to_fg(hex_color)
        return lambda text: f"{prefix}{text}{RESET}"

    def bg(self, hex_color: str) -> Callable[[str], str]:
        """Return a function that applies background color to text."""
        prefix = hex_to_bg(hex_color)
        return lambda text: f"{prefix}{text}{RESET}"

    def fg_bg(self, fg_hex: str, bg_hex: str) -> Callable[[str], str]:
        """Return a function that applies both foreground and background."""
        fg_prefix = hex_to_fg(fg_hex)
        bg_prefix = hex_to_bg(bg_hex)
        return lambda text: f"{fg_prefix}{bg_prefix}{text}{RESET}"

    # Convenience color functions
    def accent_fg(self, text: str) -> str:
        return f"{hex_to_fg(self.accent)}{text}{RESET}"

    def muted_fg(self, text: str) -> str:
        return f"{hex_to_fg(self.muted)}{text}{RESET}"

    def border_fg(self, text: str) -> str:
        return f"{hex_to_fg(self.border)}{text}{RESET}"

    # Tool backgrounds
    def tool_pending_bg_fn(self, text: str) -> str:
        return f"{hex_to_bg(self.tool_pending_bg)}{text}{RESET}"

    def tool_success_bg_fn(self, text: str) -> str:
        return f"{hex_to_bg(self.tool_success_bg)}{text}{RESET}"

    def tool_error_bg_fn(self, text: str) -> str:
        return f"{hex_to_bg(self.tool_error_bg)}{text}{RESET}"

    # User message background
    def user_message_bg_fn(self, text: str) -> str:
        return f"{hex_to_bg(self.user_message_bg)}{text}{RESET}"

    # Thinking colors by intensity
    def thinking_fg(self, intensity: str = "medium") -> Callable[[str], str]:
        """Get thinking color function by intensity level."""
        colors = {
            "minimal": self.thinking_minimal,
            "low": self.thinking_low,
            "medium": self.thinking_medium,
            "high": self.thinking_high,
        }
        color = colors.get(intensity, self.thinking_medium)
        return self.fg(color)


# Default dark theme instance
DARK_THEME = Theme()
