"""
Color theme for diff viewer.
"""

from __future__ import annotations

from dataclasses import dataclass


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_to_fg(hex_color: str) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"\x1b[38;2;{r};{g};{b}m"


def hex_to_bg(hex_color: str) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"\x1b[48;2;{r};{g};{b}m"


RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
REVERSE = "\x1b[7m"


@dataclass
class Theme:
    """Diff viewer color theme."""

    # UI chrome
    border: str = "#505050"
    text: str = "#cccccc"
    muted: str = "#666666"
    accent: str = "#8abeb7"

    # File list
    selected_bg: str = "#3a3a3a"
    selected_fg: str = "#ffffff"

    # Diff colors - backgrounds for line highlighting
    added_bg: str = "#2d4a2d"
    removed_bg: str = "#4a2d2d"
    added_fg: str = "#98c379"
    removed_fg: str = "#e06c75"

    # Word-level diff highlighting (brighter)
    added_word_bg: str = "#3d6a3d"
    removed_word_bg: str = "#6a3d3d"

    # Hunk header
    hunk_header_fg: str = "#61afef"
    hunk_header_bg: str = "#2c323c"

    # Line numbers
    line_number_fg: str = "#5c6370"

    # Status bar
    status_bg: str = "#3e4451"
    status_fg: str = "#abb2bf"

    def fg(self, hex_color: str) -> str:
        return hex_to_fg(hex_color)

    def bg(self, hex_color: str) -> str:
        return hex_to_bg(hex_color)

    def style_added_line(self, text: str) -> str:
        return f"{hex_to_bg(self.added_bg)}{hex_to_fg(self.added_fg)}{text}{RESET}"

    def style_removed_line(self, text: str) -> str:
        return f"{hex_to_bg(self.removed_bg)}{hex_to_fg(self.removed_fg)}{text}{RESET}"

    def style_hunk_header(self, text: str) -> str:
        return f"{hex_to_bg(self.hunk_header_bg)}{hex_to_fg(self.hunk_header_fg)}{text}{RESET}"

    def style_line_number(self, text: str) -> str:
        return f"{hex_to_fg(self.line_number_fg)}{text}{RESET}"

    def style_selected(self, text: str) -> str:
        return f"{hex_to_bg(self.selected_bg)}{hex_to_fg(self.selected_fg)}{text}{RESET}"

    def style_status(self, text: str) -> str:
        return f"{hex_to_bg(self.status_bg)}{hex_to_fg(self.status_fg)}{text}{RESET}"

    def style_muted(self, text: str) -> str:
        return f"{hex_to_fg(self.muted)}{text}{RESET}"

    def style_accent(self, text: str) -> str:
        return f"{hex_to_fg(self.accent)}{text}{RESET}"


DEFAULT_THEME = Theme()
