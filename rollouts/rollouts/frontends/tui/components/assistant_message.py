"""
Assistant message component - displays streaming assistant text and thinking blocks.
"""

from __future__ import annotations

from typing import List, Optional

from ..tui import Component, Container
from ..theme import Theme, DARK_THEME, hex_to_fg, RESET
from .markdown import Markdown, DefaultMarkdownTheme
from .spacer import Spacer
from .text import Text


class AssistantMessage(Component):
    """Component that renders assistant message with text and thinking blocks."""

    def __init__(self, theme: Optional[Theme] = None) -> None:
        """Initialize assistant message component."""
        self._theme = theme or DARK_THEME
        self._content_container = Container()
        self._text_content = ""
        self._thinking_content = ""
        self._thinking_intensity = "medium"  # minimal, low, medium, high

    def append_text(self, delta: str) -> None:
        """Append text delta to current text content."""
        self._text_content += delta
        self._rebuild_content()

    def append_thinking(self, delta: str) -> None:
        """Append thinking delta to current thinking content."""
        self._thinking_content += delta
        self._rebuild_content()

    def set_text(self, text: str) -> None:
        """Set complete text content."""
        self._text_content = text
        self._rebuild_content()

    def set_thinking(self, thinking: str) -> None:
        """Set complete thinking content."""
        self._thinking_content = thinking
        self._rebuild_content()

    def set_thinking_intensity(self, intensity: str) -> None:
        """Set thinking intensity level (minimal, low, medium, high)."""
        self._thinking_intensity = intensity
        self._rebuild_content()

    def clear(self) -> None:
        """Clear all content."""
        self._text_content = ""
        self._thinking_content = ""
        self._content_container.clear()

    def invalidate(self) -> None:
        """Invalidate content container."""
        self._content_container.invalidate()

    def _rebuild_content(self) -> None:
        """Rebuild content container from current text and thinking."""
        self._content_container.clear()

        # Note: We don't add a spacer here - the previous component (UserMessage)
        # already has padding_y=1 which provides spacing. Adding a spacer here
        # would overwrite that colored padding during differential re-rendering.

        # Render thinking blocks first (if any)
        if self._thinking_content and self._thinking_content.strip():
            # Thinking with intensity-based color
            thinking_theme = _ThinkingMarkdownTheme(self._theme, self._thinking_intensity)
            thinking_md = Markdown(
                self._thinking_content.strip(),
                padding_x=1,
                padding_y=0,
                theme=thinking_theme,
            )
            self._content_container.add_child(thinking_md)
            self._content_container.add_child(Spacer(1))

        # Render text content (if any)
        if self._text_content and self._text_content.strip():
            text_md = Markdown(
                self._text_content.strip(),
                padding_x=1,
                padding_y=0,
                theme=DefaultMarkdownTheme(self._theme),
            )
            self._content_container.add_child(text_md)

    def render(self, width: int) -> List[str]:
        """Render assistant message."""
        return self._content_container.render(width)


class _ThinkingMarkdownTheme(DefaultMarkdownTheme):
    """Markdown theme for thinking blocks with intensity-based colors."""

    def __init__(self, theme: Theme, intensity: str = "medium") -> None:
        super().__init__(theme)
        # Get thinking color based on intensity
        colors = {
            "minimal": theme.thinking_minimal,
            "low": theme.thinking_low,
            "medium": theme.thinking_medium,
            "high": theme.thinking_high,
        }
        self._thinking_color = colors.get(intensity, theme.thinking_medium)
        self._color_prefix = hex_to_fg(self._thinking_color)

    def heading(self, text: str) -> str:
        return f"\x1b[3m{self._color_prefix}{text}{RESET}"  # Italic + thinking color

    def link(self, text: str) -> str:
        return f"\x1b[4m{self._color_prefix}{text}{RESET}"

    def link_url(self, text: str) -> str:
        return f"{self._color_prefix}{text}{RESET}"

    def code(self, text: str) -> str:
        return f"{self._color_prefix}{text}{RESET}"

    def code_block(self, text: str) -> str:
        return f"{self._color_prefix}{text}{RESET}"

    def code_block_border(self, text: str) -> str:
        return f"{self._color_prefix}{text}{RESET}"

    def quote(self, text: str) -> str:
        return f"\x1b[3m{self._color_prefix}{text}{RESET}"

    def quote_border(self, text: str) -> str:
        return f"{self._color_prefix}{text}{RESET}"

    def hr(self, text: str) -> str:
        return f"{self._color_prefix}{text}{RESET}"

    def list_bullet(self, text: str) -> str:
        return f"{self._color_prefix}{text}{RESET}"

    def bold(self, text: str) -> str:
        return f"\x1b[1m{self._color_prefix}{text}{RESET}"

    def italic(self, text: str) -> str:
        return f"\x1b[3m{self._color_prefix}{text}{RESET}"

    def strikethrough(self, text: str) -> str:
        return f"\x1b[9m{self._color_prefix}{text}{RESET}"

    def underline(self, text: str) -> str:
        return f"\x1b[4m{self._color_prefix}{text}{RESET}"

