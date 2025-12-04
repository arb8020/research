"""
Assistant message component - displays streaming assistant text and thinking blocks.
"""

from __future__ import annotations

from typing import List, Optional

from ..tui import Component, Container
from ..theme import Theme, DARK_THEME, hex_to_fg, hex_to_bg, RESET
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
            # Format thinking like a tool call with background
            thinking_text = f"thinking()\n\n{self._thinking_content.strip()}"
            thinking_component = Text(
                thinking_text,
                padding_x=2,
                padding_y=1,
                custom_bg_fn=lambda x: f"{hex_to_bg(self._theme.tool_pending_bg)}{x}{RESET}",
            )
            self._content_container.add_child(Spacer(1))
            self._content_container.add_child(thinking_component)

        # Render text content (if any)
        if self._text_content and self._text_content.strip():
            text_md = Markdown(
                self._text_content.strip(),
                padding_x=2,
                padding_y=0,
                theme=DefaultMarkdownTheme(self._theme),
            )
            self._content_container.add_child(text_md)

    def render(self, width: int) -> List[str]:
        """Render assistant message."""
        return self._content_container.render(width)

