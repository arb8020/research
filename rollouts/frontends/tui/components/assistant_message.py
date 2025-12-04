"""
Assistant message component - displays streaming assistant text and thinking blocks.
"""

from __future__ import annotations

from typing import List, Optional

from ..tui import Component, Container
from .markdown import Markdown, DefaultMarkdownTheme
from .spacer import Spacer
from .text import Text


class AssistantMessage(Component):
    """Component that renders assistant message with text and thinking blocks."""

    def __init__(self) -> None:
        """Initialize assistant message component."""
        self._content_container = Container()
        self._text_content = ""
        self._thinking_content = ""

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

        # Add spacer if we have any content
        has_content = (
            (self._text_content and self._text_content.strip())
            or (self._thinking_content and self._thinking_content.strip())
        )
        if has_content:
            self._content_container.add_child(Spacer(1))

        # Render thinking blocks first (if any)
        if self._thinking_content and self._thinking_content.strip():
            # Thinking in muted/dim color, italic style
            # Use a custom theme that applies muted color
            thinking_theme = _MutedMarkdownTheme()
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
                theme=DefaultMarkdownTheme(),
            )
            self._content_container.add_child(text_md)

    def render(self, width: int) -> List[str]:
        """Render assistant message."""
        return self._content_container.render(width)


class _MutedMarkdownTheme(DefaultMarkdownTheme):
    """Markdown theme with muted/dim colors for thinking blocks."""

    def heading(self, text: str) -> str:
        return f"\x1b[2m{text}\x1b[0m"  # Dim

    def link(self, text: str) -> str:
        return f"\x1b[2;4m{text}\x1b[0m"  # Dim underline

    def link_url(self, text: str) -> str:
        return f"\x1b[2m{text}\x1b[0m"  # Dim

    def code(self, text: str) -> str:
        return f"\x1b[2;33m{text}\x1b[0m"  # Dim yellow

    def code_block(self, text: str) -> str:
        return f"\x1b[2;33m{text}\x1b[0m"  # Dim yellow

    def code_block_border(self, text: str) -> str:
        return f"\x1b[2m{text}\x1b[0m"  # Dim

    def quote(self, text: str) -> str:
        return f"\x1b[2;3m{text}\x1b[0m"  # Dim italic

    def quote_border(self, text: str) -> str:
        return f"\x1b[2;36m{text}\x1b[0m"  # Dim cyan

    def hr(self, text: str) -> str:
        return f"\x1b[2m{text}\x1b[0m"  # Dim

    def list_bullet(self, text: str) -> str:
        return f"\x1b[2;36m{text}\x1b[0m"  # Dim cyan

    def bold(self, text: str) -> str:
        return f"\x1b[2;1m{text}\x1b[0m"  # Dim bold

    def italic(self, text: str) -> str:
        return f"\x1b[2;3m{text}\x1b[0m"  # Dim italic

    def strikethrough(self, text: str) -> str:
        return f"\x1b[2;9m{text}\x1b[0m"  # Dim strikethrough

    def underline(self, text: str) -> str:
        return f"\x1b[2;4m{text}\x1b[0m"  # Dim underline

