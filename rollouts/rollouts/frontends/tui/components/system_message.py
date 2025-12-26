"""
System message component - displays the system prompt.
"""

from __future__ import annotations

from ..theme import DARK_THEME, Theme
from ..tui import Container
from .text import Text


class SystemMessage(Container):
    """Component that displays the system prompt."""

    def __init__(self, content: str, theme: Theme | None = None) -> None:
        """Initialize system message component.

        Args:
            content: System prompt content
            theme: Theme for styling
        """
        super().__init__()
        self._content = content
        self._theme = theme or DARK_THEME
        self._rebuild_display()

    def _rebuild_display(self) -> None:
        """Rebuild display from current state."""
        self.clear()

        # Apply muted color to the text content
        colored_text = self._theme.muted_fg(self._content)

        text = Text(
            colored_text,
            padding_x=2,
            padding_y=0,
            gutter_prefix="⚙ ",
        )
        self.add_child(text)
