"""
User message component - displays user input.
"""

from __future__ import annotations

from typing import List, Optional

from ..tui import Container
from ..theme import Theme, DARK_THEME
from .text import Text
from .spacer import Spacer


class UserMessage(Container):
    """Component that displays a user message."""

    def __init__(self, text: str, is_first: bool = False, theme: Optional[Theme] = None) -> None:
        """Initialize user message component.

        Args:
            text: User message text
            is_first: Whether this is the first user message (affects spacing)
            theme: Theme for styling
        """
        super().__init__()
        self._theme = theme or DARK_THEME

        # Add user message text with > prefix and background color from theme
        prefixed_text = f"> {text}"
        user_text = Text(
            prefixed_text,
            padding_x=2,
            padding_y=0,
            custom_bg_fn=self._theme.user_message_bg_fn,
            theme=self._theme,
        )
        self.add_child(user_text)

