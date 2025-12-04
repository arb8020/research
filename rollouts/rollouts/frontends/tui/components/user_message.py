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

        # Add spacing before message (except first)
        if not is_first:
            self.add_child(Spacer(1))

        # Add user message text with background color from theme
        user_text = Text(
            text,
            padding_x=1,
            padding_y=1,
            custom_bg_fn=self._theme.user_message_bg_fn,
        )
        self.add_child(user_text)

        # Add spacer AFTER user message - this acts as a buffer that can be
        # safely overwritten by streaming content without affecting the colored padding
        self.add_child(Spacer(1))

