"""
User message component - displays user input.
"""

from __future__ import annotations

from typing import List

from ..tui import Container
from .text import Text
from .spacer import Spacer


class UserMessage(Container):
    """Component that displays a user message."""

    def __init__(self, text: str, is_first: bool = False) -> None:
        """Initialize user message component.

        Args:
            text: User message text
            is_first: Whether this is the first user message (affects spacing)
        """
        super().__init__()

        # Add spacing before message (except first)
        if not is_first:
            self.add_child(Spacer(1))

        # Add user message text with styling
        # User messages typically have a different background or prefix
        user_text = Text(text, padding_x=1, padding_y=0)
        self.add_child(user_text)

