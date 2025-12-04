"""
Spacer component - adds empty lines for vertical spacing.
"""

from __future__ import annotations

from typing import List

from ..tui import Component


class Spacer(Component):
    """Component that renders empty lines for vertical spacing."""

    def __init__(self, lines: int = 1) -> None:
        self._lines = lines

    def render(self, width: int) -> List[str]:
        """Render empty lines."""
        empty_line = " " * width
        return [empty_line] * self._lines

    def invalidate(self) -> None:
        """No cached state to invalidate."""
        pass
