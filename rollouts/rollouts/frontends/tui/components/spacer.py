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
        # Return empty strings, not space-padded lines
        # This matches pi-mono behavior and avoids overwriting
        # background-colored padding from adjacent components
        return [""] * self._lines

    def invalidate(self) -> None:
        """No cached state to invalidate."""
        pass
