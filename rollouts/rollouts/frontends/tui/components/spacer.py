"""
Spacer component - adds empty lines for vertical spacing.
"""

from __future__ import annotations

from ..tui import Component


class Spacer(Component):
    """Component that renders empty lines for vertical spacing."""

    def __init__(self, lines: int = 1, debug_label: str = "", debug_layout: bool = False) -> None:
        self._lines = lines
        self._debug_label = debug_label
        self._debug_layout = debug_layout

    def render(self, width: int) -> list[str]:
        """Render empty lines."""
        # Return empty strings, not space-padded lines
        # This matches pi-mono behavior and avoids overwriting
        # background-colored padding from adjacent components
        if self._debug_layout and self._debug_label:
            # Show label in debug mode
            label = f"[{self._debug_label}]"
            return [label.ljust(width)[:width]] + [""] * (self._lines - 1)
        return [""] * self._lines

    def invalidate(self) -> None:
        """No cached state to invalidate."""
        pass
