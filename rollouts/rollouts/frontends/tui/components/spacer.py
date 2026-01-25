"""
Spacer component - adds empty lines for vertical spacing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..tui import Component

if TYPE_CHECKING:
    from ..theme import Theme


class Spacer(Component):
    """Component that renders empty lines for vertical spacing.

    In compact display mode (theme.tool_display == "compact"), spacers
    between tools render as 0 lines to create a dense list.
    """

    def __init__(
        self,
        lines: int = 1,
        debug_label: str = "",
        debug_layout: bool = False,
        theme: Any | None = None,
        collapse_in_compact: bool = True,
    ) -> None:
        """Initialize spacer.

        Args:
            lines: Number of empty lines to render
            debug_label: Label to show in debug mode
            debug_layout: Whether to show debug labels
            theme: Theme for display mode awareness
            collapse_in_compact: If True, render 0 lines in compact mode
        """
        self._lines = lines
        self._debug_label = debug_label
        self._debug_layout = debug_layout
        self._theme = theme
        self._collapse_in_compact = collapse_in_compact

    def set_theme(self, theme: Any) -> None:
        """Update theme reference (for display mode changes)."""
        self._theme = theme

    def render(self, width: int) -> list[str]:
        """Render empty lines."""
        # In compact mode, collapse spacers to 0 lines
        if self._collapse_in_compact and self._theme:
            if hasattr(self._theme, "tool_display") and self._theme.tool_display == "compact":
                return []

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
