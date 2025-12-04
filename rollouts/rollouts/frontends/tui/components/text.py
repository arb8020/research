"""
Text component - displays multi-line text with word wrapping.
"""

from __future__ import annotations

from typing import Callable, Optional, List

from ..tui import Component
from ..utils import apply_background_to_line, visible_width, wrap_text_with_ansi


class Text(Component):
    """Text component with word wrapping and optional background."""

    def __init__(
        self,
        text: str = "",
        padding_x: int = 1,
        padding_y: int = 1,
        padding_top: Optional[int] = None,
        padding_bottom: Optional[int] = None,
        custom_bg_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._text = text
        self._padding_x = padding_x
        # Allow separate top/bottom padding, falling back to padding_y
        self._padding_top = padding_top if padding_top is not None else padding_y
        self._padding_bottom = padding_bottom if padding_bottom is not None else padding_y
        self._custom_bg_fn = custom_bg_fn

        # Cache for rendered output
        self._cached_text: Optional[str] = None
        self._cached_width: Optional[int] = None
        self._cached_lines: Optional[List[str]] = None

    def set_text(self, text: str) -> None:
        """Update the text content."""
        self._text = text
        self.invalidate()

    def set_custom_bg_fn(self, custom_bg_fn: Optional[Callable[[str], str]]) -> None:
        """Update the background function."""
        self._custom_bg_fn = custom_bg_fn
        self.invalidate()

    def invalidate(self) -> None:
        """Clear cached rendering."""
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    def render(self, width: int) -> List[str]:
        """Render text with word wrapping and padding."""
        # Check cache
        if (
            self._cached_lines is not None
            and self._cached_text == self._text
            and self._cached_width == width
        ):
            return self._cached_lines

        # Don't render anything if there's no actual text
        if not self._text or self._text.strip() == "":
            result: List[str] = []
            self._cached_text = self._text
            self._cached_width = width
            self._cached_lines = result
            return result

        # Replace tabs with 3 spaces
        normalized_text = self._text.replace("\t", "   ")

        # Calculate content width (subtract left/right margins)
        content_width = max(1, width - self._padding_x * 2)

        # Wrap text (this preserves ANSI codes but does NOT pad)
        wrapped_lines = wrap_text_with_ansi(normalized_text, content_width)

        # Add margins and background to each line
        left_margin = " " * self._padding_x
        right_margin = " " * self._padding_x
        content_lines: List[str] = []

        for line in wrapped_lines:
            # Add margins
            line_with_margins = left_margin + line + right_margin

            # Apply background if specified (this also pads to full width)
            if self._custom_bg_fn:
                content_lines.append(
                    apply_background_to_line(line_with_margins, width, self._custom_bg_fn)
                )
            else:
                # No background - just pad to width with spaces
                visible_len = visible_width(line_with_margins)
                padding_needed = max(0, width - visible_len)
                content_lines.append(line_with_margins + " " * padding_needed)

        # Add top padding (empty lines)
        empty_line = " " * width
        top_lines: List[str] = []
        for _ in range(self._padding_top):
            if self._custom_bg_fn:
                top_lines.append(apply_background_to_line(empty_line, width, self._custom_bg_fn))
            else:
                top_lines.append(empty_line)

        # Add bottom padding (empty lines)
        bottom_lines: List[str] = []
        for _ in range(self._padding_bottom):
            if self._custom_bg_fn:
                bottom_lines.append(apply_background_to_line(empty_line, width, self._custom_bg_fn))
            else:
                bottom_lines.append(empty_line)

        result = [*top_lines, *content_lines, *bottom_lines]

        # Update cache
        self._cached_text = self._text
        self._cached_width = width
        self._cached_lines = result

        return result if result else [""]
