"""
Text component - displays multi-line text with word wrapping.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..tui import Component
from ..utils import apply_background_to_line, visible_width, wrap_text_with_ansi

if TYPE_CHECKING:
    from ..theme import Theme


def _render_line_with_margins(
    line: str, width: int, padding_x: int, bg_fn: Callable[[str], str] | None
) -> str:
    """Render a single line with left/right margins and optional background.

    Background is applied only to content, not margins.
    """
    left_margin = " " * padding_x
    right_margin = " " * padding_x

    if bg_fn:
        content_width = width - padding_x * 2
        colored_content = apply_background_to_line(line, content_width, bg_fn)
        return left_margin + colored_content + right_margin

    # No background - add margins and pad to width
    line_with_margins = left_margin + line + right_margin
    visible_len = visible_width(line_with_margins)
    padding_needed = max(0, width - visible_len)
    return line_with_margins + " " * padding_needed


def _render_empty_lines(
    count: int,
    width: int,
    padding_x: int,
    bg_fn: Callable[[str], str] | None,
    use_compact: bool = False,
    is_top: bool = True,
) -> list[str]:
    """Render N empty lines with optional background, respecting left/right margins.

    Args:
        count: Number of padding lines
        width: Total width
        padding_x: Horizontal padding
        bg_fn: Background color function (when use_compact=True, this colors the block characters as foreground)
        use_compact: If True, use half-block unicode characters colored with foreground (no background fill)
        is_top: If True, use ▄ (lower half block) for top padding; if False, use ▀ (upper half block) for bottom padding
    """
    if count == 0:
        return []

    if bg_fn:
        left_margin = " " * padding_x
        right_margin = " " * padding_x
        content_width = width - padding_x * 2

        if use_compact:
            # Use colored half-block characters with NO background
            # Top padding: ▄ (lower half) - sits at bottom of line, separates from content above
            # Bottom padding: ▀ (upper half) - sits at top of line, separates from content below
            import re

            from ..theme import RESET

            test_output = bg_fn(" ")
            # Extract the background color code from ANSI (format: \x1b[48;2;R;G;Bm)
            bg_match = re.search(r"\x1b\[48;2;(\d+);(\d+);(\d+)m", test_output)
            if bg_match:
                r, g, b = bg_match.groups()
                # Create foreground color from the same RGB
                fg_color = f"\x1b[38;2;{r};{g};{b}m"
                # Choose block character based on position
                block_char = "▄" if is_top else "▀"
                empty_content = block_char * content_width
                colored_line = left_margin + fg_color + empty_content + RESET + right_margin
            else:
                # Fallback: just use spaces if we can't extract color
                empty_content = " " * content_width
                colored_line = left_margin + empty_content + right_margin
        else:
            # Full background color on empty space
            empty_content = " " * content_width
            colored_line = (
                left_margin
                + apply_background_to_line(empty_content, content_width, bg_fn)
                + right_margin
            )

        return [colored_line] * count

    # No background - just return empty lines
    empty_line = " " * width
    return [empty_line] * count


class Text(Component):
    """Text component with word wrapping and optional background."""

    def __init__(
        self,
        text: str = "",
        padding_x: int = 1,
        padding_y: int = 1,
        padding_top: int | None = None,
        padding_bottom: int | None = None,
        custom_bg_fn: Callable[[str], str] | None = None,
        theme: Theme | None = None,
        gutter_prefix: str | None = None,
    ) -> None:
        self._text = text
        self._padding_x = padding_x
        self._padding_top = padding_top if padding_top is not None else padding_y
        self._padding_bottom = padding_bottom if padding_bottom is not None else padding_y
        self._custom_bg_fn = custom_bg_fn
        self._theme = theme
        self._gutter_prefix = gutter_prefix

        # Cache for rendered output
        self._cached_text: str | None = None
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None
        self._cached_gutter_prefix: str | None = None

    def set_text(self, text: str) -> None:
        """Update the text content."""
        self._text = text
        self.invalidate()

    def set_custom_bg_fn(self, custom_bg_fn: Callable[[str], str] | None) -> None:
        """Update the background function."""
        self._custom_bg_fn = custom_bg_fn
        self.invalidate()

    def invalidate(self) -> None:
        """Clear cached rendering."""
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None
        self._cached_gutter_prefix = None

    def render(self, width: int) -> list[str]:
        """Render text with word wrapping and padding."""
        # Check cache
        if (
            self._cached_lines is not None
            and self._cached_text == self._text
            and self._cached_width == width
            and self._cached_gutter_prefix == self._gutter_prefix
        ):
            return self._cached_lines

        # Empty text - return empty list
        if not self._text or self._text.strip() == "":
            self._cached_text = self._text
            self._cached_width = width
            self._cached_lines = []
            return []

        # Normalize and wrap text
        normalized_text = self._text.replace("\t", "   ")
        # Account for gutter prefix width if present
        gutter_width = 0
        if self._gutter_prefix:
            gutter_width = visible_width(self._gutter_prefix) + 1  # +1 for space after prefix
        content_width = max(1, width - self._padding_x * 2 - gutter_width)
        wrapped_lines = wrap_text_with_ansi(normalized_text, content_width)

        # Check if we should use rounded corners
        use_rounded = self._theme and getattr(self._theme, "use_rounded_corners", False)

        # Render content lines with margins and background
        # If we have a gutter, render to reduced width so final line fits after adding gutter
        render_width = width - gutter_width if gutter_width > 0 else width
        content_lines = [
            _render_line_with_margins(line, render_width, self._padding_x, self._custom_bg_fn)
            for line in wrapped_lines
        ]

        # Add rounded corners if enabled
        if use_rounded and self._custom_bg_fn and content_lines:
            from ..theme import RESET, hex_to_fg

            corner_color = hex_to_fg(self._theme.border)

            # Add corners to first line
            first_line = content_lines[0]
            left_margin = " " * (self._padding_x - 1)
            content_lines[0] = (
                left_margin
                + corner_color
                + self._theme.corner_tl
                + RESET
                + first_line[self._padding_x :]
            )

            # Add corner to last line
            last_line = content_lines[-1]
            # Find where the colored content ends (before right margin)
            content_end = width - self._padding_x
            content_lines[-1] = (
                last_line[:content_end]
                + corner_color
                + self._theme.corner_bl
                + RESET
                + " " * (self._padding_x - 1)
            )

        # Add vertical padding (use render_width if we have a gutter)
        use_compact = self._theme and getattr(self._theme, "use_compact_padding", False)
        top_lines = _render_empty_lines(
            self._padding_top,
            render_width,
            self._padding_x,
            self._custom_bg_fn,
            use_compact,
            is_top=True,
        )
        bottom_lines = _render_empty_lines(
            self._padding_bottom,
            render_width,
            self._padding_x,
            self._custom_bg_fn,
            use_compact,
            is_top=False,
        )

        result = [*top_lines, *content_lines, *bottom_lines]

        # Add gutter prefix if specified
        if self._gutter_prefix:
            gutter_len = visible_width(self._gutter_prefix)

            # Find the position where visible characters start (skip ANSI codes)

            def skip_ansi_and_get_pos(line, visible_chars_to_skip):
                """Find byte position after skipping N visible characters, preserving ANSI."""
                pos = 0
                visible_count = 0
                while pos < len(line) and visible_count < visible_chars_to_skip:
                    if line[pos : pos + 2] == "\x1b[":
                        # Skip ANSI escape sequence
                        end = line.find("m", pos)
                        if end != -1:
                            pos = end + 1
                        else:
                            pos += 1
                    else:
                        # Regular visible character
                        visible_count += 1
                        pos += 1
                return pos

            # Calculate which lines are padding vs content
            num_top_padding = len(top_lines)
            num_bottom_padding = len(bottom_lines)
            num_content = len(content_lines)

            new_result = []
            for i, line in enumerate(result):
                # Padding lines get spaces instead of gutter prefix
                if i < num_top_padding or i >= (num_top_padding + num_content):
                    # This is a padding line - add spaces instead of gutter
                    pos = skip_ansi_and_get_pos(line, self._padding_x)
                    new_result.append(" " * (gutter_len + 1) + line[pos:])
                elif i == num_top_padding:
                    # First content line: add gutter prefix
                    pos = skip_ansi_and_get_pos(line, self._padding_x)
                    new_result.append(self._gutter_prefix + " " + line[pos:])
                else:
                    # Other content lines: add spacing
                    pos = skip_ansi_and_get_pos(line, self._padding_x)
                    new_result.append(" " * (gutter_len + 1) + line[pos:])
            result = new_result

        # Update cache
        self._cached_text = self._text
        self._cached_width = width
        self._cached_gutter_prefix = self._gutter_prefix
        self._cached_lines = result

        return result
