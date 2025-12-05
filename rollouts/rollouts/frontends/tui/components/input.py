"""
Input component - multi-line text editor with cursor.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from ..tui import Component
from ..theme import Theme, DARK_THEME
from ..utils import visible_width


class Input(Component):
    """Multi-line text input component with cursor support."""

    def __init__(
        self,
        border_color_fn: Optional[Callable[[str], str]] = None,
        theme: Optional[Theme] = None,
    ) -> None:
        """Initialize input component.

        Args:
            border_color_fn: Function to colorize border (text -> styled text)
            theme: Theme for styling (used if border_color_fn not provided)
        """
        self._theme = theme or DARK_THEME
        self._lines: List[str] = [""]
        self._cursor_line = 0
        self._cursor_col = 0
        self._last_width = 80
        # Use provided border_color_fn or fall back to theme
        self._border_color_fn = border_color_fn or self._theme.border_fg
        self._on_submit: Optional[Callable[[str], None]] = None
        self._on_change: Optional[Callable[[str], None]] = None
        self._disable_submit = False

        # Paste tracking
        self._pastes: dict[int, str] = {}
        self._paste_counter = 0
        self._paste_buffer = ""
        self._is_in_paste = False

    def set_on_submit(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set callback for when user submits (Enter)."""
        self._on_submit = callback

    def set_on_change(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set callback for when text changes."""
        self._on_change = callback

    def set_disable_submit(self, disabled: bool) -> None:
        """Set whether submit is disabled."""
        self._disable_submit = disabled

    def get_text(self) -> str:
        """Get current text content."""
        return "\n".join(self._lines)

    def set_text(self, text: str) -> None:
        """Set text content."""
        # Normalize line endings
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        self._lines = normalized.split("\n") if normalized else [""]
        if not self._lines:
            self._lines = [""]

        # Reset cursor to end
        self._cursor_line = len(self._lines) - 1
        self._cursor_col = len(self._lines[self._cursor_line])

        if self._on_change:
            self._on_change(self.get_text())

    def invalidate(self) -> None:
        """No cached state to invalidate."""
        pass

    def render(self, width: int) -> List[str]:
        """Render input component with cursor."""
        self._last_width = width
        horizontal = self._border_color_fn("─")

        result: List[str] = []

        # Top border
        result.append(horizontal * width)

        # Layout text lines with "> " gutter prefix
        left_padding = "> "
        content_width = width - 2  # Account for gutter prefix
        layout_lines = self._layout_text(content_width)

        for layout_line in layout_lines:
            display_text = layout_line["text"]
            visible_len = len(display_text)

            # Add cursor if this line has it
            if layout_line.get("has_cursor") and "cursor_pos" in layout_line:
                cursor_pos = layout_line["cursor_pos"]
                before = display_text[:cursor_pos]
                after = display_text[cursor_pos:]

                if after:
                    # Cursor on character - highlight it
                    cursor = f"\x1b[7m{after[0]}\x1b[0m"
                    rest_after = after[1:]
                    display_text = before + cursor + rest_after
                else:
                    # Cursor at end - add highlighted space if room
                    if len(display_text) < content_width:
                        cursor = "\x1b[7m \x1b[0m"
                        display_text = before + cursor
                        visible_len = len(display_text)
                    elif before:
                        # Line full - highlight last char
                        last_char = before[-1]
                        cursor = f"\x1b[7m{last_char}\x1b[0m"
                        display_text = before[:-1] + cursor

            # Pad to width (accounting for left padding)
            padding = " " * max(0, content_width - visible_len)
            result.append(left_padding + display_text + padding)

        # Bottom border
        result.append(horizontal * width)

        return result

    def _layout_text(self, content_width: int) -> List[dict]:
        """Layout text lines with cursor position."""
        layout_lines: List[dict] = []

        if not self._lines or (len(self._lines) == 1 and self._lines[0] == ""):
            # Empty editor
            layout_lines.append({"text": "", "has_cursor": True, "cursor_pos": 0})
            return layout_lines

        for i, line in enumerate(self._lines):
            is_current_line = i == self._cursor_line

            if len(line) <= content_width:
                # Line fits
                layout_lines.append(
                    {
                        "text": line,
                        "has_cursor": is_current_line,
                        "cursor_pos": self._cursor_col if is_current_line else 0,
                    }
                )
            else:
                # Line needs wrapping
                for pos in range(0, len(line), content_width):
                    chunk = line[pos : pos + content_width]
                    chunk_start = pos
                    chunk_end = pos + len(chunk)
                    is_last_chunk = pos + content_width >= len(line)

                    has_cursor_in_chunk = (
                        is_current_line
                        and self._cursor_col >= chunk_start
                        and (self._cursor_col <= chunk_end if is_last_chunk else self._cursor_col < chunk_end)
                    )

                    layout_lines.append(
                        {
                            "text": chunk,
                            "has_cursor": has_cursor_in_chunk,
                            "cursor_pos": self._cursor_col - chunk_start if has_cursor_in_chunk else 0,
                        }
                    )

        return layout_lines

    def handle_input(self, data: str) -> None:
        """Handle keyboard input."""
        # Handle bracketed paste mode
        if "\x1b[200~" in data:
            self._is_in_paste = True
            self._paste_buffer = ""
            data = data.replace("\x1b[200~", "")

        if self._is_in_paste:
            self._paste_buffer += data
            end_index = self._paste_buffer.find("\x1b[201~")
            if end_index != -1:
                paste_content = self._paste_buffer[:end_index]
                self._handle_paste(paste_content)
                self._is_in_paste = False
                remaining = self._paste_buffer[end_index + 6 :]
                self._paste_buffer = ""
                if remaining:
                    self.handle_input(remaining)
                return

        # Ctrl+C - let parent handle
        if len(data) > 0 and ord(data[0]) == 3:
            return

        # Enter - submit
        if len(data) == 1 and ord(data[0]) == 13:  # CR
            if self._disable_submit:
                return

            result = self.get_text().strip()
            # Replace paste markers
            for paste_id, paste_content in self._pastes.items():
                import re

                marker_pattern = rf"\[paste #{paste_id}( (\+\d+ lines|\d+ chars))?\]"
                result = re.sub(marker_pattern, paste_content, result)

            # Reset editor
            self._lines = [""]
            self._cursor_line = 0
            self._cursor_col = 0
            self._pastes.clear()
            self._paste_counter = 0

            if self._on_change:
                self._on_change("")

            if self._on_submit:
                self._on_submit(result)
            return

        # Backspace
        if len(data) > 0 and ord(data[0]) in (127, 8):
            self._handle_backspace()
            return

        # Arrow keys
        if data == "\x1b[A":  # Up
            self._move_cursor(-1, 0)
            return
        if data == "\x1b[B":  # Down
            self._move_cursor(1, 0)
            return
        if data == "\x1b[C":  # Right
            self._move_cursor(0, 1)
            return
        if data == "\x1b[D":  # Left
            self._move_cursor(0, -1)
            return

        # Regular characters
        if len(data) > 0 and ord(data[0]) >= 32:
            self._insert_character(data)

    def _insert_character(self, char: str) -> None:
        """Insert character at cursor position."""
        line = self._lines[self._cursor_line]
        before = line[: self._cursor_col]
        after = line[self._cursor_col :]
        self._lines[self._cursor_line] = before + char + after
        self._cursor_col += len(char)

        if self._on_change:
            self._on_change(self.get_text())

    def _handle_backspace(self) -> None:
        """Handle backspace key."""
        if self._cursor_col > 0:
            line = self._lines[self._cursor_line]
            before = line[: self._cursor_col - 1]
            after = line[self._cursor_col :]
            self._lines[self._cursor_line] = before + after
            self._cursor_col -= 1
        elif self._cursor_line > 0:
            # Merge with previous line
            current_line = self._lines[self._cursor_line]
            previous_line = self._lines[self._cursor_line - 1]
            self._lines[self._cursor_line - 1] = previous_line + current_line
            self._lines.pop(self._cursor_line)
            self._cursor_line -= 1
            self._cursor_col = len(previous_line)

        if self._on_change:
            self._on_change(self.get_text())

    def _move_cursor(self, delta_line: int, delta_col: int) -> None:
        """Move cursor position."""
        if delta_line != 0:
            new_line = self._cursor_line + delta_line
            if 0 <= new_line < len(self._lines):
                self._cursor_line = new_line
                # Clamp column to line length
                line_len = len(self._lines[self._cursor_line])
                self._cursor_col = min(self._cursor_col, line_len)

        if delta_col != 0:
            line = self._lines[self._cursor_line]
            if delta_col > 0:
                if self._cursor_col < len(line):
                    self._cursor_col += 1
                elif self._cursor_line < len(self._lines) - 1:
                    self._cursor_line += 1
                    self._cursor_col = 0
            else:
                if self._cursor_col > 0:
                    self._cursor_col -= 1
                elif self._cursor_line > 0:
                    self._cursor_line -= 1
                    self._cursor_col = len(self._lines[self._cursor_line])

    def _handle_paste(self, pasted_text: str) -> None:
        """Handle pasted text."""
        # Clean text
        clean_text = pasted_text.replace("\r\n", "\n").replace("\r", "\n")
        # Filter non-printable except newlines
        filtered = "".join(c for c in clean_text if c == "\n" or ord(c) >= 32)
        pasted_lines = filtered.split("\n")

        # Large paste - store and insert marker
        if len(pasted_lines) > 10 or len(filtered) > 1000:
            self._paste_counter += 1
            paste_id = self._paste_counter
            self._pastes[paste_id] = filtered

            marker = (
                f"[paste #{paste_id} +{len(pasted_lines)} lines]"
                if len(pasted_lines) > 10
                else f"[paste #{paste_id} {len(filtered)} chars]"
            )
            for char in marker:
                self._insert_character(char)
            return

        # Small paste - insert directly
        if len(pasted_lines) == 1:
            for char in pasted_lines[0]:
                self._insert_character(char)
            return

        # Multi-line paste
        current_line = self._lines[self._cursor_line]
        before_cursor = current_line[: self._cursor_col]
        after_cursor = current_line[self._cursor_col :]

        new_lines = self._lines[: self._cursor_line]
        new_lines.append(before_cursor + pasted_lines[0])
        new_lines.extend(pasted_lines[1:-1])
        new_lines.append(pasted_lines[-1] + after_cursor)
        new_lines.extend(self._lines[self._cursor_line + 1 :])

        self._lines = new_lines
        self._cursor_line += len(pasted_lines) - 1
        self._cursor_col = len(pasted_lines[-1])

        if self._on_change:
            self._on_change(self.get_text())

