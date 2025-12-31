"""
Input component - multi-line text editor with cursor.
"""

from __future__ import annotations

from collections.abc import Callable

from ..theme import DARK_THEME, Theme
from ..tui import Component


class Input(Component):
    """Multi-line text input component with cursor support."""

    def __init__(
        self,
        border_color_fn: Callable[[str], str] | None = None,
        theme: Theme | None = None,
    ) -> None:
        """Initialize input component.

        Args:
            border_color_fn: Function to colorize border (text -> styled text)
            theme: Theme for styling (used if border_color_fn not provided)
        """
        self._theme = theme or DARK_THEME
        self._lines: list[str] = [""]
        self._cursor_line = 0
        self._cursor_col = 0
        self._last_width = 80
        # Use provided border_color_fn or fall back to theme
        self._border_color_fn = border_color_fn or self._theme.border_fg
        self._on_submit: Callable[[str], None] | None = None
        self._on_change: Callable[[str], None] | None = None
        self._disable_submit = False

        # Paste tracking
        self._pastes: dict[int, str] = {}
        self._paste_counter = 0
        self._paste_buffer = ""
        self._is_in_paste = False

        # Queued messages display (shown in gray above input)
        self._queued_messages: list[str] = []

        # External editor callback (Ctrl+G)
        self._on_editor: Callable[[str], None] | None = None

        # Tab completion callback (text -> completed_text or None)
        self._on_tab_complete: Callable[[str], str | None] | None = None

        # Ghost text for completion preview
        self._ghost_text: str = ""

    def set_on_submit(self, callback: Callable[[str], None] | None) -> None:
        """Set callback for when user submits (Enter)."""
        self._on_submit = callback

    def set_on_change(self, callback: Callable[[str], None] | None) -> None:
        """Set callback for when text changes."""
        self._on_change = callback

    def set_disable_submit(self, disabled: bool) -> None:
        """Set whether submit is disabled."""
        self._disable_submit = disabled

    def set_on_editor(self, callback: Callable[[str], None] | None) -> None:
        """Set callback for when user requests external editor (Ctrl+G)."""
        self._on_editor = callback

    def set_on_tab_complete(self, callback: Callable[[str], str | None] | None) -> None:
        """Set callback for tab completion.

        Args:
            callback: Function that takes current text and returns completed text,
                     or None if no completion available.
        """
        self._on_tab_complete = callback

    def set_ghost_text(self, ghost: str) -> None:
        """Set ghost text to show as completion preview."""
        self._ghost_text = ghost

    def add_queued_message(self, message: str) -> None:
        """Add a message to the queued display."""
        self._queued_messages.append(message)

    def pop_queued_message(self) -> str | None:
        """Remove and return the first queued message, or None if empty."""
        if self._queued_messages:
            return self._queued_messages.pop(0)
        return None

    def get_queue_count(self) -> int:
        """Get number of queued messages."""
        return len(self._queued_messages)

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

    def render(self, width: int) -> list[str]:
        """Render input component with cursor."""
        self._last_width = width
        horizontal = self._border_color_fn("─")
        gray_fg = "\x1b[38;5;245m"  # Gray text for queued messages
        ghost_fg = "\x1b[38;5;240m"  # Dimmer gray for ghost text
        reset = "\x1b[0m"

        result: list[str] = []

        # Render queued messages above input (gray text)
        if self._queued_messages:
            for i, msg in enumerate(self._queued_messages):
                # Truncate long messages
                display_msg = msg if len(msg) <= width - 4 else msg[: width - 7] + "..."
                prefix = f"{gray_fg}[{i + 1}] "
                line = f"{prefix}{display_msg}{reset}"
                # Pad to width
                visible_len = len(f"[{i + 1}] ") + len(display_msg)
                padding = " " * max(0, width - visible_len)
                result.append(line + padding)

        # Top border
        result.append(horizontal * width)

        # Layout text lines with gutter prefix from theme
        gutter = self._theme.input_gutter if hasattr(self._theme, "input_gutter") else "> "
        left_padding = gutter
        content_width = width - len(gutter)  # Account for gutter prefix
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
                    # Cursor at end - add ghost text (completion hint) if available
                    ghost_suffix = ""
                    if self._ghost_text:
                        # Calculate how much ghost text fits
                        available_for_ghost = content_width - len(before) - 1  # -1 for cursor
                        if available_for_ghost > 0:
                            ghost_suffix = self._ghost_text[:available_for_ghost]
                            visible_len += len(ghost_suffix)

                    # Add highlighted cursor space
                    if len(before) + 1 + len(ghost_suffix) <= content_width:
                        cursor = "\x1b[7m \x1b[0m"
                        if ghost_suffix:
                            display_text = before + cursor + f"{ghost_fg}{ghost_suffix}{reset}"
                        else:
                            display_text = before + cursor
                        visible_len = len(before) + 1 + len(ghost_suffix)
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

    def _layout_text(self, content_width: int) -> list[dict]:
        """Layout text lines with cursor position."""
        layout_lines: list[dict] = []

        if not self._lines or (len(self._lines) == 1 and self._lines[0] == ""):
            # Empty editor
            layout_lines.append({"text": "", "has_cursor": True, "cursor_pos": 0})
            return layout_lines

        for i, line in enumerate(self._lines):
            is_current_line = i == self._cursor_line

            if len(line) <= content_width:
                # Line fits
                layout_lines.append({
                    "text": line,
                    "has_cursor": is_current_line,
                    "cursor_pos": self._cursor_col if is_current_line else 0,
                })
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
                        and (
                            self._cursor_col <= chunk_end
                            if is_last_chunk
                            else self._cursor_col < chunk_end
                        )
                    )

                    layout_lines.append({
                        "text": chunk,
                        "has_cursor": has_cursor_in_chunk,
                        "cursor_pos": self._cursor_col - chunk_start if has_cursor_in_chunk else 0,
                    })

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

        # Ctrl+G - open external editor
        if len(data) > 0 and ord(data[0]) == 7:
            if self._on_editor:
                self._on_editor(self.get_text())
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

        # Ctrl+A - move to start of line
        if len(data) == 1 and ord(data[0]) == 1:
            self._cursor_col = 0
            return

        # Ctrl+E - move to end of line
        if len(data) == 1 and ord(data[0]) == 5:
            self._cursor_col = len(self._lines[self._cursor_line])
            return

        # Tab - trigger completion
        if len(data) == 1 and ord(data[0]) == 9:
            self._handle_tab_complete()
            return

        # Ctrl+K - delete to end of line
        if len(data) == 1 and ord(data[0]) == 11:
            self._delete_to_end_of_line()
            return

        # Ctrl+U - delete to start of line
        if len(data) == 1 and ord(data[0]) == 21:
            self._delete_to_start_of_line()
            return

        # Ctrl+W - delete word backwards
        if len(data) == 1 and ord(data[0]) == 23:
            self._delete_word_backwards()
            return

        # Option/Alt+Backspace - delete word backwards
        if data == "\x1b\x7f":
            self._delete_word_backwards()
            return

        # Delete key - forward delete
        if data == "\x1b[3~":
            self._handle_forward_delete()
            return

        # Home key variants - move to start of line
        if data in ("\x1b[H", "\x1b[1~", "\x1b[7~"):
            self._cursor_col = 0
            return

        # End key variants - move to end of line
        if data in ("\x1b[F", "\x1b[4~", "\x1b[8~"):
            self._cursor_col = len(self._lines[self._cursor_line])
            return

        # Word navigation: Option/Alt+Left or Ctrl+Left
        if data in ("\x1b[1;3D", "\x1bb", "\x1b[1;5D"):
            self._move_word_backwards()
            return

        # Word navigation: Option/Alt+Right or Ctrl+Right
        if data in ("\x1b[1;3C", "\x1bf", "\x1b[1;5C"):
            self._move_word_forwards()
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

    def _handle_tab_complete(self) -> None:
        """Handle Tab key for completion."""
        if not self._on_tab_complete:
            return

        text = self.get_text()
        completed = self._on_tab_complete(text)

        if completed and completed != text:
            # Replace text with completed version
            self._lines = completed.split("\n")
            if not self._lines:
                self._lines = [""]
            self._cursor_line = len(self._lines) - 1
            self._cursor_col = len(self._lines[self._cursor_line])
            self._ghost_text = ""

            if self._on_change:
                self._on_change(completed)

    def _insert_character(self, char: str) -> None:
        """Insert character at cursor position."""
        line = self._lines[self._cursor_line]
        before = line[: self._cursor_col]
        after = line[self._cursor_col :]
        self._lines[self._cursor_line] = before + char + after
        self._cursor_col += len(char)

        # Clear ghost text on any character insertion
        self._ghost_text = ""

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

    def _delete_to_end_of_line(self) -> None:
        """Delete from cursor to end of line (Ctrl+K)."""
        current_line = self._lines[self._cursor_line]

        if self._cursor_col < len(current_line):
            # Delete from cursor to end
            self._lines[self._cursor_line] = current_line[: self._cursor_col]
        elif self._cursor_line < len(self._lines) - 1:
            # At end of line - merge with next line
            next_line = self._lines[self._cursor_line + 1]
            self._lines[self._cursor_line] = current_line + next_line
            self._lines.pop(self._cursor_line + 1)

        if self._on_change:
            self._on_change(self.get_text())

    def _delete_to_start_of_line(self) -> None:
        """Delete from cursor to start of line (Ctrl+U)."""
        current_line = self._lines[self._cursor_line]

        if self._cursor_col > 0:
            # Delete from start to cursor
            self._lines[self._cursor_line] = current_line[self._cursor_col :]
            self._cursor_col = 0
        elif self._cursor_line > 0:
            # At start of line - merge with previous line
            previous_line = self._lines[self._cursor_line - 1]
            self._lines[self._cursor_line - 1] = previous_line + current_line
            self._lines.pop(self._cursor_line)
            self._cursor_line -= 1
            self._cursor_col = len(previous_line)

        if self._on_change:
            self._on_change(self.get_text())

    def _delete_word_backwards(self) -> None:
        """Delete word backwards (Ctrl+W, Option+Backspace)."""
        current_line = self._lines[self._cursor_line]

        if self._cursor_col == 0:
            # At start of line - merge with previous line (like backspace)
            if self._cursor_line > 0:
                previous_line = self._lines[self._cursor_line - 1]
                self._lines[self._cursor_line - 1] = previous_line + current_line
                self._lines.pop(self._cursor_line)
                self._cursor_line -= 1
                self._cursor_col = len(previous_line)
        else:
            text_before = current_line[: self._cursor_col]
            delete_from = self._cursor_col
            last_char = text_before[delete_from - 1] if delete_from > 0 else ""

            # If on whitespace or punctuation, delete just that
            if self._is_word_boundary(last_char):
                delete_from -= 1
            else:
                # Delete run of non-boundary characters (the "word")
                while delete_from > 0:
                    ch = text_before[delete_from - 1]
                    if self._is_word_boundary(ch):
                        break
                    delete_from -= 1

            self._lines[self._cursor_line] = (
                current_line[:delete_from] + current_line[self._cursor_col :]
            )
            self._cursor_col = delete_from

        if self._on_change:
            self._on_change(self.get_text())

    def _handle_forward_delete(self) -> None:
        """Handle forward delete (Delete key)."""
        current_line = self._lines[self._cursor_line]

        if self._cursor_col < len(current_line):
            # Delete character at cursor
            before = current_line[: self._cursor_col]
            after = current_line[self._cursor_col + 1 :]
            self._lines[self._cursor_line] = before + after
        elif self._cursor_line < len(self._lines) - 1:
            # At end of line - merge with next line
            next_line = self._lines[self._cursor_line + 1]
            self._lines[self._cursor_line] = current_line + next_line
            self._lines.pop(self._cursor_line + 1)

        if self._on_change:
            self._on_change(self.get_text())

    def _is_word_boundary(self, char: str) -> bool:
        """Check if character is a word boundary."""
        if not char:
            return True
        # Whitespace
        if char in " \t\n\r":
            return True
        # Punctuation
        if char in "(){}[]<>.,;:'\"!?+-=*/\\|&%^$#@~`":
            return True
        return False

    def _move_word_backwards(self) -> None:
        """Move cursor one word backwards."""
        current_line = self._lines[self._cursor_line]

        if self._cursor_col == 0:
            # At start of line - move to end of previous line
            if self._cursor_line > 0:
                self._cursor_line -= 1
                self._cursor_col = len(self._lines[self._cursor_line])
            return

        text_before = current_line[: self._cursor_col]
        new_col = self._cursor_col
        last_char = text_before[new_col - 1] if new_col > 0 else ""

        # If on boundary, skip it
        if self._is_word_boundary(last_char):
            new_col -= 1

        # Skip the word (non-boundary characters)
        while new_col > 0:
            ch = text_before[new_col - 1]
            if self._is_word_boundary(ch):
                break
            new_col -= 1

        self._cursor_col = new_col

    def _move_word_forwards(self) -> None:
        """Move cursor one word forwards."""
        current_line = self._lines[self._cursor_line]

        if self._cursor_col >= len(current_line):
            # At end of line - move to start of next line
            if self._cursor_line < len(self._lines) - 1:
                self._cursor_line += 1
                self._cursor_col = 0
            return

        new_col = self._cursor_col
        char_at_cursor = current_line[new_col] if new_col < len(current_line) else ""

        # If on boundary, skip it
        if self._is_word_boundary(char_at_cursor):
            new_col += 1

        # Skip the word (non-boundary characters)
        while new_col < len(current_line):
            ch = current_line[new_col]
            if self._is_word_boundary(ch):
                break
            new_col += 1

        self._cursor_col = new_col
