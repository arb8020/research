"""Text utilities for TUI rendering.

Handles ANSI-aware text width calculation, word wrapping, and truncation.

Extracted from rollouts/frontends/tui/utils.py - same code, new home.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections.abc import Callable
from typing import NamedTuple

# Pre-compiled patterns for performance (avoid re-compiling on every call)
TERMINAL_CONTROL_PATTERN = re.compile(
    r"\x1b\[\?2004[hl]"  # Bracketed paste enable/disable
    r"|\x1b\[20[01]~"  # Bracketed paste start/end markers
    r"|\x1b\[\d+~"  # Other ~ terminated sequences (function keys, etc.)
)

# ANSI escape sequences (SGR codes and others)
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z~]")


def strip_terminal_control_sequences(text: str) -> str:
    """Strip terminal control sequences that should never appear in content.

    This includes bracketed paste sequences and other control codes that
    can leak into text when returning from external editors.
    """
    return TERMINAL_CONTROL_PATTERN.sub("", text)


@functools.lru_cache(maxsize=1024)
def visible_width(text: str) -> int:
    """Calculate the visible width of a string in terminal columns.

    Handles:
    - ANSI escape sequences (zero width)
    - Terminal control sequences like bracketed paste (zero width)
    - Wide characters (CJK, emoji = 2 columns)
    - Combining characters (zero width)
    - Tabs (converted to 3 spaces)

    Results are cached (LRU, 1024 entries) since the same strings are
    often measured repeatedly during rendering.
    """
    # Normalize tabs
    text = text.replace("\t", "   ")

    # Strip terminal control sequences (bracketed paste, etc.)
    text = strip_terminal_control_sequences(text)

    # Strip ANSI escape sequences using pre-compiled pattern
    text_no_ansi = ANSI_ESCAPE_PATTERN.sub("", text)

    width = 0
    for char in text_no_ansi:
        # Get East Asian Width property
        ea_width = unicodedata.east_asian_width(char)
        cat = unicodedata.category(char)

        # Combining marks have zero width
        if cat.startswith("M"):
            continue

        # Wide characters (CJK, emoji) take 2 columns
        if ea_width in ("F", "W"):
            width += 2
        else:
            width += 1

    return width


class AnsiCode(NamedTuple):
    """Extracted ANSI escape code."""

    code: str
    length: int


def extract_ansi_code(text: str, pos: int) -> AnsiCode | None:
    """Extract ANSI escape sequence at given position.

    Handles:
    - SGR codes ending in m (colors, styles)
    - Cursor codes ending in G, K, H, J
    - Special sequences ending in ~ (function keys, bracketed paste)

    Returns None if no escape sequence at position.
    """
    if pos >= len(text) or text[pos] != "\x1b":
        return None
    if pos + 1 >= len(text) or text[pos + 1] != "[":
        return None

    j = pos + 2
    # Include ~ for bracketed paste and function key sequences
    while j < len(text) and text[j] not in "mGKHJ~":
        j += 1

    if j < len(text):
        return AnsiCode(code=text[pos : j + 1], length=j + 1 - pos)

    return None


class AnsiCodeTracker:
    """Track active ANSI SGR codes to preserve styling across line breaks."""

    def __init__(self) -> None:
        self._active_codes: list[str] = []

    def process(self, ansi_code: str) -> None:
        """Process an ANSI code, updating active state."""
        if not ansi_code.endswith("m"):
            return

        # Full reset clears everything
        if ansi_code in ("\x1b[0m", "\x1b[m"):
            self._active_codes.clear()
        else:
            self._active_codes.append(ansi_code)

    def get_active_codes(self) -> str:
        """Get string of all active codes to reapply styling."""
        return "".join(self._active_codes)

    def has_active_codes(self) -> bool:
        """Check if there are any active codes."""
        return len(self._active_codes) > 0


def _update_tracker_from_text(text: str, tracker: AnsiCodeTracker) -> None:
    """Scan text for ANSI codes and update tracker."""
    i = 0
    while i < len(text):
        result = extract_ansi_code(text, i)
        if result:
            tracker.process(result.code)
            i += result.length
        else:
            i += 1


def _split_into_tokens_with_ansi(text: str) -> list[str]:
    """Split text into tokens (words/whitespace) while keeping ANSI codes attached."""
    tokens: list[str] = []
    current = ""
    in_whitespace = False
    i = 0

    while i < len(text):
        result = extract_ansi_code(text, i)
        if result:
            current += result.code
            i += result.length
            continue

        char = text[i]
        char_is_space = char == " "

        if char_is_space != in_whitespace and current:
            tokens.append(current)
            current = ""

        in_whitespace = char_is_space
        current += char
        i += 1

    if current:
        tokens.append(current)

    return tokens


def wrap_text_with_ansi(text: str, width: int) -> list[str]:
    """Wrap text with ANSI codes preserved.

    Does word wrapping only - NO padding, NO background colors.
    Returns lines where each line is <= width visible chars.
    Active ANSI codes are preserved across line breaks.

    Args:
        text: Text to wrap (may contain ANSI codes and newlines)
        width: Maximum visible width per line

    Returns:
        Array of wrapped lines (NOT padded to width)
    """
    if not text:
        return [""]

    # Handle newlines by processing each line separately
    input_lines = text.split("\n")
    result: list[str] = []

    for input_line in input_lines:
        result.extend(_wrap_single_line(input_line, width))

    return result if result else [""]


def _wrap_single_line(line: str, width: int) -> list[str]:
    """Wrap a single line (no embedded newlines)."""
    if not line:
        return [""]

    visible_len = visible_width(line)
    if visible_len <= width:
        return [line]

    wrapped: list[str] = []
    tracker = AnsiCodeTracker()
    tokens = _split_into_tokens_with_ansi(line)

    current_line = ""
    current_visible_length = 0

    for token in tokens:
        token_visible_length = visible_width(token)
        is_whitespace = token.strip() == ""

        # Token itself is too long - break it character by character
        if token_visible_length > width and not is_whitespace:
            if current_line:
                wrapped.append(current_line)
                current_line = ""
                current_visible_length = 0

            # Break long token
            broken = _break_long_word(token, width, tracker)
            wrapped.extend(broken[:-1])
            current_line = broken[-1] if broken else ""
            current_visible_length = visible_width(current_line)
            continue

        # Check if adding this token would exceed width
        total_needed = current_visible_length + token_visible_length

        if total_needed > width and current_visible_length > 0:
            # Wrap to next line - don't carry trailing whitespace
            wrapped.append(current_line.rstrip())
            if is_whitespace:
                # Don't start new line with whitespace
                current_line = tracker.get_active_codes()
                current_visible_length = 0
            else:
                current_line = tracker.get_active_codes() + token
                current_visible_length = token_visible_length
        else:
            # Add to current line
            current_line += token
            current_visible_length += token_visible_length

        _update_tracker_from_text(token, tracker)

    if current_line:
        wrapped.append(current_line)

    return wrapped if wrapped else [""]


def _break_long_word(word: str, width: int, tracker: AnsiCodeTracker) -> list[str]:
    """Break a word that's too long to fit on one line."""
    lines: list[str] = []
    current_line = tracker.get_active_codes()
    current_width = 0

    # Separate ANSI codes from visible content
    segments: list[tuple[str, str]] = []  # (type, value)
    i = 0

    while i < len(word):
        result = extract_ansi_code(word, i)
        if result:
            segments.append(("ansi", result.code))
            i += result.length
        else:
            # Add single character as grapheme
            segments.append(("grapheme", word[i]))
            i += 1

    # Process segments
    for seg_type, seg_value in segments:
        if seg_type == "ansi":
            current_line += seg_value
            tracker.process(seg_value)
            continue

        grapheme = seg_value
        grapheme_width = visible_width(grapheme)

        if current_width + grapheme_width > width:
            lines.append(current_line)
            current_line = tracker.get_active_codes()
            current_width = 0

        current_line += grapheme
        current_width += grapheme_width

    if current_line:
        lines.append(current_line)

    return lines if lines else [""]


def truncate_to_width(text: str, max_width: int, ellipsis: str = "...") -> str:
    """Truncate text to fit within a maximum visible width, adding ellipsis if needed.

    Properly handles ANSI escape codes (they don't count toward width).

    Args:
        text: Text to truncate (may contain ANSI codes)
        max_width: Maximum visible width
        ellipsis: Ellipsis string to append when truncating

    Returns:
        Truncated text with ellipsis if it exceeded max_width
    """
    text_visible_width = visible_width(text)

    if text_visible_width <= max_width:
        return text

    ellipsis_width = visible_width(ellipsis)
    target_width = max_width - ellipsis_width

    if target_width <= 0:
        return ellipsis[:max_width]

    current_width = 0
    truncate_at = 0
    i = 0

    while i < len(text) and current_width < target_width:
        # Skip ANSI escape sequences
        if text[i] == "\x1b" and i + 1 < len(text) and text[i + 1] == "[":
            j = i + 2
            while j < len(text) and not text[j].isalpha():
                j += 1
            j += 1  # Include the final letter
            truncate_at = j
            i = j
            continue

        char = text[i]
        char_width = visible_width(char)

        if current_width + char_width > target_width:
            break

        current_width += char_width
        truncate_at = i + 1
        i += 1

    # Add reset code before ellipsis to prevent styling leaking into it
    return text[:truncate_at] + "\x1b[0m" + ellipsis


def slice_ansi(text: str, start: int, end: int) -> str:
    """Extract a horizontal slice of text by visible column positions.

    Like text[start:end] but works with ANSI codes - the slice is based on
    visible character positions, not byte positions. ANSI codes that were
    active at the start position are prepended to maintain styling.

    This is equivalent to bubble tea's ansi.Cut(text, start, end).

    Args:
        text: Text to slice (may contain ANSI codes)
        start: Starting visible column (0-indexed, inclusive)
        end: Ending visible column (exclusive)

    Returns:
        Sliced text with ANSI codes preserved

    Invariants:
        - visible_width(result) <= end - start
        - Result preserves ANSI styling from the sliced region

    Example:
        slice_ansi("\\x1b[31mhello world\\x1b[0m", 2, 7) -> "\\x1b[31mllo w\\x1b[0m"
    """
    assert start >= 0, f"start must be >= 0, got {start}"
    assert end >= 0, f"end must be >= 0, got {end}"

    if start >= end:
        return ""

    tracker = AnsiCodeTracker()
    result_chars: list[str] = []
    current_col = 0
    i = 0

    # First pass: skip to start position, tracking ANSI codes
    while i < len(text) and current_col < start:
        ansi = extract_ansi_code(text, i)
        if ansi:
            tracker.process(ansi.code)
            i += ansi.length
            continue

        char = text[i]
        char_width = visible_width(char)

        # Check if this character spans the start position
        if current_col + char_width > start:
            # Character starts before start but extends into our slice
            # Include it if any part is visible
            break

        current_col += char_width
        i += 1

    # Prepend active ANSI codes to maintain styling
    prefix = tracker.get_active_codes()

    # Second pass: collect characters from start to end
    while i < len(text) and current_col < end:
        ansi = extract_ansi_code(text, i)
        if ansi:
            result_chars.append(ansi.code)
            tracker.process(ansi.code)
            i += ansi.length
            continue

        char = text[i]
        char_width = visible_width(char)

        # Include character if it starts before end
        result_chars.append(char)
        current_col += char_width
        i += 1

    # Add reset at end to prevent style leaking
    suffix = "\x1b[0m" if tracker.has_active_codes() or prefix else ""

    result = prefix + "".join(result_chars) + suffix

    # Assert invariant: result width should not exceed requested slice width
    result_width = visible_width(result)
    max_expected = end - start
    assert result_width <= max_expected, (
        f"slice_ansi result width {result_width} > expected max {max_expected}. "
        f"start={start}, end={end}, text={repr(text[:80])}"
    )

    return result


def apply_background_to_line(line: str, width: int, bg_fn: Callable[[str], str]) -> str:
    """Apply background color to a line, padding to full width.

    Args:
        line: Line of text (may contain ANSI codes)
        width: Total width to pad to
        bg_fn: Background color function (text -> styled text)

    Returns:
        Line with background applied and padded to width
    """
    visible_len = visible_width(line)
    padding_needed = max(0, width - visible_len)
    padding = " " * padding_needed
    with_padding = line + padding
    return bg_fn(with_padding)
