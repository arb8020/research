"""
ANSI-aware text utilities for terminal rendering.
"""

from __future__ import annotations

import re
import unicodedata

# Pattern to match ANSI escape sequences
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def visible_width(text: str) -> int:
    """Calculate visible width of text, ignoring ANSI codes.

    Handles:
    - ANSI escape sequences (zero width)
    - Wide characters (CJK, emoji = 2 columns)
    - Combining characters (zero width)
    - Tabs (converted to 4 spaces)
    """
    text = text.replace("\t", "    ")
    text_no_ansi = ANSI_PATTERN.sub("", text)

    width = 0
    for char in text_no_ansi:
        cat = unicodedata.category(char)
        if cat.startswith("M"):  # Combining marks
            continue
        ea_width = unicodedata.east_asian_width(char)
        if ea_width in ("F", "W"):  # Full-width
            width += 2
        else:
            width += 1

    return width


def pad_line(text: str, width: int, bg: str = "", reset: str = "\x1b[0m") -> str:
    """Pad a line to exact width, accounting for ANSI codes.

    Args:
        text: Text with possible ANSI codes
        width: Target visible width
        bg: Background color code to apply to padding
        reset: Reset code at end

    Returns:
        Line padded to exactly `width` visible characters
    """
    current_width = visible_width(text)
    padding_needed = max(0, width - current_width)

    if bg:
        return f"{text}{bg}{' ' * padding_needed}{reset}"
    else:
        return f"{text}{' ' * padding_needed}"


def truncate(text: str, max_width: int, ellipsis: str = "~") -> str:
    """Truncate text to max visible width, preserving ANSI codes."""
    if visible_width(text) <= max_width:
        return text

    result = []
    current_width = 0
    target = max_width - len(ellipsis)
    i = 0

    while i < len(text) and current_width < target:
        # Check for ANSI sequence
        if text[i] == "\x1b" and i + 1 < len(text) and text[i + 1] == "[":
            j = i + 2
            while j < len(text) and not text[j].isalpha():
                j += 1
            j += 1  # Include final letter
            result.append(text[i:j])
            i = j
            continue

        char = text[i]
        char_width = visible_width(char)

        if current_width + char_width > target:
            break

        result.append(char)
        current_width += char_width
        i += 1

    return "".join(result) + "\x1b[0m" + ellipsis
