"""Text utilities for TUI rendering.

Re-exports from pytui.text - all code now lives there.
"""

from pytui.text import (
    TERMINAL_CONTROL_PATTERN,
    AnsiCode,
    AnsiCodeTracker,
    apply_background_to_line,
    extract_ansi_code,
    strip_terminal_control_sequences,
    truncate_to_width,
    visible_width,
    wrap_text_with_ansi,
)

__all__ = [
    "TERMINAL_CONTROL_PATTERN",
    "AnsiCode",
    "AnsiCodeTracker",
    "apply_background_to_line",
    "extract_ansi_code",
    "strip_terminal_control_sequences",
    "truncate_to_width",
    "visible_width",
    "wrap_text_with_ansi",
]
