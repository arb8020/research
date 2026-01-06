"""Shared formatting utilities for environment tool output.

Provides consistent formatting across all environments for:
- Extracting text from tool results
- Formatting tool output with truncation and theming
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..frontends.tui.theme import Theme


def get_text_output(result: dict[str, Any] | None) -> str:
    """Extract text output from tool result.

    Handles multiple result formats:
    - {"content": "string"} - direct string content
    - {"content": [{"type": "text", "text": "..."}]} - content blocks
    - {"content": {"content": [...]}} - legacy nested structure

    Returns:
        Extracted text with ANSI codes and carriage returns stripped.
    """
    if not result:
        return ""

    content = result.get("content", {})

    # Direct string content
    if isinstance(content, str):
        return content

    # Content block list
    if isinstance(content, list):
        text_blocks = [c for c in content if isinstance(c, dict) and c.get("type") == "text"]
        text_output = "\n".join(c.get("text", "") for c in text_blocks if c.get("text"))
        return _strip_ansi(text_output)

    # Legacy nested structure
    if isinstance(content, dict):
        content_list = content.get("content", [])
        if isinstance(content_list, list):
            text_blocks = [
                c for c in content_list if isinstance(c, dict) and c.get("type") == "text"
            ]
            text_output = "\n".join(c.get("text", "") for c in text_blocks if c.get("text"))
            return _strip_ansi(text_output)

    return ""


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape codes and carriage returns."""
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    return text.replace("\r", "")


def format_tool_output(
    header: str,
    result: dict[str, Any] | None,
    expanded: bool,
    theme: Theme | None = None,
    max_lines: int = 10,
    success_summary: str | None = None,
    error_summary: str | None = None,
    style_fn: str | None = None,
) -> str:
    """Format tool output with consistent styling.

    Args:
        header: Tool call header line (e.g., "bash(command='ls')")
        result: Tool result dict with 'content' and optional 'isError'
        expanded: Whether to show full output or truncate
        theme: Optional theme for styling
        max_lines: Max lines to show when not expanded
        success_summary: Summary text for successful execution
        error_summary: Summary text for failed execution
        style_fn: Theme method name to style output lines (e.g., 'diff_context_fg')

    Returns:
        Formatted string for TUI display.
    """
    text = header

    if not result:
        return text

    output = get_text_output(result).strip()
    if not output:
        return text

    is_error = result.get("isError", False)
    lines = output.split("\n")
    display_count = len(lines) if expanded else max_lines
    display_lines = lines[:display_count]
    remaining = len(lines) - display_count

    # Add summary line if provided
    summary = error_summary if is_error else success_summary
    if summary:
        text += f"\n⎿ {summary}"

    # Style and append lines
    for line in display_lines:
        styled_line = line
        if theme and style_fn:
            style_method = getattr(theme, style_fn, None)
            if style_method:
                styled_line = style_method(line)
        text += f"\n  {styled_line}"

    if remaining > 0:
        text += f"\n  ... ({remaining} more lines)"

    return text


def shorten_path(path: str) -> str:
    """Convert absolute path to tilde notation if in home directory."""
    import os

    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def replace_tabs(text: str) -> str:
    """Replace tabs with spaces for consistent rendering."""
    return text.replace("\t", "   ")
