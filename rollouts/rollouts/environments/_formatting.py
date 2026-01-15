"""Shared formatting utilities for environment tool output.

Provides consistent formatting across all environments for:
- Extracting text from tool results
- Formatting tool output with truncation and theming
- Default tool rendering (works for any tool with zero config)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..dtypes import DetailLevel, ToolRenderConfig
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


def _default_header(tool_name: str, args: dict[str, Any]) -> str:
    """Generate default header for a tool call.

    Format: tool_name(arg1=..., arg2=...)
    Truncates long values to keep header readable.
    """
    if not args:
        return f"{tool_name}()"

    parts = []
    for key, value in list(args.items())[:3]:  # Max 3 args in header
        if isinstance(value, str):
            display = repr(value[:30] + "..." if len(value) > 30 else value)
        else:
            display = repr(value)
            if len(display) > 30:
                display = display[:30] + "..."
        parts.append(f"{key}={display}")

    if len(args) > 3:
        parts.append("...")

    return f"{tool_name}({', '.join(parts)})"


def format_tool_output(
    header: str,
    result: dict[str, Any] | None,
    detail_level: DetailLevel | bool,
    theme: Theme | None = None,
    config: ToolRenderConfig | None = None,
) -> str:
    """Format tool output with consistent styling.

    Args:
        header: Tool call header line (e.g., "bash(command='ls')")
        result: Tool result dict with 'content' and optional 'isError'
        detail_level: DetailLevel enum or bool for backward compat (True=EXPANDED, False=STANDARD)
        theme: Optional theme for styling
        config: Optional render config (uses defaults if None)

    Returns:
        Formatted string for TUI display.
    """
    # Import here to avoid circular imports
    from ..dtypes import DetailLevel, ToolRenderConfig

    if config is None:
        config = ToolRenderConfig()

    # Handle backward compatibility: bool -> DetailLevel
    if isinstance(detail_level, bool):
        detail_level = DetailLevel.EXPANDED if detail_level else DetailLevel.STANDARD

    text = header

    if not result:
        return text

    output = get_text_output(result).strip()
    if not output:
        return text

    is_error = result.get("isError", False)
    lines = output.split("\n")

    # Get max lines for this detail level
    max_lines = config.get_max_lines(detail_level)
    if max_lines < 0:  # -1 = unlimited
        display_count = len(lines)
    else:
        display_count = min(len(lines), max_lines)

    display_lines = lines[:display_count]
    remaining = len(lines) - display_count

    # Add summary line if provided
    summary = config.error_summary if is_error else config.success_summary
    if summary:
        text += f"\n⎿ {summary}"

    # Style and append lines
    for line in display_lines:
        styled_line = line
        if theme and config.style_fn:
            style_method = getattr(theme, config.style_fn, None)
            if style_method:
                styled_line = style_method(line)
        text += f"\n  {styled_line}"

    if remaining > 0:
        text += f"\n  ... ({remaining} more lines)"

    return text


def format_tool(
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any] | None,
    detail_level: DetailLevel | bool,
    theme: Theme | None = None,
    config: ToolRenderConfig | None = None,
) -> str:
    """Format any tool output using config or sensible defaults.

    This is the main entry point for tool formatting. It handles:
    - Custom formatters (for complex tools like edit/write)
    - Custom headers (for tools that want specific arg display)
    - Default formatting (works for any tool with zero config)

    Args:
        tool_name: Name of the tool
        args: Tool arguments
        result: Tool result dict
        detail_level: DetailLevel enum or bool for backward compat (True=EXPANDED, False=STANDARD)
        theme: Optional theme for styling
        config: Optional render config (uses defaults if None)

    Returns:
        Formatted string for TUI display.
    """
    # Import here to avoid circular imports
    from ..dtypes import DetailLevel, ToolRenderConfig

    if config is None:
        config = ToolRenderConfig()

    # Handle backward compatibility: bool -> DetailLevel
    if isinstance(detail_level, bool):
        detail_level = DetailLevel.EXPANDED if detail_level else DetailLevel.STANDARD

    # If custom formatter provided, use it exclusively
    if config.custom_formatter:
        return config.custom_formatter(tool_name, args, result, detail_level, theme)

    # Build header
    if config.header_fn:
        header = config.header_fn(tool_name, args)
    else:
        header = _default_header(tool_name, args)

    return format_tool_output(header, result, detail_level, theme, config)


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
