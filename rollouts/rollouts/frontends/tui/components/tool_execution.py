"""
Tool execution component - displays tool calls with arguments and results.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..tui import Container
from .text import Text


class ToolExecution(Container):
    """Component that renders a tool call with its result (updateable)."""

    def __init__(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        bg_fn_pending: Callable[[str], str] | None = None,
        bg_fn_success: Callable[[str], str] | None = None,
        bg_fn_error: Callable[[str], str] | None = None,
        theme: Any | None = None,
        formatter: Callable[[str, dict[str, Any], dict[str, Any] | None, bool, Any], str]
        | None = None,
    ) -> None:
        """Initialize tool execution component.

        Args:
            tool_name: Name of the tool
            args: Tool arguments (may be partial during streaming)
            bg_fn_pending: Background color function for pending state
            bg_fn_success: Background color function for success state
            bg_fn_error: Background color function for error state
            theme: Theme for styling
            formatter: Formatter function from environment: (tool_name, args, result, expanded, theme) -> str
        """
        super().__init__()
        self._tool_name = tool_name
        self._args = args or {}
        self._result: dict[str, Any] | None = None
        self._expanded = False
        self._theme = theme
        self._formatter = formatter

        # Default background functions (can be overridden)
        self._bg_fn_pending = bg_fn_pending or (lambda x: x)
        self._bg_fn_success = bg_fn_success or (lambda x: x)
        self._bg_fn_error = bg_fn_error or (lambda x: x)

        self._content_text: Text | None = None
        self._rebuild_display()

    def update_args(self, args: dict[str, Any]) -> None:
        """Update tool arguments (called during streaming)."""
        self._args = args
        self._rebuild_display()

    def update_result(
        self,
        result: dict[str, Any],
        is_error: bool = False,
    ) -> None:
        """Update tool result.

        Args:
            result: Result data (may contain 'content' list, 'details' dict, or 'text' string)
            is_error: Whether this is an error result
        """
        # Don't double-wrap if result already has expected structure
        # Expected structure: {"content": [...], "details": {...}}
        # We add isError at top level
        if "content" in result or "details" in result:
            self._result = {**result, "isError": is_error}
        else:
            # Legacy: wrap in content
            self._result = {"content": result, "isError": is_error}
        self._rebuild_display()

    def set_expanded(self, expanded: bool) -> None:
        """Set whether to show expanded output."""
        self._expanded = expanded
        self._rebuild_display()

    def _rebuild_display(self) -> None:
        """Rebuild the display from current state."""
        self.clear()

        # Determine background function based on state
        if self._result:
            bg_fn = self._bg_fn_error if self._result.get("isError") else self._bg_fn_success
        else:
            bg_fn = self._bg_fn_pending

        # Format tool execution text
        formatted_text = self._format_tool_execution()

        # Create text component with background and gutter prefix
        # Get gutter and padding from theme if available
        if self._theme:
            if self._result:
                gutter = (
                    self._theme.tool_error_gutter
                    if self._result.get("isError")
                    else self._theme.tool_success_gutter
                )
            else:
                gutter = self._theme.tool_success_gutter  # Pending state uses success gutter
            padding_y = self._theme.tool_padding_y if hasattr(self._theme, "tool_padding_y") else 0
        else:
            # Fallback if no theme provided
            gutter = "☹ " if (self._result and self._result.get("isError")) else "☺ "
            padding_y = 0

        self._content_text = Text(
            formatted_text,
            padding_x=2,
            padding_y=padding_y,
            custom_bg_fn=bg_fn,
            gutter_prefix=gutter,
        )
        self.add_child(self._content_text)

    def _get_text_output(self) -> str:
        """Extract text output from result."""
        if not self._result:
            return ""

        content = self._result.get("content", {})

        # If content is a string, return it directly
        if isinstance(content, str):
            return content

        # If content is a dict with a "content" key, extract from that
        if isinstance(content, dict):
            content_list = content.get("content", [])
            if isinstance(content_list, list):
                text_blocks = [
                    c for c in content_list if isinstance(c, dict) and c.get("type") == "text"
                ]
                text_output = "\n".join(c.get("text", "") for c in text_blocks if c.get("text"))

                # Strip ANSI codes and carriage returns
                import re

                text_output = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text_output)  # Strip ANSI
                text_output = text_output.replace("\r", "")  # Strip carriage returns
                return text_output

        return ""

    def _format_tool_execution(self) -> str:
        """Format tool execution display.

        Uses environment-provided formatter if available, otherwise falls back
        to a generic format showing tool name and arguments.
        """
        # Use environment formatter if provided
        if self._formatter:
            return self._formatter(
                self._tool_name, self._args, self._result, self._expanded, self._theme
            )

        # Generic fallback for tools without a formatter
        return self._format_generic()

    def _format_generic(self) -> str:
        """Generic tool formatter - shows name(params) and output."""
        if self._args:
            params_list = []
            for key, value in self._args.items():
                if isinstance(value, str):
                    # Truncate long strings
                    display_value = value if len(value) <= 50 else value[:47] + "..."
                    params_list.append(f"{key}={repr(display_value)}")
                else:
                    params_list.append(f"{key}={value}")
            params_str = ", ".join(params_list)
            text = f"{self._tool_name}({params_str})"
        else:
            text = f"{self._tool_name}()"

        if self._result:
            output = self._get_text_output()
            if output:
                is_error = self._result.get("isError", False)
                summary = "Failed" if is_error else "Completed"
                text += f"\n⎿ {summary}"
                lines = output.split("\n")
                max_lines = len(lines) if self._expanded else 10
                for line in lines[:max_lines]:
                    text += "\n  " + line
                remaining = len(lines) - max_lines
                if remaining > 0:
                    text += f"\n  ... ({remaining} more lines)"

        return text
