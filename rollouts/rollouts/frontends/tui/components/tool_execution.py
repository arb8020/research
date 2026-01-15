"""
Tool execution component - displays tool calls with arguments and results.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..tui import Container
from .text import Text

if TYPE_CHECKING:
    from ....dtypes import DetailLevel


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
        render_config: Any | None = None,  # ToolRenderConfig, Any to avoid circular import
    ) -> None:
        """Initialize tool execution component.

        Args:
            tool_name: Name of the tool
            args: Tool arguments (may be partial during streaming)
            bg_fn_pending: Background color function for pending state
            bg_fn_success: Background color function for success state
            bg_fn_error: Background color function for error state
            theme: Theme for styling
            formatter: Legacy formatter function (prefer render_config)
            render_config: ToolRenderConfig from environment (preferred)
        """
        super().__init__()
        self._tool_name = tool_name
        self._args = args or {}
        self._result: dict[str, Any] | None = None
        # Import here to avoid circular imports at module level
        from ....dtypes import DetailLevel

        self._detail_level: DetailLevel = DetailLevel.STANDARD
        self._theme = theme
        self._formatter = formatter
        self._render_config = render_config

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
        """Set whether to show expanded output (legacy, prefer set_detail_level)."""
        from ....dtypes import DetailLevel

        self._detail_level = DetailLevel.EXPANDED if expanded else DetailLevel.STANDARD
        self._rebuild_display()

    def set_detail_level(self, level: DetailLevel) -> None:
        """Set the detail level for output display."""
        self._detail_level = level
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

        Priority:
        1. render_config (new, preferred)
        2. formatter (legacy)
        3. default formatting (works for any tool)
        """
        from ....environments._formatting import format_tool

        # Use render_config if provided (preferred)
        if self._render_config:
            return format_tool(
                self._tool_name,
                self._args,
                self._result,
                self._detail_level,
                self._theme,
                self._render_config,
            )

        # Use legacy formatter if provided
        if self._formatter:
            # Legacy formatters expect bool, convert detail_level
            from ....dtypes import DetailLevel

            expanded_bool = self._detail_level >= DetailLevel.EXPANDED
            return self._formatter(
                self._tool_name, self._args, self._result, expanded_bool, self._theme
            )

        # Default formatting - works for any tool with zero config
        return format_tool(
            self._tool_name, self._args, self._result, self._detail_level, self._theme
        )
