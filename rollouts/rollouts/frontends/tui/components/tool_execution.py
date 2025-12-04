"""
Tool execution component - displays tool calls with arguments and results.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from ..tui import Component, Container
from .spacer import Spacer
from .text import Text
from ..utils import visible_width


def _shorten_path(path: str) -> str:
    """Convert absolute path to tilde notation if in home directory."""
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def _replace_tabs(text: str) -> str:
    """Replace tabs with spaces for consistent rendering."""
    return text.replace("\t", "   ")


class ToolExecution(Container):
    """Component that renders a tool call with its result (updateable)."""

    def __init__(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        bg_fn_pending: Optional[Callable[[str], str]] = None,
        bg_fn_success: Optional[Callable[[str], str]] = None,
        bg_fn_error: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Initialize tool execution component.

        Args:
            tool_name: Name of the tool
            args: Tool arguments (may be partial during streaming)
            bg_fn_pending: Background color function for pending state
            bg_fn_success: Background color function for success state
            bg_fn_error: Background color function for error state
        """
        super().__init__()
        self._tool_name = tool_name
        self._args = args or {}
        self._result: Optional[Dict[str, Any]] = None
        self._expanded = False

        # Default background functions (can be overridden)
        self._bg_fn_pending = bg_fn_pending or (lambda x: x)
        self._bg_fn_success = bg_fn_success or (lambda x: x)
        self._bg_fn_error = bg_fn_error or (lambda x: x)

        self._content_text: Optional[Text] = None
        self._rebuild_display()

    def update_args(self, args: Dict[str, Any]) -> None:
        """Update tool arguments (called during streaming)."""
        self._args = args
        self._rebuild_display()

    def update_result(
        self,
        result: Dict[str, Any],
        is_error: bool = False,
    ) -> None:
        """Update tool result.

        Args:
            result: Result data (may contain 'content' list or 'text' string)
            is_error: Whether this is an error result
        """
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
        self._content_text = Text(formatted_text, padding_x=2, padding_y=1, custom_bg_fn=bg_fn, gutter_prefix="🤖")
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
                text_blocks = [c for c in content_list if isinstance(c, dict) and c.get("type") == "text"]
                text_output = "\n".join(c.get("text", "") for c in text_blocks if c.get("text"))
                
                # Strip ANSI codes and carriage returns
                import re
                text_output = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text_output)  # Strip ANSI
                text_output = text_output.replace("\r", "")  # Strip carriage returns
                return text_output

        return ""

    def _format_tool_execution(self) -> str:
        """Format tool execution display based on tool type."""
        text = ""

        if self._tool_name == "bash":
            command = self._args.get("command", "")
            text = f"bash(command={repr(command or '...')})"

            if self._result:
                output = self._get_text_output().strip()
                if output:
                    lines = output.split("\n")
                    max_lines = len(lines) if self._expanded else 5
                    display_lines = lines[:max_lines]
                    remaining = len(lines) - max_lines

                    # Add summary line with ⎿, then indented content
                    is_error = self._result.get("isError", False)
                    summary = "Command failed" if is_error else "Command completed"
                    text += f"\n⎿ {summary}"
                    for line in display_lines:
                        text += "\n  " + line
                    if remaining > 0:
                        text += f"\n  ... ({remaining} more lines)"

        elif self._tool_name == "read":
            path = _shorten_path(self._args.get("file_path") or self._args.get("path") or "")
            offset = self._args.get("offset")
            limit = self._args.get("limit")

            params = f"file_path={repr(path if path else '...')}"
            if offset is not None:
                params += f", offset={offset}"
            if limit is not None:
                params += f", limit={limit}"

            text = f"read({params})"

            if self._result:
                output = self._get_text_output()
                lines = output.split("\n")
                max_lines = len(lines) if self._expanded else 10
                display_lines = lines[:max_lines]
                remaining = len(lines) - max_lines

                # Add summary line with ⎿, then indented content
                total_lines = len(lines)
                summary = f"Read {total_lines} line{'s' if total_lines != 1 else ''}"
                text += f"\n⎿ {summary}"
                for line in display_lines:
                    text += "\n  " + _replace_tabs(line)
                if remaining > 0:
                    text += f"\n  ... ({remaining} more lines)"

        elif self._tool_name == "write":
            path = _shorten_path(self._args.get("file_path") or self._args.get("path") or "")
            file_content = self._args.get("content", "")
            lines = file_content.split("\n") if file_content else []
            total_lines = len(lines)

            text = f"write(file_path={repr(path if path else '...')})"

            if file_content:
                max_lines = len(lines) if self._expanded else 10
                display_lines = lines[:max_lines]
                remaining = len(lines) - max_lines

                # Add summary line with ⎿, then indented content
                summary = f"Wrote {total_lines} line{'s' if total_lines != 1 else ''} to {path or '...'}"
                text += f"\n⎿ {summary}"
                for line in display_lines:
                    text += "\n  " + _replace_tabs(line)
                if remaining > 0:
                    text += f"\n  ... ({remaining} more lines)"

        elif self._tool_name == "edit":
            path = _shorten_path(self._args.get("file_path") or self._args.get("path") or "")
            old_string = self._args.get("old_string", "")
            new_string = self._args.get("new_string", "")

            text = f"edit(file_path={repr(path if path else '...')}, old_string=..., new_string=...)"

            if self._result:
                output = self._get_text_output()
                if output:
                    # Add summary line with ⎿, then indented content
                    is_error = self._result.get("isError", False)
                    summary = "Edit failed" if is_error else f"Updated {path or '...'}"
                    text += f"\n⎿ {summary}"
                    lines = output.split("\n")
                    for line in lines:
                        text += "\n  " + line

        else:
            # Generic tool - show name(params) with robot emoji prefix
            if self._args:
                params_list = []
                for key, value in self._args.items():
                    if isinstance(value, str):
                        params_list.append(f"{key}={repr(value)}")
                    else:
                        params_list.append(f"{key}={value}")
                params_str = ", ".join(params_list)
                text = f"{self._tool_name}({params_str})"
            else:
                text = f"{self._tool_name}()"

            output = self._get_text_output()
            if output:
                text += "\n\n" + output

        return text

