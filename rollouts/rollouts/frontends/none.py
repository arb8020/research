"""NoneFrontend - simple stdout-only frontend.

This is the simplest frontend implementation - it just prints to stdout
with no terminal manipulation. Useful for:
- Scripting/automation where you want plain text output
- Debugging without TUI interference
- Piping output to files or other processes
- Environments without terminal support (e.g., CI)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import trio

if TYPE_CHECKING:
    from rollouts.dtypes import StreamEvent, ToolCall


class NoneFrontend:
    """No TUI - just print to stdout.

    Minimal frontend that prints streaming text and tool calls to stdout.
    User input is read via standard input().

    Example usage:
        frontend = NoneFrontend()
        await run_interactive(trajectory, endpoint, frontend=frontend)
    """

    def __init__(self, show_tool_calls: bool = True, show_thinking: bool = False):
        """Initialize NoneFrontend.

        Args:
            show_tool_calls: Whether to print tool call info
            show_thinking: Whether to print thinking/reasoning tokens
        """
        self.show_tool_calls = show_tool_calls
        self.show_thinking = show_thinking
        self._current_loader_text: str | None = None

    async def start(self) -> None:
        """No initialization needed for stdout."""
        pass

    async def stop(self) -> None:
        """No cleanup needed for stdout."""
        # Ensure final newline
        print(flush=True)

    async def handle_event(self, event: StreamEvent) -> None:
        """Handle streaming event by printing to stdout.

        Args:
            event: Streaming event from agent loop
        """
        from rollouts.dtypes import (
            StreamDone,
            StreamError,
            TextDelta,
            ThinkingDelta,
            ToolCallEnd,
            ToolCallStart,
            ToolResultReceived,
        )

        if isinstance(event, TextDelta):
            print(event.delta, end="", flush=True)

        elif isinstance(event, ThinkingDelta) and self.show_thinking:
            print(f"\033[2m{event.delta}\033[0m", end="", flush=True)

        elif isinstance(event, ToolCallStart) and self.show_tool_calls:
            print(f"\n\033[1m[Tool: {event.tool_name}]\033[0m", flush=True)

        elif isinstance(event, ToolCallEnd) and self.show_tool_calls:
            args_str = ", ".join(f"{k}={v!r}" for k, v in event.tool_call.args.items())
            if len(args_str) > 100:
                args_str = args_str[:97] + "..."
            print(f"  Args: {args_str}", flush=True)

        elif isinstance(event, ToolResultReceived) and self.show_tool_calls:
            preview = event.content[:200] + "..." if len(event.content) > 200 else event.content
            prefix = "\033[31mError:\033[0m" if event.is_error else "Result:"
            print(f"  {prefix} {preview}", flush=True)

        elif isinstance(event, StreamDone):
            print(flush=True)

        elif isinstance(event, StreamError):
            print(f"\n\033[31mStream error: {event.error}\033[0m", file=sys.stderr)

    async def get_input(self, prompt: str = "") -> str:
        """Get user input via stdin.

        Uses trio.to_thread to avoid blocking the event loop.

        Args:
            prompt: Prompt to display

        Returns:
            User's input string
        """
        # Ensure we're on a new line before prompt
        print()

        # Use trio's thread runner to avoid blocking
        if prompt:
            return await trio.to_thread.run_sync(input, prompt)
        else:
            return await trio.to_thread.run_sync(input, "> ")

    async def confirm_tool(self, tool_call: ToolCall) -> bool:
        """Confirm tool execution via stdin.

        Args:
            tool_call: Tool call to confirm

        Returns:
            True if approved, False if rejected
        """
        args_str = ", ".join(f"{k}={v!r}" for k, v in tool_call.args.items())
        if len(args_str) > 100:
            args_str = args_str[:97] + "..."

        print(f"\n\033[33m⚠️  Confirm tool: {tool_call.name}({args_str})\033[0m")
        print("  [y] approve  [n] reject  [Enter=approve]")

        response = await trio.to_thread.run_sync(input, "  > ")
        return response.lower() in ("", "y", "yes")

    def show_loader(self, text: str) -> None:
        """Show loading text (simple print).

        Args:
            text: Loading status text
        """
        self._current_loader_text = text
        print(f"\033[2m{text}\033[0m", end="\r", flush=True)

    def hide_loader(self) -> None:
        """Clear loading text."""
        if self._current_loader_text:
            # Clear the line
            print(" " * len(self._current_loader_text), end="\r", flush=True)
            self._current_loader_text = None
