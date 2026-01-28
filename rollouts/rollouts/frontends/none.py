"""NoneFrontend - simple stdout-only frontend.

This is the simplest frontend implementation - it just prints to stdout
with no terminal manipulation. Useful for:
- Scripting/automation where you want plain text output
- Debugging without TUI interference
- Piping output to files or other processes
- Environments without terminal support (e.g., CI)

Note: Input uses simple input() which doesn't support multiline paste.
For multiline input support, use TUIFrontend instead.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import trio

if TYPE_CHECKING:
    from ..dtypes import StreamEvent, ToolCall


def _should_use_color() -> bool:
    """Determine if color output should be used.

    Follows the NO_COLOR standard (https://no-color.org/) and checks
    if stdout is a TTY.
    """
    # NO_COLOR env var disables color (any value, including empty string)
    if "NO_COLOR" in os.environ:
        return False
    # FORCE_COLOR enables color even if not a TTY
    if "FORCE_COLOR" in os.environ:
        return True
    # Default: use color only if stdout is a TTY
    return sys.stdout.isatty()


class NoneFrontend:
    """No TUI - just print to stdout.

    Minimal frontend that prints streaming text and tool calls to stdout.
    User input is read via standard input().

    Example usage:
        frontend = NoneFrontend()
        await run_interactive(trajectory, endpoint, frontend=frontend)
    """

    def __init__(
        self,
        show_tool_calls: bool = True,
        show_thinking: bool = False,
        verbose: bool = False,
        color: bool | None = None,
    ) -> None:
        """Initialize NoneFrontend.

        Args:
            show_tool_calls: Whether to print tool call info
            show_thinking: Whether to print thinking/reasoning tokens
            verbose: Show full tool args and results (default: compact one-liner)
            color: Force color on/off. None = auto-detect (respects NO_COLOR env var)
        """
        self.show_tool_calls = show_tool_calls
        self.show_thinking = show_thinking
        self.verbose = verbose
        self._use_color = color if color is not None else _should_use_color()
        self._after_tool = False
        # Track pending tool headers by tool_call_id for compact display
        self._pending_headers: dict[str, str] = {}
        self._tool_line_count = 0

    async def start(self) -> None:
        """No initialization needed for stdout."""
        pass

    async def stop(self) -> None:
        """No cleanup needed for stdout."""
        # Ensure final newline
        print("\n", end="", flush=True)

    def _format_compact_header(self, tool_name: str, args: dict) -> str:
        """Format tool call as compact one-liner: tool(key_arg=...)"""
        if not args:
            return f"{tool_name}()"

        # Show first/most important arg only, truncated
        key, value = next(iter(args.items()))
        if isinstance(value, str):
            # Remove newlines and truncate
            value = value.replace("\n", " ").replace("\r", "")
            display = value[:30] + "..." if len(value) > 30 else value
            display = repr(display)
        else:
            display = repr(value)
            if len(display) > 30:
                display = display[:30] + "..."

        if len(args) > 1:
            return f"{tool_name}({key}={display}, ...)"
        return f"{tool_name}({key}={display})"

    async def handle_event(self, event: StreamEvent) -> None:
        """Handle streaming event by printing to stdout.

        Args:
            event: Streaming event from agent loop
        """
        from ..dtypes import (
            RetryEnd,
            RetryStart,
            StreamDone,
            StreamError,
            StreamStart,
            TextDelta,
            ThinkingDelta,
            ToolCallEnd,
            ToolCallStart,
            ToolResultReceived,
        )

        # Colors (empty strings if color disabled)
        if self._use_color:
            LIGHT_GREY = "\033[37m"
            DARK_GREY = "\033[90m"
            YELLOW = "\033[33m"
            RED = "\033[31m"
            RESET = "\033[0m"
        else:
            LIGHT_GREY = DARK_GREY = YELLOW = RED = RESET = ""

        if isinstance(event, StreamStart):
            # StreamStart handled - agent prompt already shown by get_input
            pass

        elif isinstance(event, RetryStart):
            # Show retry status with error context
            error_hint = event.error_message[:60] if event.error_message else "transient error"
            print(
                f"{YELLOW}⟳ Retrying ({event.attempt}/{event.max_attempts}) "
                f"in {int(event.delay_seconds)}s - {error_hint}{RESET}",
                flush=True,
            )

        elif isinstance(event, RetryEnd):
            if not event.success:
                print(f"{RED}✗ Failed after {event.attempt} attempts{RESET}", flush=True)

        elif isinstance(event, TextDelta):
            # Print newline before text if we were showing tools
            if self._after_tool:
                print(flush=True)  # newline to separate tools from text
                self._after_tool = False
            print(f"{LIGHT_GREY}{event.delta}{RESET}", end="", flush=True)

        elif isinstance(event, ThinkingDelta) and self.show_thinking:
            print(f"{DARK_GREY}{event.delta}{RESET}", end="", flush=True)

        elif isinstance(event, ToolCallStart) and self.show_tool_calls:
            if self.verbose:
                # Newline before first tool if coming from text
                if not self._after_tool:
                    print(flush=True)
                print(f"{DARK_GREY}> {event.tool_name}{RESET}", end="", flush=True)
                self._after_tool = True

        elif isinstance(event, ToolCallEnd) and self.show_tool_calls:
            header = self._format_compact_header(event.tool_call.name, dict(event.tool_call.args))
            # Newline before first tool if coming from text
            if not self._after_tool:
                print(flush=True)
            if self.verbose:
                args_str = ", ".join(f"{k}={v!r}" for k, v in event.tool_call.args.items())
                if len(args_str) > 100:
                    args_str = args_str[:97] + "..."
                print(f"{DARK_GREY}  {event.tool_call.name}{RESET}({args_str})", flush=True)
            else:
                print(f"{DARK_GREY}  {header}{RESET}", flush=True)
            self._after_tool = True

        elif isinstance(event, ToolResultReceived) and self.show_tool_calls:
            if self.verbose:
                content = event.content if isinstance(event.content, str) else str(event.content)
                preview = content[:200] + "..." if len(content) > 200 else content
                prefix = f"{RED}Error:{RESET}" if event.is_error else "Result:"
                print(f"    {prefix} {preview}", flush=True)
            # Don't set _after_tool here - let ToolCallEnd control it

        elif isinstance(event, StreamDone):
            # Don't print newline here - StreamDone fires after each LLM call,
            # but tool results come after it in agentic loops. TextDelta handles
            # the newline when text resumes.
            pass

        elif isinstance(event, StreamError):
            print(f"\n{RED}Stream error: {event.error}{RESET}\n", end="", flush=True)

    async def get_input(self, prompt: str = "") -> str:
        """Get user input via stdin.

        TODO(multiline): The raw terminal multiline input implementation had rendering
        bugs (ghost lines appearing when text wrapped). For now, we use simple input()
        which doesn't support multiline paste. To fix this properly, we need to either:
        1. Extract the TUI's Input component rendering logic for use in stdout mode
        2. Or use a full-screen approach like the TUI does (which avoids the cursor
           positioning issues that cause ghost lines)
        See test files in repo root (test_*.py) for investigation history.

        Args:
            prompt: Prompt to display

        Returns:
            User's input string
        """
        # Ensure we're on a new line before prompt
        print("\n", end="", flush=True)

        display_prompt = prompt if prompt else "> "

        def _get_input() -> str:
            try:
                return input(display_prompt)
            except EOFError as e:
                # stdin closed (e.g., running from Claude Code without TTY)
                # Raise KeyboardInterrupt to signal graceful exit
                raise KeyboardInterrupt("stdin closed (EOF)") from e

        result = await trio.to_thread.run_sync(_get_input, abandon_on_cancel=True)

        # Show agent prompt immediately so user knows their input was received
        if self._use_color:
            print("\033[32m❯\033[0m ", end="", flush=True)
        else:
            print("❯ ", end="", flush=True)

        return result

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

        if self._use_color:
            print(
                f"\n\033[33m⚠️  Confirm tool: {tool_call.name}({args_str})\033[0m\n",
                end="",
                flush=True,
            )
        else:
            print(f"\n⚠️  Confirm tool: {tool_call.name}({args_str})\n", end="", flush=True)
        print("  [y] approve  [n] reject  [Enter=approve]\n", end="", flush=True)

        response = await trio.to_thread.run_sync(input, "  > ")
        return response.lower() in ("", "y", "yes")

    def show_loader(self, text: str) -> None:
        """Show loading text (no-op for NoneFrontend, we use StreamStart instead)."""
        pass

    def hide_loader(self) -> None:
        """Hide loading text (no-op for NoneFrontend)."""
        pass
