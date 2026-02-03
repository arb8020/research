"""MinimalFrontend - OpenCode-style minimal output with icons.

A clean, minimal frontend inspired by OpenCode's redesigned run command.
Features:
- Icon-based tool display (e.g., "* Glob '*.ts' - 5 matches")
- Inline (single-line) vs block (with output) formatting
- TTY-aware output (styled for terminal, raw for pipes)
- Compact diff/edit display

Icons:
- * glob/grep (search)
- > read/list (input)
- < edit/write (output)
- $ bash (shell)
- # todo (tasks)
- % webfetch (network)
- @ task (subagent)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import TYPE_CHECKING

import trio

if TYPE_CHECKING:
    from ..dtypes import StreamEvent, ToolCall


def _should_use_color() -> bool:
    """Determine if color output should be used."""
    if "NO_COLOR" in os.environ:
        return False
    if "FORCE_COLOR" in os.environ:
        return True
    return sys.stdout.isatty()


class MinimalFrontend:
    """OpenCode-style minimal frontend with icon-based tool display.

    Provides a clean, compact output format suitable for:
    - Non-interactive scripting
    - Clean terminal output without TUI
    - Piping to files while maintaining readability

    Example output:
        > agent - claude-sonnet-4

        * Glob "*.py" in src/ - 12 matches
        > Read src/main.py
        < Edit src/main.py
          @@ -10,3 +10,5 @@
          +    new_line = True
        $ pytest tests/
          PASSED (3 tests)

        Done! Created the new feature.
    """

    # Tool icons (Unicode, matching OpenCode style)
    ICONS = {
        # Search tools
        "glob": "✱",
        "grep": "✱",
        "ripgrep": "✱",
        # Read tools
        "read": "→",
        "list": "→",
        "ls": "→",
        "skill": "→",
        # Write tools
        "edit": "←",
        "write": "←",
        "patch": "←",
        # Shell
        "bash": "$",
        "shell": "$",
        "computer": "$",
        # Tasks/agents
        "task_pending": "•",
        "task_done": "✓",
        "dispatch_agent": "•",
        "todowrite": "#",
        "todoread": "#",
        # Network
        "webfetch": "%",
        "web_search": "◈",
        "codesearch": "◇",
        "mcp": "%",
        # Default
        "default": "·",
    }

    # Tools that show output in a block (vs inline single-line)
    # Most tools now show inline summaries; only todowrite needs block for checklist
    BLOCK_TOOLS = {"todowrite"}

    def __init__(
        self,
        show_tool_calls: bool = True,
        show_thinking: bool = False,
        show_output: bool = True,
        color: bool | None = None,
        agent: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize MinimalFrontend.

        Args:
            show_tool_calls: Whether to print tool call info
            show_thinking: Whether to print thinking/reasoning tokens
            show_output: Whether to show tool output (for block tools)
            color: Force color on/off. None = auto-detect
            agent: Agent name to show in header (e.g., "coder")
            model: Model name to show in header (e.g., "claude-sonnet-4")
        """
        self.show_tool_calls = show_tool_calls
        self.show_thinking = show_thinking
        self.show_output = show_output
        self._use_color = color if color is not None else _should_use_color()
        self._is_tty = sys.stdout.isatty()

        # Header info
        self._agent = agent
        self._model = model
        self._header_shown = False

        # State tracking
        self._after_tool = False
        self._pending_results: dict[str, dict] = {}  # tool_call_id -> {name, args}

        # Buffers for complete-before-print (like OpenCode)
        self._text_buffer: list[str] = []
        self._thinking_buffer: list[str] = []

        # Spinner state (uses threading for smooth animation)
        self._spinner_active = False
        self._spinner_stop_event: threading.Event | None = None
        self._spinner_thread: threading.Thread | None = None
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0

    # -------------------------------------------------------------------------
    # Color/Style helpers
    # -------------------------------------------------------------------------

    def _style(self, text: str, code: str) -> str:
        """Apply ANSI style if color enabled."""
        if not self._use_color:
            return text
        return f"{code}{text}\033[0m"

    def _dim(self, text: str) -> str:
        return self._style(text, "\033[90m")

    def _bold(self, text: str) -> str:
        return self._style(text, "\033[1m")

    def _green(self, text: str) -> str:
        return self._style(text, "\033[32m")

    def _yellow(self, text: str) -> str:
        return self._style(text, "\033[33m")

    def _red(self, text: str) -> str:
        return self._style(text, "\033[31m")

    def _italic(self, text: str) -> str:
        return self._style(text, "\033[3m")

    # -------------------------------------------------------------------------
    # Output helpers
    # -------------------------------------------------------------------------

    def _print(self, *args: str, end: str = "\n") -> None:
        """Print with flush."""
        print(*args, end=end, flush=True)

    def _empty(self) -> None:
        """Print empty line."""
        self._print()

    def _clear_line(self) -> None:
        """Clear current line (for spinner)."""
        if self._is_tty:
            print("\r\033[K", end="", flush=True)

    def _spinner_loop(self) -> None:
        """Background thread that animates the spinner."""
        while self._spinner_stop_event and not self._spinner_stop_event.is_set():
            frame = self._spinner_frames[self._spinner_idx % len(self._spinner_frames)]
            print(f"\r{self._dim(frame)}", end="", flush=True)
            self._spinner_idx += 1
            time.sleep(0.08)  # ~12 fps

    def _start_spinner(self) -> None:
        """Start the spinner in a background thread."""
        if not self._is_tty or self._spinner_active:
            return
        self._spinner_active = True
        self._spinner_idx = 0
        self._spinner_stop_event = threading.Event()
        self._spinner_thread = threading.Thread(target=self._spinner_loop, daemon=True)
        self._spinner_thread.start()

    def _stop_spinner(self) -> None:
        """Stop and clear the spinner."""
        if not self._spinner_active:
            return
        self._spinner_active = False
        if self._spinner_stop_event:
            self._spinner_stop_event.set()
        if self._spinner_thread:
            self._spinner_thread.join(timeout=0.2)
            self._spinner_thread = None
        self._spinner_stop_event = None
        self._clear_line()

    def _show_header(self) -> None:
        """Show agent/model header on first output."""
        if self._header_shown:
            return
        self._header_shown = True

        if self._agent or self._model:
            parts = []
            if self._agent:
                parts.append(self._agent)
            if self._model:
                parts.append(self._model)
            header = " · ".join(parts)
            self._empty()
            self._print(f"> {header}")
            self._empty()

    def _inline(self, icon: str, title: str, description: str | None = None) -> None:
        """Print inline tool output (single line, no indent like OpenCode)."""
        suffix = f" {self._dim(description)}" if description else ""
        self._print(f"{icon} {title}{suffix}")

    def _block(self, icon: str, title: str, output: str | None = None) -> None:
        """Print block tool output (with content below, no indent like OpenCode)."""
        self._empty()
        self._inline(icon, title)
        if output and self.show_output:
            self._print(output.strip())
        self._empty()

    # -------------------------------------------------------------------------
    # Tool formatters
    # -------------------------------------------------------------------------

    def _get_icon(self, tool_name: str) -> str:
        """Get icon for tool."""
        name = tool_name.lower()
        return self.ICONS.get(name, self.ICONS["default"])

    def _is_block_tool(self, tool_name: str) -> bool:
        """Check if tool should use block format."""
        return tool_name.lower() in self.BLOCK_TOOLS

    def _format_glob(self, args: dict, result: str | None) -> tuple[str, str | None]:
        """Format glob tool."""
        pattern = args.get("pattern", "")
        path = args.get("path", "")

        title = f'Glob "{pattern}"'
        suffix = f"in {self._normalize_path(path)}" if path else ""

        # Try to extract match count from result
        count_str = ""
        if result:
            lines = result.strip().split("\n")
            count = len([l for l in lines if l.strip()])
            count_str = f"{count} {'match' if count == 1 else 'matches'}"

        # Format: suffix · count (like OpenCode)
        if suffix and count_str:
            desc = f"{suffix} · {count_str}"
        elif count_str:
            desc = count_str
        elif suffix:
            desc = suffix
        else:
            desc = None

        return title, desc

    def _format_grep(self, args: dict, result: str | None) -> tuple[str, str | None]:
        """Format grep tool."""
        pattern = args.get("pattern", "")
        path = args.get("path", "")

        title = f'Grep "{pattern}"'
        suffix = f"in {self._normalize_path(path)}" if path else ""

        count_str = ""
        if result:
            lines = result.strip().split("\n")
            count = len([l for l in lines if l.strip()])
            count_str = f"{count} {'match' if count == 1 else 'matches'}"

        # Format: suffix · count (like OpenCode)
        if suffix and count_str:
            desc = f"{suffix} · {count_str}"
        elif count_str:
            desc = count_str
        elif suffix:
            desc = suffix
        else:
            desc = None

        return title, desc

    def _format_read(self, args: dict, result: str | None) -> tuple[str, str | None]:
        """Format read tool."""
        path = args.get("file_path") or args.get("path", "")
        title = f"Read {self._normalize_path(path)}"

        # Show offset/limit if specified
        extras = []
        if args.get("offset"):
            extras.append(f"offset={args['offset']}")
        if args.get("limit"):
            extras.append(f"limit={args['limit']}")

        desc = f"[{', '.join(extras)}]" if extras else None
        return title, desc

    def _format_edit(self, args: dict, result: str | None) -> tuple[str, str | None]:
        """Format edit tool with +N, -M lines summary."""
        path = args.get("file_path") or args.get("path", "")
        title = f"Edit {self._normalize_path(path)}"

        # Parse diff to get line counts
        additions = 0
        deletions = 0
        if result:
            for line in result.split("\n"):
                if line.startswith("+") and not line.startswith("+++"):
                    additions += 1
                elif line.startswith("-") and not line.startswith("---"):
                    deletions += 1

        if additions or deletions:
            desc = f"+{additions}, -{deletions} lines"
        else:
            desc = None
        return title, desc

    def _format_write(self, args: dict, result: str | None) -> tuple[str, str | None]:
        """Format write tool."""
        path = args.get("file_path") or args.get("path", "")
        content = args.get("content", "")
        lines = len(content.split("\n")) if content else 0
        title = f"Write {self._normalize_path(path)}"
        desc = f"{lines} lines"
        return title, desc

    def _format_bash(self, args: dict, result: str | None) -> tuple[str, str | None]:
        """Format bash tool with output summary."""
        cmd = args.get("command", "")
        # Truncate long commands
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."

        # Parse result for summary
        # Result format varies, but we can count lines
        if result:
            lines = result.strip().split("\n")
            line_count = len(lines) if lines != [""] else 0
            desc = f"{line_count} lines"
        else:
            desc = None
        return cmd, desc

    def _format_task(self, args: dict, result: str | None) -> tuple[str, str | None]:
        """Format task/subagent tool."""
        desc = args.get("description", "")
        agent = args.get("subagent_type", "agent")
        title = desc or f"{agent} task"
        suffix = f"{agent} agent" if desc else None
        return title, suffix

    def _format_todo(self, args: dict, result: str | None) -> tuple[str, str | None]:
        """Format todo tool."""
        todos = args.get("todos", [])
        if not todos:
            return "Todos", None

        lines = []
        for item in todos:
            status = item.get("status", "pending")
            content = item.get("content", "")
            marker = "[x]" if status == "completed" else "[ ]"
            lines.append(f"{marker} {content}")

        return "Todos", "\n".join(lines)

    def _format_default(self, name: str, args: dict, result: str | None) -> tuple[str, str | None]:
        """Default tool formatter."""
        if not args:
            return name, None

        # Show first arg, truncated
        key, value = next(iter(args.items()))
        if isinstance(value, str):
            value = value.replace("\n", " ")[:30]
            if len(value) == 30:
                value += "..."
            return f'{name} {key}="{value}"', None

        return f"{name} {key}={value!r}"[:50], None

    def _normalize_path(self, path: str) -> str:
        """Normalize path for display (relative if possible)."""
        if not path:
            return ""
        import os.path

        if os.path.isabs(path):
            try:
                return os.path.relpath(path)
            except ValueError:
                return path
        return path

    def _format_tool(self, name: str, args: dict, result: str | None = None) -> None:
        """Format and print a tool call."""
        icon = self._get_icon(name)
        name_lower = name.lower()

        # Route to specific formatter
        if name_lower in ("glob",):
            title, desc = self._format_glob(args, result)
        elif name_lower in ("grep", "ripgrep"):
            title, desc = self._format_grep(args, result)
        elif name_lower in ("read",):
            title, desc = self._format_read(args, result)
        elif name_lower in ("edit", "patch"):
            title, desc = self._format_edit(args, result)
        elif name_lower in ("write",):
            title, desc = self._format_write(args, result)
        elif name_lower in ("bash", "shell", "computer"):
            title, desc = self._format_bash(args, result)
        elif name_lower in ("task", "dispatch_agent"):
            title, desc = self._format_task(args, result)
        elif name_lower in ("todowrite",):
            title, desc = self._format_todo(args, result)
        else:
            title, desc = self._format_default(name, args, result)

        # Print as block or inline
        if self._is_block_tool(name) and desc:
            self._block(icon, title, desc)
        else:
            self._inline(icon, title, desc)

    # -------------------------------------------------------------------------
    # Frontend protocol
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Show header on start."""
        self._show_header()

    async def stop(self) -> None:
        """Ensure final newline."""
        self._print()

    def _flush_thinking(self) -> None:
        """Flush thinking buffer and print."""
        if not self._thinking_buffer:
            return
        text = "".join(self._thinking_buffer).strip()
        self._thinking_buffer.clear()
        if not text:
            return

        # Print newline before thinking if we were showing tools
        if self._after_tool:
            self._empty()
            self._after_tool = False

        # OpenCode style: "Thinking: {text}" in italic+dim
        line = f"Thinking: {text}"
        self._empty()
        self._print(self._dim(self._italic(line)))
        self._empty()

    def _flush_text(self) -> None:
        """Flush text buffer and print."""
        if not self._text_buffer:
            return
        text = "".join(self._text_buffer).strip()
        self._text_buffer.clear()
        if not text:
            return

        # Print newline before text if we were showing tools
        if self._after_tool:
            self._empty()
            self._after_tool = False

        self._empty()
        self._print(text)
        self._empty()

    async def handle_event(self, event: StreamEvent) -> None:
        """Handle streaming event."""
        from ..dtypes import (
            LLMCallStart,
            RetryEnd,
            RetryStart,
            StreamDone,
            StreamError,
            StreamStart,
            TextDelta,
            TextEnd,
            ThinkingDelta,
            ThinkingEnd,
            ToolCallEnd,
            ToolCallStart,
            ToolResultReceived,
        )

        if isinstance(event, LLMCallStart):
            # LLM call starting - spinner already running from get_input()
            pass

        elif isinstance(event, StreamStart):
            # Stream starting - keep spinner running
            pass

        elif isinstance(event, RetryStart):
            self._stop_spinner()
            error_hint = event.error_message[:60] if event.error_message else "transient error"
            self._print(
                f"{self._yellow('~')} Retrying ({event.attempt}/{event.max_attempts}) "
                f"in {int(event.delay_seconds)}s - {self._dim(error_hint)}"
            )
            # Restart spinner for retry
            self._start_spinner()

        elif isinstance(event, RetryEnd):
            if not event.success:
                self._stop_spinner()
                self._print(f"{self._red('!')} Failed after {event.attempt} attempts")

        elif isinstance(event, TextDelta):
            # Buffer text until complete
            self._text_buffer.append(event.delta)

        elif isinstance(event, TextEnd):
            # Stop spinner before printing text
            self._stop_spinner()
            self._text_buffer.clear()  # Clear buffer, use complete content
            text = event.content.strip()
            if text:
                if self._after_tool:
                    self._empty()
                    self._after_tool = False
                self._empty()
                self._print(text)
                self._empty()

        elif isinstance(event, ThinkingDelta) and self.show_thinking:
            # Buffer thinking until complete
            self._thinking_buffer.append(event.delta)

        elif isinstance(event, ThinkingEnd) and self.show_thinking:
            # Stop spinner before printing thinking
            self._stop_spinner()
            self._thinking_buffer.clear()  # Clear buffer, use complete content
            text = event.content.strip()
            if text:
                if self._after_tool:
                    self._empty()
                    self._after_tool = False
                line = f"Thinking: {text}"
                self._empty()
                self._print(self._dim(self._italic(line)))
                self._empty()

        elif isinstance(event, ToolCallStart) and self.show_tool_calls:
            # Store for later when we have args
            self._pending_results[event.tool_call_id] = {"name": event.tool_name, "args": {}}

        elif isinstance(event, ToolCallEnd) and self.show_tool_calls:
            name = event.tool_call.name
            args = dict(event.tool_call.args)

            # Store for result matching
            self._pending_results[event.tool_call.id] = {"name": name, "args": args}

            # Don't print yet for block tools - wait for result
            if not self._is_block_tool(name):
                # Stop spinner before printing tool
                self._stop_spinner()
                self._format_tool(name, args)
                self._after_tool = True
                # Restart spinner for next operation
                self._start_spinner()

        elif isinstance(event, ToolResultReceived) and self.show_tool_calls:
            # Match with pending tool call
            pending = self._pending_results.pop(event.tool_call_id, None)
            if pending:
                name = pending["name"]
                args = pending["args"]
                result = event.content if isinstance(event.content, str) else str(event.content)

                # For block tools, now we can print with output
                if self._is_block_tool(name):
                    # Stop spinner before printing tool
                    self._stop_spinner()
                    self._format_tool(name, args, result if not event.is_error else f"Error: {result}")
                    self._after_tool = True
                    # Restart spinner for next operation
                    self._start_spinner()

        elif isinstance(event, StreamDone):
            # Flush any remaining buffers
            self._flush_thinking()
            self._flush_text()

        elif isinstance(event, StreamError):
            self._stop_spinner()
            self._empty()
            self._print(f"{self._red('!')} Error: {event.error}")
            self._empty()

    async def get_input(self, prompt: str = "") -> str:
        """Get user input via stdin."""
        # OpenCode style: just "> " prompt, ignore passed prompt
        self._empty()

        def _get_input() -> str:
            try:
                return input("> ")
            except EOFError as e:
                raise KeyboardInterrupt("stdin closed (EOF)") from e

        result = await trio.to_thread.run_sync(_get_input, abandon_on_cancel=True)
        # Start spinner after user submits their message
        self._start_spinner()
        return result

    async def confirm_tool(self, tool_call: ToolCall) -> bool:
        """Confirm tool execution via stdin."""
        args_str = ", ".join(f"{k}={v!r}" for k, v in tool_call.args.items())
        if len(args_str) > 80:
            args_str = args_str[:77] + "..."

        self._empty()
        self._print(f"  {self._yellow('?')} Confirm: {tool_call.name}({args_str})")
        self._print("    [y] approve  [n] reject  [Enter=approve]")

        response = await trio.to_thread.run_sync(input, "    > ")
        return response.lower() in ("", "y", "yes")

    def show_loader(self, text: str) -> None:
        """No-op for minimal frontend."""
        pass

    def hide_loader(self) -> None:
        """No-op for minimal frontend."""
        pass
