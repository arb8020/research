"""HeadlessJsonFrontend - Bidirectional NDJSON for scripting and testing.

Unlike JsonFrontend which is output-only, this frontend supports:
- Reading NDJSON input from stdin (user messages, slash commands)
- Emitting NDJSON output to stdout (responses, command results)
- Processing slash commands like the TUI does

Designed for:
- Testing slash commands programmatically
- Scripted agent interactions
- Building external UIs that communicate via JSON
- CI/CD pipelines with interactive flows

Input format (one JSON per line):
    {"type": "user", "text": "hello"}
    {"type": "user", "text": "/env list"}
    {"type": "user", "text": "/model anthropic/claude-sonnet-4-20250514"}

Output format (one JSON per line):
    {"type": "system", "subtype": "init", "session_id": "...", "tools": [...]}
    {"type": "slash_command", "command": "/env list", "result": "Available environments:..."}
    {"type": "assistant", "message": {"content": [...]}}
    {"type": "result", "subtype": "success", "session_id": "...", "num_turns": 1}
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

if TYPE_CHECKING:
    from rollouts.dtypes import Endpoint, Environment, StreamEvent, ToolCall, Trajectory


@dataclass
class HeadlessJsonFrontend:
    """Bidirectional NDJSON streaming frontend.

    Reads JSON input from stdin, emits JSON output to stdout.
    Supports slash commands for testing /env, /model, /thinking, etc.
    """

    # Configuration
    input_file: Any = field(default_factory=lambda: sys.stdin)
    output_file: Any = field(default_factory=lambda: sys.stdout)
    include_thinking: bool = False
    include_timing: bool = True

    # State (initialized in __post_init__ or start)
    _current_text: list[str] = field(default_factory=list)
    _current_thinking: list[str] = field(default_factory=list)
    _current_tool_calls: list[dict] = field(default_factory=list)
    _tool_results: dict[str, dict] = field(default_factory=dict)
    _start_time: float = 0
    _num_turns: int = 0
    _session_id: str | None = None
    _tools: list[str] = field(default_factory=list)

    # For slash command handling (mirrors InteractiveAgentRunner interface)
    environment: Environment | None = None
    endpoint: Endpoint | None = None
    session_store: Any = None
    session_id: str | None = None
    trajectory: Trajectory | None = None

    # Input buffer for async reading
    _input_lines: list[str] = field(default_factory=list)
    _input_exhausted: bool = False

    def _emit(self, obj: dict) -> None:
        """Emit a single NDJSON line."""
        print(json.dumps(obj, ensure_ascii=False), file=self.output_file, flush=True)

    def _flush_assistant_turn(self) -> None:
        """Emit current assistant turn if there's content."""
        content = []

        # Add thinking if present and enabled
        if self._current_thinking and self.include_thinking:
            thinking_text = "".join(self._current_thinking).strip()
            if thinking_text:
                content.append({"type": "thinking", "thinking": thinking_text})

        # Add text content
        text = "".join(self._current_text).strip()
        if text:
            content.append({"type": "text", "text": text})

        # Add tool calls
        content.extend(self._current_tool_calls)

        if content:
            self._emit({"type": "assistant", "message": {"content": content}})
            self._num_turns += 1

        # Reset state
        self._current_text = []
        self._current_thinking = []
        self._current_tool_calls = []

    async def start(self) -> None:
        """Emit init event."""
        self._start_time = time.time()
        self._emit({
            "type": "system",
            "subtype": "init",
            "session_id": self._session_id or self.session_id or "",
            "tools": self._tools,
            "environment": self._get_current_env_name(),
        })

    async def stop(self) -> None:
        """Emit result event."""
        # Flush any remaining content
        self._flush_assistant_turn()

        result = {
            "type": "result",
            "subtype": "success",
            "session_id": self._session_id or self.session_id or "",
            "num_turns": self._num_turns,
        }

        if self.include_timing:
            result["duration_ms"] = int((time.time() - self._start_time) * 1000)

        self._emit(result)

    async def handle_event(self, event: StreamEvent) -> None:
        """Handle streaming event by emitting JSON."""
        from rollouts.dtypes import (
            StreamDone,
            StreamError,
            TextDelta,
            ThinkingDelta,
            ToolCallEnd,
            ToolResultReceived,
        )

        if isinstance(event, TextDelta):
            self._current_text.append(event.delta)

        elif isinstance(event, ThinkingDelta):
            self._current_thinking.append(event.delta)

        elif isinstance(event, ToolCallEnd):
            self._current_tool_calls.append({
                "type": "tool_use",
                "id": event.tool_call.id,
                "name": event.tool_call.name,
                "input": dict(event.tool_call.args),
            })

        elif isinstance(event, ToolResultReceived):
            # Flush assistant turn before tool result
            self._flush_assistant_turn()

            # Emit tool result as user message
            self._emit({
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": event.tool_call_id,
                            "content": event.content,
                            "is_error": event.is_error,
                        }
                    ]
                },
            })

        elif isinstance(event, StreamDone):
            self._flush_assistant_turn()

        elif isinstance(event, StreamError):
            self._flush_assistant_turn()
            self._emit({
                "type": "error",
                "error": str(event.error),
            })

    def _get_current_env_name(self) -> str:
        """Get current environment name (mirrors TUI helper)."""
        if self.environment is None:
            return "none"

        if hasattr(self.environment, "environments"):
            names = []
            for env in self.environment.environments:
                if hasattr(env, "get_name"):
                    names.append(env.get_name())
                else:
                    names.append(type(env).__name__)
            return "+".join(names) if names else "composed"

        if hasattr(self.environment, "get_name"):
            return self.environment.get_name()

        return type(self.environment).__name__

    async def _handle_slash_command(self, command: str) -> tuple[bool, str | None]:
        """Handle slash commands, returning (handled, expanded_text)."""
        from .tui.slash_commands import handle_slash_command

        result = await handle_slash_command(self, command)

        # Emit slash command result
        if result.message:
            self._emit({
                "type": "slash_command",
                "command": command,
                "handled": result.handled,
                "result": result.message,
            })

        if result.handled:
            return True, None
        elif result.expanded_text:
            return False, result.expanded_text
        else:
            return False, None

    async def switch_session(self, new_session_id: str) -> bool:
        """Switch to a different session (called by /env, /slice)."""
        if not self.session_store:
            return False

        session, err = await self.session_store.get(new_session_id)
        if err or not session:
            return False

        self.session_id = new_session_id
        self._session_id = new_session_id

        # Update trajectory
        from rollouts.dtypes import Trajectory

        self.trajectory = Trajectory(messages=session.messages)

        self._emit({
            "type": "system",
            "subtype": "session_switched",
            "session_id": new_session_id,
        })

        return True

    async def get_input(self, prompt: str = "") -> str:
        """Get user input from stdin NDJSON.

        Reads JSON lines from stdin. Each line should be:
            {"type": "user", "text": "..."}

        Slash commands are processed internally.
        """
        while True:
            # Read next line from stdin
            line = await self._read_input_line()

            if line is None:
                # EOF - signal to stop
                raise EOFError("No more input")

            # Parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                self._emit({
                    "type": "error",
                    "error": f"Invalid JSON input: {e}",
                    "line": line,
                })
                continue

            # Extract text
            text = data.get("text", "")
            if not text:
                self._emit({
                    "type": "error",
                    "error": "Missing 'text' field in input",
                    "input": data,
                })
                continue

            # Handle slash commands
            if text.startswith("/"):
                handled, expanded_text = await self._handle_slash_command(text)
                if handled:
                    continue
                if expanded_text:
                    return expanded_text

            return text

    async def _read_input_line(self) -> str | None:
        """Read a single line from stdin asynchronously."""
        # Use trio's async file reading
        try:
            # For testing, check if we have buffered lines
            if self._input_lines:
                return self._input_lines.pop(0)

            if self._input_exhausted:
                return None

            # Read from stdin using trio
            line = await trio.to_thread.run_sync(self.input_file.readline)
            if not line:
                self._input_exhausted = True
                return None
            return line.strip()
        except Exception:
            self._input_exhausted = True
            return None

    async def confirm_tool(self, tool_call: ToolCall) -> bool:
        """Auto-approve all tools in headless mode."""
        return True

    def show_loader(self, text: str) -> None:
        """Emit loader event."""
        self._emit({"type": "system", "subtype": "loader", "text": text})

    def hide_loader(self) -> None:
        """Emit loader hide event."""
        self._emit({"type": "system", "subtype": "loader_hide"})

    def set_status(
        self,
        *,
        model: str | None = None,
        session_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost: float | None = None,
        env_info: dict[str, str] | None = None,
    ) -> None:
        """Track status for result emission."""
        if session_id is not None:
            self._session_id = session_id
            self.session_id = session_id

    def set_tools(self, tools: list[str]) -> None:
        """Set tool names for init event."""
        self._tools = tools


async def test_slash_command(
    command: str,
    env_spec: str = "coding",
    working_dir: Path | None = None,
) -> dict:
    """Test a slash command and return the result as a dict.

    Convenience function for testing slash commands without full agent setup.

    Args:
        command: Slash command to test (e.g., "/env list")
        env_spec: Environment spec (e.g., "coding", "coding+ask_user")
        working_dir: Working directory for environment

    Returns:
        Dict with command result

    Example:
        >>> import trio
        >>> result = trio.run(test_slash_command, "/env list")
        >>> print(result["result"])
    """
    from io import StringIO

    from rollouts.frontends.tui.slash_commands import (
        _create_environment_from_spec,
        handle_slash_command,
    )

    if working_dir is None:
        working_dir = Path.cwd()

    # Create environment
    env, err = _create_environment_from_spec(env_spec, working_dir)
    if err:
        return {"error": err, "handled": False}

    # Create minimal frontend as runner stand-in
    frontend = HeadlessJsonFrontend(
        input_file=StringIO(),
        output_file=StringIO(),
    )
    frontend.environment = env
    frontend.session_id = None
    frontend.session_store = None

    # Handle command
    result = await handle_slash_command(frontend, command)

    return {
        "command": command,
        "handled": result.handled,
        "result": result.message,
        "expanded_text": result.expanded_text,
    }
