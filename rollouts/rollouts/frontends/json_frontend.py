"""JsonFrontend - NDJSON output for scripting and piping.

Emits newline-delimited JSON for each event, compatible with
Claude Code's stream-json format. Designed for:
- Scripting and automation
- Piping to other tools (jq, etc.)
- Building custom UIs that consume JSON
- CI/CD pipelines
"""

from __future__ import annotations

import json
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rollouts.dtypes import StreamEvent, ToolCall


class JsonFrontend:
    """NDJSON streaming frontend.

    Emits one JSON object per line for each event. Format is compatible
    with Claude Code's stream-json output.

    Example output:
        {"type":"system","subtype":"init","session_id":"abc123","tools":["read","write"]}
        {"type":"assistant","message":{"content":[{"type":"text","text":"Hello"}]}}
        {"type":"result","subtype":"success","session_id":"abc123","num_turns":1}

    Example usage:
        frontend = JsonFrontend()
        states = await run_interactive(trajectory, endpoint, frontend=frontend)
    """

    def __init__(
        self,
        file: object | None = None,
        include_thinking: bool = False,
        include_timing: bool = True,
    ) -> None:
        """Initialize JsonFrontend.

        Args:
            file: Output file (default: sys.stdout)
            include_thinking: Include thinking/reasoning tokens in output
            include_timing: Include timing information in result
        """
        self.file = file or sys.stdout
        self.include_thinking = include_thinking
        self.include_timing = include_timing

        # Signal to runner that stdout should be kept clean (no status messages)
        self.json_mode = True

        # State for aggregating turns
        self._current_text: list[str] = []
        self._current_thinking: list[str] = []
        self._current_tool_calls: list[dict] = []
        self._tool_results: dict[str, dict] = {}  # tool_call_id -> result
        self._start_time: float = 0
        self._num_turns: int = 0
        self._session_id: str | None = None
        self._tools: list[str] = []

    def _emit(self, obj: dict) -> None:
        """Emit a single NDJSON line."""
        print(json.dumps(obj, ensure_ascii=False), file=self.file, flush=True)

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
            "session_id": self._session_id or "",
            "tools": self._tools,
        })

    async def stop(self) -> None:
        """Emit result event."""
        # Flush any remaining content
        self._flush_assistant_turn()

        result = {
            "type": "result",
            "subtype": "success",
            "session_id": self._session_id or "",
            "num_turns": self._num_turns,
        }

        if self.include_timing:
            result["duration_ms"] = int((time.time() - self._start_time) * 1000)

        self._emit(result)

    async def handle_event(self, event: StreamEvent) -> None:
        """Handle streaming event by emitting JSON.

        Args:
            event: StreamEvent from agent loop
        """
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

    async def get_input(self, prompt: str = "") -> str:
        """Get user input - not supported in JSON mode.

        For non-interactive use, input should be provided upfront.
        This raises an error if called.
        """
        raise RuntimeError(
            "JsonFrontend does not support interactive input. "
            "Use -p to provide input or resume with --session."
        )

    async def confirm_tool(self, tool_call: ToolCall) -> bool:
        """Auto-approve all tools in JSON mode."""
        return True

    def show_loader(self, text: str) -> None:
        """No-op for JSON mode."""
        pass

    def hide_loader(self) -> None:
        """No-op for JSON mode."""
        pass

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

    def set_tools(self, tools: list[str]) -> None:
        """Set tool names for init event."""
        self._tools = tools
