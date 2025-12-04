"""
Agent renderer - connects StreamEvents to TUI components.
"""

from __future__ import annotations

from typing import Any, Optional

from rollouts.dtypes import (
    LLMCallStart,
    StreamEvent,
    StreamStart,
    TextStart,
    TextDelta,
    TextEnd,
    ThinkingStart,
    ThinkingDelta,
    ThinkingEnd,
    ToolCallStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallError,
    StreamDone,
    StreamError,
)

from .tui import TUI, Container
from .components.assistant_message import AssistantMessage
from .components.tool_execution import ToolExecution
from .components.user_message import UserMessage
from .components.loader import Loader
from .components.spacer import Spacer


def _cyan(text: str) -> str:
    """Colorize text cyan."""
    return f"\x1b[36m{text}\x1b[0m"


def _dim(text: str) -> str:
    """Colorize text dim/muted."""
    return f"\x1b[2m{text}\x1b[0m"


def _bg_pending(text: str) -> str:
    """Background color for pending tool."""
    return f"\x1b[48;5;238m{text}\x1b[0m"  # Dark gray background


def _bg_success(text: str) -> str:
    """Background color for successful tool."""
    return f"\x1b[48;5;22m{text}\x1b[0m"  # Dark green background


def _bg_error(text: str) -> str:
    """Background color for error tool."""
    return f"\x1b[48;5;52m{text}\x1b[0m"  # Dark red background


class AgentRenderer:
    """Renders agent StreamEvents to TUI."""

    def __init__(self, tui: TUI) -> None:
        """Initialize agent renderer.

        Args:
            tui: TUI instance to render to
        """
        self.tui = tui
        self.chat_container = Container()
        self.tui.add_child(self.chat_container)

        # Current streaming state
        self.current_message: Optional[AssistantMessage] = None
        self.current_thinking_index: Optional[int] = None
        self.current_text_index: Optional[int] = None

        # Tool tracking: tool_call_id -> ToolExecution component
        self.pending_tools: dict[str, ToolExecution] = {}

        # Loader animation
        self.loader: Optional[Loader] = None
        self.status_container = Container()
        self.tui.add_child(self.status_container)

        # Track content blocks by index
        self.content_blocks: dict[int, dict[str, Any]] = {}

    async def handle_event(self, event: StreamEvent) -> None:
        """Route StreamEvent to appropriate handler.

        Args:
            event: StreamEvent to handle
        """
        match event:
            case LLMCallStart():
                self._handle_llm_call_start()

            case StreamStart():
                self._handle_stream_start()

            case TextStart(content_index=idx):
                self._handle_text_start(idx)

            case TextDelta(content_index=idx, delta=delta):
                self._handle_text_delta(idx, delta)

            case TextEnd(content_index=idx, content=content):
                self._handle_text_end(idx, content)

            case ThinkingStart(content_index=idx):
                self._handle_thinking_start(idx)

            case ThinkingDelta(content_index=idx, delta=delta):
                self._handle_thinking_delta(idx, delta)

            case ThinkingEnd(content_index=idx, content=content):
                self._handle_thinking_end(idx, content)

            case ToolCallStart(content_index=idx, tool_call_id=tool_id, tool_name=name):
                self._handle_tool_call_start(idx, tool_id, name)

            case ToolCallDelta(content_index=idx, tool_call_id=tool_id, partial_args=args):
                self._handle_tool_call_delta(idx, tool_id, args)

            case ToolCallEnd(content_index=idx, tool_call=tc):
                self._handle_tool_call_end(idx, tc)

            case ToolCallError(content_index=idx, tool_call_id=tool_id, tool_name=name, error=err):
                self._handle_tool_call_error(idx, tool_id, name, err)

            case StreamDone():
                self._handle_stream_done()

            case StreamError(error=err):
                self._handle_stream_error(err)

        self.tui.request_render()

    def _handle_llm_call_start(self) -> None:
        """Handle LLM call start - show 'Calling LLM...' loader."""
        if self.loader:
            self.loader.stop()
        self.status_container.clear()
        self.loader = Loader(
            "Calling LLM...",
            spinner_color_fn=_cyan,
            text_color_fn=_dim,
        )
        self.status_container.add_child(self.loader)

    def _handle_stream_start(self) -> None:
        """Handle stream start - hide loader since content is about to stream."""
        # Content is about to stream, hide the loader
        if self.loader:
            self.loader.stop()
            self.loader = None
        self.status_container.clear()
        # Note: We could show a different loader here like "Streaming..."
        # but it's cleaner to just hide it since the streaming text is visible
        self.loader = Loader(
            "Streaming... (Ctrl+C to interrupt)",
            spinner_color_fn=_cyan,
            text_color_fn=_dim,
        )
        self.status_container.add_child(self.loader)

    def _handle_text_start(self, content_index: int) -> None:
        """Handle text block start."""
        # Create assistant message if needed
        if self.current_message is None:
            self.current_message = AssistantMessage()
            self.chat_container.add_child(self.current_message)

        self.current_text_index = content_index
        self.content_blocks[content_index] = {"type": "text", "content": ""}

    def _handle_text_delta(self, content_index: int, delta: str) -> None:
        """Handle text delta - append to current message."""
        if self.current_message is None:
            # Start new message if we don't have one
            self.current_message = AssistantMessage()
            self.chat_container.add_child(self.current_message)
            self.current_text_index = content_index

        if content_index == self.current_text_index:
            self.current_message.append_text(delta)

        # Track content
        if content_index in self.content_blocks:
            self.content_blocks[content_index]["content"] += delta

    def _handle_text_end(self, content_index: int, content: str) -> None:
        """Handle text block end."""
        if self.current_message and content_index == self.current_text_index:
            self.current_message.set_text(content)
            self.current_text_index = None

    def _handle_thinking_start(self, content_index: int) -> None:
        """Handle thinking block start."""
        # Create assistant message if needed
        if self.current_message is None:
            self.current_message = AssistantMessage()
            self.chat_container.add_child(self.current_message)

        self.current_thinking_index = content_index
        self.content_blocks[content_index] = {"type": "thinking", "content": ""}

    def _handle_thinking_delta(self, content_index: int, delta: str) -> None:
        """Handle thinking delta - append to current message."""
        if self.current_message is None:
            # Start new message if we don't have one
            self.current_message = AssistantMessage()
            self.chat_container.add_child(self.current_message)
            self.current_thinking_index = content_index

        if content_index == self.current_thinking_index:
            self.current_message.append_thinking(delta)

        # Track content
        if content_index in self.content_blocks:
            self.content_blocks[content_index]["content"] += delta

    def _handle_thinking_end(self, content_index: int, content: str) -> None:
        """Handle thinking block end."""
        if self.current_message and content_index == self.current_thinking_index:
            self.current_message.set_thinking(content)
            self.current_thinking_index = None

    def _handle_tool_call_start(self, content_index: int, tool_call_id: str, tool_name: str) -> None:
        """Handle tool call start - create tool component."""
        # Finalize current message if we have one
        if self.current_message:
            # Message is complete, clear reference
            self.current_message = None
            self.current_text_index = None
            self.current_thinking_index = None

        # Create tool execution component
        if tool_call_id not in self.pending_tools:
            self.chat_container.add_child(Spacer(1))
            tool_component = ToolExecution(
                tool_name,
                args={},
                bg_fn_pending=_bg_pending,
                bg_fn_success=_bg_success,
                bg_fn_error=_bg_error,
            )
            self.chat_container.add_child(tool_component)
            self.pending_tools[tool_call_id] = tool_component

        self.content_blocks[content_index] = {
            "type": "toolCall",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "args": {},
        }

    def _handle_tool_call_delta(self, content_index: int, tool_call_id: str, partial_args: dict) -> None:
        """Handle tool call delta - update tool args."""
        if tool_call_id in self.pending_tools:
            self.pending_tools[tool_call_id].update_args(partial_args)

        # Track content
        if content_index in self.content_blocks:
            self.content_blocks[content_index]["args"] = partial_args

    def _handle_tool_call_end(self, content_index: int, tool_call) -> None:
        """Handle tool call end - tool is complete (but not executed yet)."""
        tool_id = tool_call.id
        if tool_id in self.pending_tools:
            # Update with final args
            # ToolCall.args is a Mapping[str, Any], convert to dict
            if hasattr(tool_call.args, "items"):
                args_dict = dict(tool_call.args)
            else:
                args_dict = tool_call.args
            self.pending_tools[tool_id].update_args(args_dict)

    def _handle_tool_call_error(self, content_index: int, tool_call_id: str, tool_name: str, error: str) -> None:
        """Handle tool call error."""
        if tool_call_id in self.pending_tools:
            self.pending_tools[tool_call_id].update_result(
                {"content": [{"type": "text", "text": error}]},
                is_error=True,
            )
            # Remove from pending (error is final)
            del self.pending_tools[tool_call_id]

    def _handle_stream_done(self) -> None:
        """Handle stream done - hide loader."""
        if self.loader:
            self.loader.stop()
            self.loader = None
            self.status_container.clear()

        # Finalize current message
        self.current_message = None
        self.current_text_index = None
        self.current_thinking_index = None

    def _handle_stream_error(self, error: str) -> None:
        """Handle stream error - show error message."""
        if self.loader:
            self.loader.stop()
            self.loader = None
            self.status_container.clear()

        # Show error in chat
        from .components.text import Text

        error_text = Text(f"Error: {error}", padding_x=1, padding_y=0)
        self.chat_container.add_child(error_text)

    def add_user_message(self, text: str, is_first: bool = False) -> None:
        """Add a user message to the chat.

        Args:
            text: User message text
            is_first: Whether this is the first user message
        """
        user_component = UserMessage(text, is_first=is_first)
        self.chat_container.add_child(user_component)
        self.tui.request_render()

    def set_tool_result(self, tool_call_id: str, result: dict, is_error: bool = False) -> None:
        """Set tool execution result.

        Args:
            tool_call_id: Tool call ID
            result: Result data (may contain 'content' list or be a string)
            is_error: Whether this is an error result
        """
        if tool_call_id in self.pending_tools:
            # Normalize result format
            if isinstance(result, str):
                result_dict = {"content": [{"type": "text", "text": result}]}
            else:
                result_dict = result

            self.pending_tools[tool_call_id].update_result(result_dict, is_error=is_error)
            # Remove from pending (result is final)
            del self.pending_tools[tool_call_id]
            self.tui.request_render()

