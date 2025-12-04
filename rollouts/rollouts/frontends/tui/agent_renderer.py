"""
Agent renderer - connects StreamEvents to TUI components.
"""

from __future__ import annotations

import json
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
    ToolResultReceived,
    StreamDone,
    StreamError,
)

from .tui import TUI, Container
from .theme import Theme
from .components.assistant_message import AssistantMessage
from .components.tool_execution import ToolExecution
from .components.user_message import UserMessage
from .components.spacer import Spacer


class AgentRenderer:
    """Renders agent StreamEvents to TUI."""

    def __init__(self, tui: TUI, debug_layout: bool = False) -> None:
        """Initialize agent renderer.

        Args:
            tui: TUI instance to render to
            debug_layout: Show component boundaries and spacing
        """
        self.tui = tui
        self.theme = tui.theme
        self.debug_layout = debug_layout
        self.chat_container = Container()
        self.tui.add_child(self.chat_container)

        # Current streaming state
        self.current_message: Optional[AssistantMessage] = None
        self.current_thinking_index: Optional[int] = None
        self.current_text_index: Optional[int] = None

        # Tool tracking: tool_call_id -> ToolExecution component
        self.pending_tools: dict[str, ToolExecution] = {}

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

            case ToolResultReceived(tool_call_id=tool_id, content=content, is_error=is_err, error=err):
                self._handle_tool_result(tool_id, content, is_err, err)

            case StreamDone():
                self._handle_stream_done()

            case StreamError(error=err):
                self._handle_stream_error(err)

        self.tui.request_render()

    def _handle_llm_call_start(self) -> None:
        """Handle LLM call start - show 'Calling LLM...' loader."""
        self.tui.show_loader(
            "Calling LLM...",
            spinner_color_fn=self.theme.fg(self.theme.accent),
            text_color_fn=self.theme.fg(self.theme.muted),
        )

    def _handle_stream_start(self) -> None:
        """Handle stream start - switch to streaming loader."""
        self.tui.show_loader(
            "Streaming... (Ctrl+C to interrupt)",
            spinner_color_fn=self.theme.fg(self.theme.accent),
            text_color_fn=self.theme.fg(self.theme.muted),
        )

    def _handle_text_start(self, content_index: int) -> None:
        """Handle text block start."""
        # Create assistant message if needed
        if self.current_message is None:
            self.chat_container.add_child(Spacer(1, debug_label="before-assistant", debug_layout=self.debug_layout))
            self.current_message = AssistantMessage(theme=self.theme, debug_layout=self.debug_layout)
            self.chat_container.add_child(self.current_message)

        self.current_text_index = content_index
        self.content_blocks[content_index] = {"type": "text", "content": ""}

    def _handle_text_delta(self, content_index: int, delta: str) -> None:
        """Handle text delta - append to current message."""
        if self.current_message is None:
            # Start new message if we don't have one
            self.chat_container.add_child(Spacer(1, debug_label="before-assistant", debug_layout=self.debug_layout))
            self.current_message = AssistantMessage(theme=self.theme, debug_layout=self.debug_layout)
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
            self.chat_container.add_child(Spacer(1, debug_label="before-assistant", debug_layout=self.debug_layout))
            self.current_message = AssistantMessage(theme=self.theme, debug_layout=self.debug_layout)
            self.chat_container.add_child(self.current_message)

        self.current_thinking_index = content_index
        self.content_blocks[content_index] = {"type": "thinking", "content": ""}

    def _handle_thinking_delta(self, content_index: int, delta: str) -> None:
        """Handle thinking delta - append to current message."""
        if self.current_message is None:
            # Start new message if we don't have one
            self.chat_container.add_child(Spacer(1, debug_label="before-assistant", debug_layout=self.debug_layout))
            self.current_message = AssistantMessage(theme=self.theme, debug_layout=self.debug_layout)
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
        # Hide loader - we're now showing tool UI instead
        self.tui.hide_loader()

        # Finalize current message if we have one
        if self.current_message:
            # Message is complete, clear reference
            self.current_message = None
            self.current_text_index = None
            self.current_thinking_index = None

        # Create tool execution component
        if tool_call_id not in self.pending_tools:
            self.chat_container.add_child(Spacer(1, debug_label="before-tool", debug_layout=self.debug_layout))
            tool_component = ToolExecution(
                tool_name,
                args={},
                bg_fn_pending=self.theme.tool_pending_bg_fn,
                bg_fn_success=self.theme.tool_success_bg_fn,
                bg_fn_error=self.theme.tool_error_bg_fn,
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

    def _handle_tool_result(self, tool_call_id: str, content: str, is_error: bool, error: Optional[str]) -> None:
        """Handle tool execution result - update tool component from pending to success/error."""
        if tool_call_id in self.pending_tools:
            result_text = error if is_error and error else content
            self.pending_tools[tool_call_id].update_result(
                {"content": [{"type": "text", "text": result_text}]},
                is_error=is_error,
            )
            # Remove from pending (result is final)
            del self.pending_tools[tool_call_id]

    def _handle_stream_done(self) -> None:
        """Handle stream done - hide loader."""
        self.tui.hide_loader()

        # Finalize current message
        self.current_message = None
        self.current_text_index = None
        self.current_thinking_index = None

    def _handle_stream_error(self, error: str) -> None:
        """Handle stream error - show error message."""
        self.tui.hide_loader()

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
        if not is_first:
            self.chat_container.add_child(Spacer(1, debug_label="before-user", debug_layout=self.debug_layout))
        user_component = UserMessage(text, is_first=is_first, theme=self.theme)
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

    def render_history(self, messages: list, skip_system: bool = True) -> None:
        """Render historical messages from a resumed session.

        Args:
            messages: List of Message objects to render
            skip_system: Whether to skip system messages (default True)
        """
        from rollouts.dtypes import Message, TextContent, ThinkingContent, ToolCallContent

        for msg in messages:
            if not isinstance(msg, Message):
                continue

            # Skip system messages
            if skip_system and msg.role == "system":
                continue

            if msg.role == "user":
                self._render_user_message(msg)
            elif msg.role == "assistant":
                # Convert assistant message to events and replay through handlers
                self._replay_assistant_message_as_events(msg)
            elif msg.role == "tool":
                # Convert tool result to event and replay through handler
                self._replay_tool_result_as_event(msg)

        self.tui.request_render()

    def _render_user_message(self, msg) -> None:
        """Render a user message from history."""
        content = msg.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            text = "\n".join(text_parts)
        else:
            text = str(content) if content else ""

        if text:
            is_first = len(self.chat_container.children) == 0
            if not is_first:
                self.chat_container.add_child(Spacer(1, debug_label="before-user", debug_layout=self.debug_layout))
            user_component = UserMessage(text, is_first=is_first, theme=self.theme)
            self.chat_container.add_child(user_component)

    def _replay_assistant_message_as_events(self, msg) -> None:
        """Convert assistant message from history into events and replay through handlers.

        This ensures history rendering uses the exact same code path as live streaming.
        """
        from rollouts.dtypes import TextContent, ThinkingContent, ToolCallContent

        content = msg.content
        if content is None:
            return

        # Handle string content - convert to TextContent
        if isinstance(content, str):
            if content:
                content = [TextContent(text=content)]
            else:
                return

        # Handle list of content blocks
        if not isinstance(content, list):
            return

        content_index = 0
        for block in content:
            # Handle dataclass types
            if isinstance(block, TextContent):
                # Simulate text streaming: start -> delta -> end
                self._handle_text_start(content_index)
                self._handle_text_delta(content_index, block.text)
                self._handle_text_end(content_index, block.text)
                content_index += 1

            elif isinstance(block, ThinkingContent):
                # Simulate thinking streaming: start -> delta -> end
                self._handle_thinking_start(content_index)
                self._handle_thinking_delta(content_index, block.thinking)
                self._handle_thinking_end(content_index, block.thinking)
                content_index += 1

            elif isinstance(block, ToolCallContent):
                # Simulate tool call: start -> end
                # Create a minimal ToolCall object for the handler
                from rollouts.dtypes import ToolCall
                tool_call = ToolCall(
                    id=block.id,
                    name=block.name,
                    args=dict(block.arguments)
                )
                self._handle_tool_call_start(content_index, block.id, block.name)
                self._handle_tool_call_end(content_index, tool_call)
                content_index += 1

            # Handle legacy dict format
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        self._handle_text_start(content_index)
                        self._handle_text_delta(content_index, text)
                        self._handle_text_end(content_index, text)
                        content_index += 1

                elif block_type == "thinking":
                    thinking = block.get("thinking", "")
                    if thinking:
                        self._handle_thinking_start(content_index)
                        self._handle_thinking_delta(content_index, thinking)
                        self._handle_thinking_end(content_index, thinking)
                        content_index += 1

                elif block_type in ("tool_use", "toolCall"):
                    tool_name = block.get("name", "unknown")
                    tool_id = block.get("id", "")
                    tool_args = block.get("input", block.get("arguments", {}))

                    from rollouts.dtypes import ToolCall
                    tool_call = ToolCall(
                        id=tool_id,
                        name=tool_name,
                        args=tool_args
                    )
                    self._handle_tool_call_start(content_index, tool_id, tool_name)
                    self._handle_tool_call_end(content_index, tool_call)
                    content_index += 1

        # Simulate stream done to finalize the message
        self._handle_stream_done()

    def _replay_tool_result_as_event(self, msg) -> None:
        """Convert tool result from history into event and replay through handler."""
        tool_call_id = msg.tool_call_id
        if not tool_call_id:
            return

        content = msg.content
        if isinstance(content, str):
            result_text = content
        elif isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            result_text = "\n".join(text_parts)
        else:
            result_text = str(content) if content else ""

        # Replay through the same handler used for live tool results
        self._handle_tool_result(tool_call_id, result_text, is_error=False, error=None)

    def debug_dump_chat(self) -> None:
        """Dump chat container state as JSONL for debugging."""
        print("\n=== CHAT CONTAINER DEBUG DUMP ===")
        for i, child in enumerate(self.chat_container.children):
            state = {
                "index": i,
                "type": type(child).__name__,
            }
            if hasattr(child, "debug_state"):
                state.update(child.debug_state())
            print(json.dumps(state))
        print(f"=== TOTAL: {len(self.chat_container.children)} components ===\n")

