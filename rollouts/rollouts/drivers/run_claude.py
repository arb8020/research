"""run_claude - Claude Code CLI as a run_agent-compatible backend.

Same signature as run_agent:
    async def run_claude(state: AgentState, config: RunConfig) -> list[AgentState]

Uses bidirectional stream-json mode to communicate with Claude Code CLI.
Events are parsed and forwarded to config.on_chunk, and accumulated into
messages for dual-write to rollouts session format.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal as sig
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

from ..dtypes import (
    LLMCallEnd,
    LLMCallStart,
    Message,
    StopReason,
    StreamDone,
    StreamError,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCall,
    ToolCallEnd,
    ToolCallStart,
    ToolResultReceived,
    Trajectory,
)

if TYPE_CHECKING:
    from ..agents import AgentState
    from ..dtypes import RunConfig

logger = logging.getLogger(__name__)


async def run_claude(
    state: AgentState,
    config: RunConfig,
    *,
    model: str = "sonnet",
    cwd: Path | None = None,
) -> list[AgentState]:
    """Run Claude Code CLI as the agent backend.

    Uses bidirectional stream-json mode for multi-turn conversation.
    Accumulates events into messages for dual-write session persistence.
    Supports interruption via Escape - restarts Claude with --resume to continue.

    Args:
        state: Initial agent state with trajectory
        config: Run configuration with callbacks (on_chunk, handle_no_tool)
        model: Claude Code model (sonnet, opus, haiku)
        cwd: Working directory (defaults to current)

    Returns:
        List of agent states from the run
    """
    cwd = cwd or Path.cwd()
    states: list[AgentState] = []
    current_state = state

    # Find claude binary
    claude_bin = shutil.which("claude")
    if claude_bin is None:
        await config.on_chunk(
            StreamError(
                error="Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )
        )
        return [replace(current_state, stop=StopReason.ERROR)]

    # Session tracking for resume after interrupt
    session_id: str | None = None
    cancelled = False
    interrupted = False
    first_run = True

    # Outer loop: restart Claude after interrupt
    while True:
        # Build command
        cmd = [
            claude_bin,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--include-partial-messages",
            "--model",
            model,
        ]

        # Resume session if we have one (after interrupt)
        if session_id:
            cmd.extend(["--resume", session_id])
            logger.info(f"Resuming Claude session: {session_id}")
        else:
            logger.info(f"Starting Claude Code: {claude_bin} --model {model}")

        # Spawn process
        proc = await trio.lowlevel.open_process(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
            env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "rollouts-driver"},
        )

        parser = _ClaudeEventParser()
        accumulator = _MessageAccumulator()
        stdout_buffer = b""
        interrupted = False

        async def watch_for_cancel() -> None:
            """Watch for cancel scope and terminate process."""
            nonlocal cancelled
            while True:
                await trio.sleep(0.1)
                if config.cancel_scope and config.cancel_scope.cancel_called:
                    logger.info("Cancel requested, terminating Claude process")
                    cancelled = True
                    proc.terminate()
                    return

        async def watch_for_interrupt() -> None:
            """Watch for interrupt flag and send SIGINT to Claude process."""
            nonlocal interrupted
            if not config.interrupt_flag:
                return  # No interrupt flag configured
            while True:
                await trio.sleep(0.05)  # Check frequently
                if config.interrupt_flag[0]:
                    logger.info("Interrupt requested, sending SIGINT to Claude process")
                    config.interrupt_flag[0] = False  # Reset flag
                    interrupted = True
                    try:
                        proc.send_signal(sig.SIGINT)
                    except ProcessLookupError:
                        return  # Process already gone

        async def read_events_until_done() -> bool:
            """Read and emit events until turn completes. Returns True if more input needed."""
            nonlocal stdout_buffer, session_id

            while True:
                # Read line from stdout
                while b"\n" not in stdout_buffer:
                    try:
                        chunk = await proc.stdout.receive_some(4096)
                    except trio.ClosedResourceError:
                        return False  # Process terminated
                    if not chunk:
                        return False  # EOF
                    stdout_buffer += chunk

                line, stdout_buffer = stdout_buffer.split(b"\n", 1)

                try:
                    msg = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    continue

                for event in parser.parse(msg):
                    await config.on_chunk(event)
                    accumulator.handle_event(event)

                    if isinstance(event, StreamDone):
                        # Turn complete - Claude is waiting for input
                        return True
                    if isinstance(event, StreamError):
                        return False

                # Capture session ID for resume
                if parser._session_id and not session_id:
                    session_id = parser._session_id
                    logger.debug(f"Captured Claude session ID: {session_id}")

            return False

        async def send_message(text: str) -> None:
            """Send user message to Claude via stdin."""
            msg = json.dumps({"type": "user", "message": {"role": "user", "content": text}})
            await proc.stdin.send_all((msg + "\n").encode())
            logger.debug(f"Sent user message: {text[:100]}...")

        process_exited_normally = False

        try:
            async with trio.open_nursery() as nursery:
                # Start cancel and interrupt watchers
                nursery.start_soon(watch_for_cancel)
                nursery.start_soon(watch_for_interrupt)

                try:
                    # Only send initial message on first run (not resume)
                    if first_run:
                        messages = list(current_state.actor.trajectory.messages)
                        if messages and messages[-1].role == "user":
                            last_user_msg = messages[-1]
                            content = (
                                last_user_msg.content
                                if isinstance(last_user_msg.content, str)
                                else str(last_user_msg.content)
                            )
                            await send_message(content)
                        else:
                            # No user message yet - get one via handle_no_tool
                            new_state = await config.handle_no_tool(current_state, config)
                            if new_state.stop:
                                current_state = new_state
                                nursery.cancel_scope.cancel()
                                states.append(current_state)
                                return states
                            # Check if a new user message was added
                            if len(new_state.actor.trajectory.messages) > len(
                                current_state.actor.trajectory.messages
                            ):
                                last_msg = new_state.actor.trajectory.messages[-1]
                                if last_msg.role == "user":
                                    content = (
                                        last_msg.content
                                        if isinstance(last_msg.content, str)
                                        else str(last_msg.content)
                                    )
                                    await send_message(content)
                                    current_state = new_state
                        first_run = False

                    # Main loop: read events, handle no-tool, send input
                    while True:
                        if cancelled:
                            current_state = replace(current_state, stop=StopReason.ABORTED)
                            process_exited_normally = True
                            break

                        needs_input = await read_events_until_done()

                        if not needs_input:
                            # Process ended - check if interrupted
                            break

                        # Get accumulated assistant message and update trajectory
                        assistant_msg = accumulator.get_message()
                        if assistant_msg:
                            new_messages = list(current_state.actor.trajectory.messages) + [
                                assistant_msg
                            ]
                            new_trajectory = Trajectory(messages=new_messages)
                            current_state = replace(
                                current_state,
                                actor=replace(current_state.actor, trajectory=new_trajectory),
                            )
                            accumulator.reset()

                        # Turn complete - call handle_no_tool to get next input
                        new_state = await config.handle_no_tool(current_state, config)

                        if new_state.stop:
                            current_state = new_state
                            process_exited_normally = True
                            break

                        # Check if a new user message was added
                        if len(new_state.actor.trajectory.messages) > len(
                            current_state.actor.trajectory.messages
                        ):
                            last_msg = new_state.actor.trajectory.messages[-1]
                            if last_msg.role == "user":
                                content = (
                                    last_msg.content
                                    if isinstance(last_msg.content, str)
                                    else str(last_msg.content)
                                )
                                await send_message(content)
                                current_state = new_state
                                # Reset parser for next turn
                                parser.reset()

                        states.append(current_state)
                finally:
                    # Cancel the watchers when main loop exits
                    nursery.cancel_scope.cancel()

        except trio.Cancelled:
            current_state = replace(current_state, stop=StopReason.ABORTED)
            raise
        finally:
            # Cleanup process
            try:
                proc.terminate()
                with trio.move_on_after(5.0):
                    await proc.wait()
            except ProcessLookupError:
                pass

        # Decide whether to restart or exit
        if cancelled or process_exited_normally:
            # User cancelled or normal exit - done
            break

        if interrupted and session_id:
            # Interrupted - restart with resume
            logger.info(f"Restarting Claude with session {session_id}")
            # Reset parser state but keep session_id
            parser.reset()
            accumulator.reset()
            continue
        # Unknown exit - done
        break

    states.append(current_state)
    return states


class _MessageAccumulator:
    """Accumulate stream events into a Message."""

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self.tool_calls: list[ToolCall] = []

    def handle_event(self, event: Any) -> None:
        """Process an event and accumulate content."""
        if isinstance(event, TextDelta):
            self.text_parts.append(event.delta)
        elif isinstance(event, ThinkingDelta):
            self.thinking_parts.append(event.delta)
        elif isinstance(event, ToolCallEnd):
            self.tool_calls.append(event.tool_call)

    def get_message(self) -> Message | None:
        """Get accumulated message, or None if empty."""
        text = "".join(self.text_parts)
        if not text and not self.tool_calls:
            return None

        # Build content - for now just text
        # TODO: include tool calls in structured format
        return Message(role="assistant", content=text)

    def reset(self) -> None:
        """Reset accumulator for next turn."""
        self.text_parts = []
        self.thinking_parts = []
        self.tool_calls = []


class _ClaudeEventParser:
    """Parse Claude Code NDJSON messages into StreamEvents."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset parser state for new turn."""
        self._started = False
        self._streamed_content = False
        self._active_text: dict[int, str] = {}
        self._active_thinking: dict[int, str] = {}
        self._active_tools: dict[str, dict] = {}
        self._session_id: str | None = None
        self._model: str = "unknown"

    def parse(self, msg: dict) -> list:
        """Parse a single NDJSON message into zero or more StreamEvents."""
        events = []
        msg_type = msg.get("type")

        match msg_type:
            case "system":
                if msg.get("subtype") == "init":
                    self._session_id = msg.get("session_id")
                    self._model = msg.get("model", "unknown")
                    events.append(LLMCallStart())

            case "stream_event":
                self._streamed_content = True
                event = msg.get("event", {})
                events.extend(self._parse_stream_event(event))

            case "assistant":
                if not self._streamed_content:
                    message = msg.get("message", {})
                    content = message.get("content", [])

                    if not self._started:
                        events.append(StreamStart())
                        self._started = True

                    for i, block in enumerate(content):
                        events.extend(self._parse_content_block(i, block))

            case "result":
                is_error = msg.get("is_error", False)
                if is_error:
                    events.append(StreamError(error=msg.get("result", "Unknown error")))
                else:
                    usage = msg.get("usage", {})
                    events.append(
                        LLMCallEnd(
                            duration_ms=float(msg.get("duration_ms", 0)),
                            provider="claude-code",
                            model=self._model,
                            tokens_in=usage.get("input_tokens"),
                            tokens_out=usage.get("output_tokens"),
                            status="success" if not is_error else "error",
                        )
                    )
                    events.append(StreamDone(finish_reason="stop"))

        return events

    def _parse_stream_event(self, event: dict) -> list:
        """Parse streaming delta events."""
        events = []
        event_type = event.get("type")
        index = event.get("index", 0)

        match event_type:
            case "message_start":
                if not self._started:
                    events.append(StreamStart())
                    self._started = True

            case "content_block_start":
                block = event.get("content_block", {})
                block_type = block.get("type")

                if block_type == "text":
                    events.append(TextStart(content_index=index))
                    self._active_text[index] = ""
                elif block_type == "thinking":
                    events.append(ThinkingStart(content_index=index))
                    self._active_thinking[index] = ""
                elif block_type == "tool_use":
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "")
                    events.append(
                        ToolCallStart(
                            content_index=index,
                            tool_call_id=tool_id,
                            tool_name=tool_name,
                        )
                    )
                    self._active_tools[tool_id] = {"name": tool_name, "args_json": ""}

            case "content_block_delta":
                delta = event.get("delta", {})
                delta_type = delta.get("type")

                if delta_type == "text_delta":
                    text = delta.get("text", "")
                    events.append(TextDelta(content_index=index, delta=text))
                    if index in self._active_text:
                        self._active_text[index] += text

                elif delta_type == "thinking_delta":
                    text = delta.get("thinking", "")
                    events.append(ThinkingDelta(content_index=index, delta=text))
                    if index in self._active_thinking:
                        self._active_thinking[index] += text

                elif delta_type == "input_json_delta":
                    partial = delta.get("partial_json", "")
                    for tool_id, tool in self._active_tools.items():
                        tool["args_json"] += partial
                        break

            case "content_block_stop":
                if index in self._active_text:
                    events.append(
                        TextEnd(
                            content_index=index,
                            content=self._active_text.pop(index),
                        )
                    )
                if index in self._active_thinking:
                    events.append(
                        ThinkingEnd(
                            content_index=index,
                            content=self._active_thinking.pop(index),
                        )
                    )

            case "message_stop":
                for tool_id, tool in list(self._active_tools.items()):
                    try:
                        args = json.loads(tool["args_json"]) if tool["args_json"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    events.append(
                        ToolCallEnd(
                            content_index=0,
                            tool_call=ToolCall(
                                id=tool_id,
                                name=tool["name"],
                                args=args,
                            ),
                        )
                    )
                self._active_tools.clear()

        return events

    def _parse_content_block(self, index: int, block: dict) -> list:
        """Parse complete content block (non-streaming mode)."""
        events = []
        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text", "")
            events.append(TextStart(content_index=index))
            events.append(TextDelta(content_index=index, delta=text))
            events.append(TextEnd(content_index=index, content=text))

        elif block_type == "thinking":
            text = block.get("thinking", "")
            events.append(ThinkingStart(content_index=index))
            events.append(ThinkingDelta(content_index=index, delta=text))
            events.append(ThinkingEnd(content_index=index, content=text))

        elif block_type == "tool_use":
            tool_id = block.get("id", "")
            tool_name = block.get("name", "")
            args = block.get("input", {})
            events.append(
                ToolCallStart(
                    content_index=index,
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                )
            )
            events.append(
                ToolCallEnd(
                    content_index=index,
                    tool_call=ToolCall(id=tool_id, name=tool_name, args=args),
                )
            )

        elif block_type == "tool_result":
            tool_call_id = block.get("tool_use_id", "")
            content = block.get("content", "")
            is_error = block.get("is_error", False)
            events.append(
                ToolResultReceived(
                    tool_call_id=tool_call_id,
                    content=content if isinstance(content, str) else str(content),
                    is_error=is_error,
                )
            )

        return events
