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
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

from ..core import SessionStatus
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
    from ..agents import AgentState, RunConfig
    from ..store import SessionStore

logger = logging.getLogger(__name__)


async def run_claude(
    state: AgentState,
    config: RunConfig,
    *,
    model: str = "sonnet",
    cwd: Path | None = None,
    resume_session_id: str | None = None,
    autonomous: bool = False,
    allowed_tools: list[str] | None = None,
) -> list[AgentState]:
    """Run Claude Code CLI as the agent backend.

    Uses bidirectional stream-json mode for multi-turn conversation.
    Accumulates events into messages for dual-write session persistence.

    Args:
        state: Initial agent state with trajectory
        config: Run configuration with callbacks (on_chunk, handle_no_tool)
        model: Claude Code model (sonnet, opus, haiku)
        cwd: Working directory (defaults to current)
        resume_session_id: Claude Code session ID to resume (after interrupt)
        autonomous: If True, run without user input - agent runs to completion.
            Used for evals. Skips handle_no_tool and closes stdin when done.

    Returns:
        List of agent states from the run
    """
    import signal as sig

    cwd = cwd or Path.cwd()
    states: list[AgentState] = []
    current_state = state

    # Debug tracing to file (TUI captures stderr)
    import time

    _trace_file = Path.home() / ".rollouts" / "claude-driver-trace.log"
    _trace_file.parent.mkdir(parents=True, exist_ok=True)

    def _trace(msg: str) -> None:
        with open(_trace_file, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    _trace(
        f"=== run_claude START resume_session_id={resume_session_id} session_id={current_state.session_id} ==="
    )

    # Session store for dual-write (rollouts native format)
    # Note: Session creation is handled by the runner (orchestration layer)
    # before calling run_claude. We just use session_store for message persistence.
    session_store: SessionStore | None = config.session_store

    # Find claude binary
    claude_bin = shutil.which("claude")
    if claude_bin is None:
        await config.on_chunk(
            StreamError(
                error="Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )
        )
        return [replace(current_state, stop=StopReason.ERROR)]

    # Extract initial prompt for autonomous mode
    initial_prompt: str | None = None
    if autonomous:
        messages = list(current_state.actor.trajectory.messages)
        if messages and messages[-1].role == "user":
            last_msg = messages[-1]
            initial_prompt = (
                last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
            )
        if not initial_prompt:
            await config.on_chunk(
                StreamError(error="Autonomous mode requires a user message in trajectory")
            )
            return [replace(current_state, stop=StopReason.ERROR)]

    # Build command
    cmd = [
        claude_bin,
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
        "--model",
        model,
    ]

    if autonomous:
        # Autonomous mode: use -p flag, no stdin interaction
        cmd.extend(["-p", initial_prompt])
        logger.info(f"Starting Claude Code (autonomous): {cmd[0]} --model {model}")
    else:
        # Interactive mode: bidirectional stream-json
        cmd.extend(["--input-format", "stream-json", "--include-partial-messages"])
        # Resume from previous session if provided
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
            logger.info(f"Resuming Claude Code session: {resume_session_id}")
        else:
            logger.info(f"Starting Claude Code: {cmd[0]} --model {model}")

    # Spawn process
    proc = await trio.lowlevel.open_process(
        cmd,
        stdin=subprocess.PIPE if not autonomous else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd),
        env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "rollouts-driver"},
    )

    parser = _ClaudeEventParser()
    accumulator = _MessageAccumulator()
    stdout_buffer = b""
    interrupted = False

    def _format_message_with_history(messages: list[Message]) -> str:
        """Format the last user message, optionally including conversation history.

        If there's only one user message, just return it.
        If there's prior conversation history, format it as context.
        """
        if not messages:
            return ""

        # Extract the last user message
        last_user_content = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                break

        # Check if we have prior messages (history before the last user message)
        # Only include history if there are messages before the last user message
        history_messages = []
        for msg in messages[:-1]:  # All but the last
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Skip [interrupted] markers
            if content.strip() == "[interrupted]":
                continue
            history_messages.append((msg.role, content))

        if not history_messages:
            return last_user_content

        # Format history as context
        history_lines = ["[Previous conversation]"]
        for role, content in history_messages:
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            role_label = "User" if role == "user" else "Assistant"
            history_lines.append(f"{role_label}: {content}")

        history_lines.append("")
        history_lines.append("[Current message]")
        history_lines.append(last_user_content)

        return "\n".join(history_lines)

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
        nonlocal stdout_buffer

        while True:
            # Read line from stdout
            while b"\n" not in stdout_buffer:
                try:
                    chunk = await proc.stdout.receive_some(4096)
                    _trace(f"Received chunk: {len(chunk)} bytes")
                except trio.ClosedResourceError:
                    _trace("ClosedResourceError - process terminated")
                    return False  # Process terminated
                if not chunk:
                    _trace("EOF - no more data")
                    return False  # EOF
                stdout_buffer += chunk

            line, stdout_buffer = stdout_buffer.split(b"\n", 1)
            _trace(f"Got line: {line[:200]}")

            try:
                msg = json.loads(line.decode().strip())
            except json.JSONDecodeError:
                _trace("JSON decode error, skipping line")
                await trio.lowlevel.checkpoint()  # Checkpoint for cancellation (ASYNC913)
                continue

            for event in parser.parse(msg):
                await config.on_chunk(event)
                accumulator.handle_event(event)

                if isinstance(event, StreamDone):
                    # Turn complete - Claude is waiting for input
                    return True
                if isinstance(event, StreamError):
                    return False

        return False

    async def send_message(text: str) -> None:
        """Send user message to Claude via stdin."""
        msg = json.dumps({"type": "user", "message": {"role": "user", "content": text}})
        await proc.stdin.send_all((msg + "\n").encode())
        logger.debug(f"Sent user message: {text[:100]}...")

    early_exit = False  # Flag for early return (ASYNC121: don't return inside nursery)

    try:
        async with trio.open_nursery() as nursery:
            # Start interrupt watcher (Escape key)
            # Note: Ctrl+C cancellation is handled by trio's cancel scope - when
            # cancel_scope.cancel() is called, trio raises Cancelled which we catch below.
            nursery.start_soon(watch_for_interrupt)

            try:
                # Autonomous mode: prompt already passed via -p flag, skip stdin setup
                # Interactive mode: send initial message via stdin
                if not autonomous:
                    # When resuming, Claude Code already has the session context
                    # Don't re-send the last user message
                    if not resume_session_id:
                        # Send initial user message from trajectory
                        messages = list(current_state.actor.trajectory.messages)
                        if messages and messages[-1].role == "user":
                            # Have a pending user message - send it (possibly with history)
                            content = _format_message_with_history(messages)
                            await send_message(content)
                        else:
                            # No user message yet (or last was assistant) - get one
                            new_state = await config.handle_no_tool(current_state, config)
                            if new_state.stop:
                                current_state = new_state
                                states.append(current_state)
                                early_exit = True
                                nursery.cancel_scope.cancel()
                                # Don't return here - let nursery exit cleanly first
                            elif len(new_state.actor.trajectory.messages) > len(
                                current_state.actor.trajectory.messages
                            ):
                                # Check if a new user message was added
                                last_msg = new_state.actor.trajectory.messages[-1]
                                if last_msg.role == "user":
                                    # Include history if we have prior messages
                                    all_messages = list(new_state.actor.trajectory.messages)
                                    content = _format_message_with_history(all_messages)
                                    await send_message(content)
                                    current_state = new_state

                # Main loop: read events, handle no-tool, send input
                if not early_exit:
                    _trace("Entering main loop")
                    while True:
                        _trace("Calling read_events_until_done...")
                        needs_input = await read_events_until_done()
                        _trace(f"read_events_until_done returned: needs_input={needs_input}")

                        if not needs_input:
                            # Process ended or error
                            _trace("needs_input=False, breaking main loop")
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

                            # Dual-write: persist assistant message to rollouts session
                            if session_store and current_state.session_id:
                                await session_store.append_message(
                                    current_state.session_id, assistant_msg
                                )

                        # Autonomous mode: no user input, just keep reading until done
                        # The -p flag means Claude runs to completion without stdin
                        if autonomous:
                            _trace("Autonomous mode: continuing to read events")
                            continue

                        # Turn complete - call handle_no_tool to get next input
                        new_state = await config.handle_no_tool(current_state, config)

                        if new_state.stop:
                            current_state = new_state
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

                                # Dual-write: persist user message to rollouts session
                                if session_store and current_state.session_id:
                                    await session_store.append_message(
                                        current_state.session_id, last_msg
                                    )

                        states.append(current_state)
            finally:
                # Cancel the watcher when main loop exits
                nursery.cancel_scope.cancel()

    except trio.Cancelled:
        current_state = replace(current_state, stop=StopReason.ABORTED)
        # Don't re-raise - return the state so caller can access session_id
        # The caller (runner) handles Cancelled at its own boundary
        _trace(f"Cancelled - returning with session_id={current_state.session_id}")

        # Update session status
        if session_store and current_state.session_id:
            await session_store.update(current_state.session_id, status=SessionStatus.ABORTED)
            logger.info(f"Session {current_state.session_id} aborted (Ctrl+C)")

        states.append(current_state)
        return states
    finally:
        # Cleanup
        try:
            proc.terminate()
            with trio.move_on_after(5.0):
                await proc.wait()
        except ProcessLookupError:
            pass

    # Early exit case - return immediately after nursery cleanup (ASYNC121)
    if early_exit:
        return states

    # Handle interrupted state
    _trace(f"Post-loop: interrupted={interrupted}, session_id={parser._session_id}")
    if interrupted:
        _trace("Setting INTERRUPTED")
        # Add partial assistant message to trajectory (even if empty) so that
        # on retry, run_claude sees messages[-1].role == "assistant" and waits
        # for NEW user input instead of re-sending the old message.
        partial_text = "".join(accumulator.text_parts)
        if partial_text:
            partial_msg = Message(role="assistant", content=partial_text + "\n\n[interrupted]")
        else:
            partial_msg = Message(role="assistant", content="[interrupted]")
        new_messages = list(current_state.actor.trajectory.messages) + [partial_msg]
        new_trajectory = Trajectory(messages=new_messages)
        current_state = replace(
            current_state,
            stop=StopReason.INTERRUPTED,
            actor=replace(current_state.actor, trajectory=new_trajectory),
        )
        logger.info("Interrupted - added partial assistant message to trajectory")

    # Set driver_session_id from parsed session_id so runner can display it
    if parser._session_id:
        current_state = replace(current_state, driver_session_id=parser._session_id)

    # Update session status based on stop reason
    if session_store and current_state.session_id:
        if current_state.stop == StopReason.INTERRUPTED:
            status = SessionStatus.INTERRUPTED
        elif current_state.stop == StopReason.ABORTED:
            status = SessionStatus.ABORTED
        elif current_state.stop == StopReason.ERROR:
            status = SessionStatus.FAILED
        elif current_state.stop == StopReason.END_TURN:
            status = SessionStatus.COMPLETED
        else:
            status = SessionStatus.COMPLETED

        await session_store.update(current_state.session_id, status=status)
        logger.info(f"Updated session {current_state.session_id} status to {status.value}")

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
