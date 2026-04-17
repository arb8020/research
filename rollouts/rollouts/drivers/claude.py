"""Claude Code driver - parse stream-json output into StreamEvents.

Spawns Claude Code CLI and translates its NDJSON output to StreamEvents
that any Frontend can consume.

Usage:
    # Single prompt (non-interactive)
    driver = ClaudeDriver(cwd="/path/to/repo", model="sonnet")
    async for event in driver.run("Fix the bug in auth.py"):
        print(event)

    # Interactive/multi-turn (bidirectional streaming)
    driver = ClaudeDriver(cwd="/path/to/repo", interactive=True)
    async for event in driver.start():  # Start session, wait for input
        print(event)
    await driver.send_input("Fix the bug")
    async for event in driver.events():  # Get response
        print(event)

Claude Code stream-json format (with --verbose --output-format stream-json):
    {"type": "system", "subtype": "init", "session_id": "...", ...}
    {"type": "assistant", "message": {"content": [...], ...}}
    {"type": "result", "subtype": "success", "num_turns": 3, ...}

With --include-partial-messages, you also get streaming deltas:
    {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "H"}}}

Bidirectional input format (--input-format stream-json):
    {"type": "user", "message": {"role": "user", "content": "..."}}
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trio

from ..dtypes import (
    LLMCallEnd,
    LLMCallStart,
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
)
from .runner import _FlushAssistantMessage

logger = logging.getLogger(__name__)


@dataclass
class ClaudeDriver:
    """Drive Claude Code CLI and emit StreamEvents.

    Spawns `claude` with --output-format stream-json and parses the NDJSON
    output into StreamEvents that frontends can render.

    Attributes:
        cwd: Working directory for Claude Code
        model: Model to use (e.g., "sonnet", "opus", "claude-sonnet-4-20250514")
        include_partial: Enable --include-partial-messages for streaming deltas
        system_prompt: Optional system prompt override
        allowed_tools: Optional list of allowed tools
        timeout_seconds: Timeout for reads (not entire run)
        interactive: Enable bidirectional streaming for multi-turn
        resume_session_id: Optional session ID to resume (--resume flag)
    """

    cwd: Path
    model: str = "sonnet"
    include_partial: bool = True
    system_prompt: str | None = None
    allowed_tools: list[str] | None = None
    timeout_seconds: float = 600.0
    interactive: bool = False
    resume_session_id: str | None = None
    on_raw_line: Callable[[str], Awaitable[None]] | None = field(default=None, repr=False)

    # Runtime state (trio process)
    _proc: trio.Process | None = field(default=None, repr=False)
    _parser: _ClaudeEventParser | None = field(default=None, repr=False)
    _stdout_buffer: bytes = field(default=b"", repr=False)
    _session_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd)
        self._stdout_buffer = b""
        self._session_id = None

    @property
    def session_id(self) -> str | None:
        """Get the Claude Code session ID (available after init message received)."""
        return self._session_id

    def _build_cmd(self, prompt: str | None = None, bidirectional: bool = False) -> list[str]:
        """Build the claude CLI command."""
        claude_bin = shutil.which("claude")
        if claude_bin is None:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )

        cmd = [
            claude_bin,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--model",
            self.model,
        ]

        if bidirectional:
            cmd.append("--input-format")
            cmd.append("stream-json")

        if self.include_partial:
            cmd.append("--include-partial-messages")

        if self.system_prompt:
            cmd.extend(["--system-prompt", self.system_prompt])

        if self.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])

        if self.resume_session_id:
            cmd.extend(["--resume", self.resume_session_id])

        if prompt:
            cmd.append(prompt)

        return cmd

    async def run(self, prompt: str) -> AsyncIterator:
        """Run Claude Code with a prompt, yielding StreamEvents.

        For single-turn usage. For multi-turn, use start() + send_input() + events().

        Args:
            prompt: The task/prompt to send

        Yields:
            StreamEvent instances as Claude processes the task
        """
        try:
            cmd = self._build_cmd(prompt=prompt, bidirectional=False)
        except RuntimeError as e:
            yield StreamError(error=str(e))
            return

        logger.info(f"Starting Claude Code: {cmd[0]} --model {self.model}")

        try:
            self._proc = await trio.lowlevel.open_process(
                cmd,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.cwd),
                env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "rollouts-driver"},
            )

            self._parser = _ClaudeEventParser()

            async for event in self._read_events():
                yield event

            await self._proc.wait()

        except Exception as e:
            logger.exception(f"Claude Code driver failed: {e}")
            yield StreamError(error=str(e))
        finally:
            await self._cleanup()

    async def start(self) -> None:
        """Start an interactive session (bidirectional streaming).

        Spawns Claude Code with --input-format stream-json, ready to receive
        user messages via send_input(). Use events() to read responses.

        Note: Claude doesn't emit output until it receives the first message,
        so this just spawns the process. Call send_input() then events() to
        get the first response.
        """
        cmd = self._build_cmd(prompt=None, bidirectional=True)

        logger.info(f"Starting Claude Code (interactive): {cmd[0]} --model {self.model}")

        self._proc = await trio.lowlevel.open_process(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.cwd),
            env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "rollouts-driver"},
        )

        self._parser = _ClaudeEventParser()

    async def send_input(self, text: str) -> None:
        """Send a user message to the interactive session.

        Args:
            text: The user message to send

        Raises:
            RuntimeError: If session not started or already closed
        """
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Session not started. Call start() first.")

        if self._proc.returncode is not None:
            raise RuntimeError("Session already closed.")

        # Format: {"type": "user", "message": {"role": "user", "content": "..."}}
        msg = json.dumps({"type": "user", "message": {"role": "user", "content": text}})
        await self._proc.stdin.send_all((msg + "\n").encode())
        logger.debug(f"Sent user message: {text[:100]}...")

    async def events(self) -> AsyncIterator:
        """Read events from the interactive session.

        Call after send_input() to get the assistant's response.
        Yields events until the current turn completes.

        Yields:
            StreamEvent instances for the current turn
        """
        if self._proc is None or self._parser is None:
            raise RuntimeError("Session not started. Call start() first.")

        async for event in self._read_events():
            yield event
            # Stop after result (turn complete) — reset parser for next turn
            if isinstance(event, (StreamDone, StreamError)):
                self._parser = _ClaudeEventParser()
                break

    async def _read_events(self) -> AsyncIterator:
        """Read and parse events from stdout."""
        assert self._proc is not None
        assert self._proc.stdout is not None
        assert self._parser is not None

        while True:
            # Read a line from stdout (with timeout)
            line: bytes | None = None
            with trio.move_on_after(self.timeout_seconds) as cancel_scope:
                # Read until newline, handling partial reads
                while b"\n" not in self._stdout_buffer:
                    chunk = await self._proc.stdout.receive_some(4096)
                    if not chunk:
                        # EOF
                        if self._stdout_buffer:
                            line = self._stdout_buffer
                            self._stdout_buffer = b""
                        break
                    self._stdout_buffer += chunk

                if b"\n" in self._stdout_buffer:
                    line, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)

            if cancel_scope.cancelled_caught:
                yield StreamError(error=f"Timeout after {self.timeout_seconds}s")
                break

            if line is None:
                break

            raw_line = line.decode("utf-8", errors="replace")
            if raw_line.endswith("\r"):
                raw_line = raw_line[:-1]
            if self.on_raw_line is not None and raw_line:
                await self.on_raw_line(raw_line)

            try:
                msg = json.loads(raw_line.strip())
            except json.JSONDecodeError:
                continue

            for event in self._parser.parse(msg):
                yield event

            # Capture session_id when init message is parsed
            if self._parser._session_id and not self._session_id:
                self._session_id = self._parser._session_id

    async def abort(self) -> None:
        """Abort the current run."""
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up subprocess."""
        if self._proc is not None:
            try:
                self._proc.terminate()
                with trio.move_on_after(5.0):
                    await self._proc.wait()
            except ProcessLookupError:
                pass
            finally:
                # Ensure process is killed if terminate didn't work
                if self._proc.returncode is None:
                    try:
                        self._proc.kill()
                    except ProcessLookupError:
                        pass
            self._proc = None
        self._parser = None
        self._stdout_buffer = b""


class _ClaudeEventParser:
    """Parse Claude Code NDJSON messages into StreamEvents.

    Tracks state across messages to properly emit start/end events
    and accumulate content for complete blocks.
    """

    def __init__(self) -> None:
        self._started = False
        self._streamed_content = False  # True if we received stream_event deltas
        self._content_index = 0
        self._active_text: dict[int, str] = {}  # content_index -> accumulated text
        self._active_thinking: dict[int, str] = {}
        self._active_tools: dict[str, dict] = {}  # tool_call_id -> {name, args_json}
        self._session_id: str | None = None
        self._model: str = "unknown"

    def parse(self, msg: dict[str, Any]) -> list:
        """Parse a single NDJSON message into zero or more StreamEvents."""
        events = []
        msg_type = msg.get("type")

        match msg_type:
            case "system":
                # Session initialization
                if msg.get("subtype") == "init":
                    self._session_id = msg.get("session_id")
                    self._model = msg.get("model", "unknown")
                    events.append(LLMCallStart())

            case "stream_event":
                # Streaming delta events (with --include-partial-messages)
                self._streamed_content = True
                event = msg.get("event", {})
                events.extend(self._parse_stream_event(event))

            case "assistant":
                # Complete assistant message
                # Skip if we already streamed via stream_event (avoid duplicates).
                #
                # Important trust boundary: Claude's streamed tool_use events can be
                # lossy for arguments, while the persisted session JSONL often contains
                # the complete tool input payloads. For completed runs, prefer the
                # session file as the authoritative source of truth and treat the live
                # stream as provisional UI/debug data.
                if self._streamed_content:
                    pass  # Already handled via streaming
                else:
                    # Non-streaming mode: parse complete message
                    message = msg.get("message", {})
                    content = message.get("content", [])

                    if not self._started:
                        events.append(StreamStart())
                        self._started = True

                    for i, block in enumerate(content):
                        events.extend(self._parse_content_block(i, block))

            case "result":
                # Final result
                is_error = msg.get("is_error", False)
                if is_error:
                    events.append(StreamError(error=msg.get("result", "Unknown error")))
                else:
                    # Emit usage info
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

    def _parse_stream_event(self, event: dict[str, Any]) -> list:
        """Parse a stream_event (delta events from --include-partial-messages)."""
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
                    self._active_tools[tool_id] = {
                        "name": tool_name,
                        "args_json": "",
                        "index": index,
                    }

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
                    # Tool call argument streaming — route by content_index
                    partial = delta.get("partial_json", "")
                    for tool in self._active_tools.values():
                        if tool["index"] == index:
                            tool["args_json"] += partial
                            break

            case "content_block_stop":
                # Emit end events for completed blocks
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
                # Message complete - emit any pending tool calls then flush the turn.
                # _FlushAssistantMessage tells _EventAccumulator to finalize the current
                # assistant message so each LLM response becomes a separate Message object.
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
                events.append(_FlushAssistantMessage())

        return events

    def _parse_content_block(self, index: int, block: dict[str, Any]) -> list:
        """Parse a complete content block (from assistant message without streaming)."""
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
            # Session refactor move 2 parity: pass error through too so the
            # accumulator can promote it to Message.error on the tool-role record.
            error = block.get("error")
            events.append(
                ToolResultReceived(
                    tool_call_id=tool_call_id,
                    content=content if isinstance(content, str) else str(content),
                    is_error=is_error,
                    error=error,
                )
            )

        return events
