"""Codex driver - parse --json output into StreamEvents.

Spawns Codex CLI and translates its NDJSON output to StreamEvents
that any Frontend can consume.

Usage:
    driver = CodexDriver(cwd="/path/to/repo", model="gpt-4.1-mini")
    async for event in driver.run("Fix the bug in auth.py"):
        print(event)

Codex --json format:
    {"type": "thread.started", "thread_id": "..."}
    {"type": "turn.started"}
    {"type": "item.started", "item": {"id": "...", "type": "command_execution", ...}}
    {"type": "item.completed", "item": {"id": "...", "type": "agent_message", "text": "..."}}
    {"type": "turn.completed", "usage": {"input_tokens": ..., "output_tokens": ...}}
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections.abc import AsyncIterator
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
    ToolCall,
    ToolCallEnd,
    ToolCallStart,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolResultReceived,
)

logger = logging.getLogger(__name__)


@dataclass
class CodexDriver:
    """Drive Codex CLI and emit StreamEvents.

    Spawns `codex exec` with --json and parses the NDJSON output into
    StreamEvents that frontends can render.

    Unlike ClaudeDriver, Codex exec mode is single-turn. For multi-turn
    conversations, use `interactive=True` which spawns a new `codex exec resume`
    process for each turn.

    Attributes:
        cwd: Working directory for Codex (must be a git repo)
        model: Model to use (e.g., "gpt-4.1-mini", "o3", "o4-mini")
        sandbox: Sandbox mode ("read-only", "workspace-write", "danger-full-access")
        timeout_seconds: Timeout for the entire run
        resume_session_id: Optional session ID to resume (codex exec resume)
        interactive: Enable multi-turn mode (spawns new process per turn)
    """

    cwd: Path
    model: str = "gpt-4.1-mini"
    sandbox: str = "read-only"
    timeout_seconds: float = 600.0
    resume_session_id: str | None = None
    interactive: bool = False

    # Runtime state
    _proc: trio.Process | None = field(default=None, repr=False)
    _session_id: str | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)
    _stdout_buffer: bytes = field(default=b"", repr=False)

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd)
        self._session_id = self.resume_session_id
        self._started = False
        self._stdout_buffer = b""

    @property
    def session_id(self) -> str | None:
        """Get the Codex session ID."""
        return self._session_id

    async def run(self, prompt: str) -> AsyncIterator:
        """Run Codex with a prompt, yielding StreamEvents.

        Args:
            prompt: The task/prompt to send

        Yields:
            StreamEvent instances as Codex processes the task
        """
        codex_bin = shutil.which("codex")
        if codex_bin is None:
            yield StreamError(
                error="Codex CLI not found. Install from: https://github.com/openai/codex"
            )
            return

        # Use stored session_id for resume (set from previous run or resume_session_id)
        session_to_resume = self._session_id

        if session_to_resume:
            # Resume mode: codex exec resume <session_id> <prompt> --json
            cmd = [
                codex_bin,
                "exec",
                "resume",
                session_to_resume,
                prompt,
                "--json",
                "--skip-git-repo-check",
                "--model",
                self.model,
            ]
            logger.info(f"Resuming Codex session {session_to_resume}: --model {self.model}")
        else:
            # New session mode
            cmd = [
                codex_bin,
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--model",
                self.model,
                "--sandbox",
                self.sandbox,
                prompt,
            ]
            logger.info(f"Starting Codex: {cmd[0]} exec --model {self.model}")

        try:
            self._proc = await trio.lowlevel.open_process(
                cmd,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.cwd),
                env={**os.environ, "CODEX_ENTRYPOINT": "rollouts-driver"},
            )

            assert self._proc.stdout is not None

            parser = _CodexEventParser()
            self._stdout_buffer = b""

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

                try:
                    msg = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    continue

                for event in parser.parse(msg):
                    yield event

                # Capture session_id when parsed
                if parser._session_id and not self._session_id:
                    self._session_id = parser._session_id

            await self._proc.wait()

        except Exception as e:
            logger.exception(f"Codex driver failed: {e}")
            yield StreamError(error=str(e))
        finally:
            await self._cleanup()

    async def start(self) -> None:
        """Start an interactive session.

        For Codex, this just marks the driver as started. The actual process
        is spawned on each send_input() call since Codex exec is single-turn.
        """
        if not self.interactive:
            raise RuntimeError("start() requires interactive=True")
        self._started = True

    async def send_input(self, text: str) -> None:
        """Send user input (stores for next events() call).

        In interactive mode, this stores the input. Call events() to get the response.
        """
        if not self.interactive:
            raise NotImplementedError("send_input() requires interactive=True")
        if not self._started:
            raise RuntimeError("Call start() first")
        # Store the pending input - will be used in events()
        self._pending_input = text

    async def events(self) -> AsyncIterator:
        """Get events for the pending input.

        Spawns `codex exec resume <session_id> <prompt>` and yields events.
        """
        if not hasattr(self, "_pending_input"):
            raise RuntimeError("Call send_input() first")

        prompt = self._pending_input
        delattr(self, "_pending_input")

        # Run the prompt and yield events
        async for event in self.run(prompt):
            yield event

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
        self._stdout_buffer = b""


class _CodexEventParser:
    """Parse Codex NDJSON messages into StreamEvents.

    Codex events are simpler than Claude's - no streaming deltas,
    just complete items.
    """

    def __init__(self) -> None:
        self._thread_id: str | None = None
        self._session_id: str | None = None
        self._model: str = "unknown"
        self._content_index = 0

    def parse(self, msg: dict[str, Any]) -> list:
        """Parse a single NDJSON message into zero or more StreamEvents."""
        events = []
        msg_type = msg.get("type")

        match msg_type:
            case "session_meta":
                # Capture session ID from session metadata (in session files)
                payload = msg.get("payload", {})
                self._session_id = payload.get("id")

            case "thread.started":
                # Capture thread_id as session_id (in live output, thread_id IS the session_id)
                self._thread_id = msg.get("thread_id")
                self._session_id = self._thread_id
                events.append(LLMCallStart())

            case "turn.started":
                events.append(StreamStart())

            case "item.started":
                item = msg.get("item", {})
                events.extend(self._parse_item_started(item))

            case "item.completed":
                item = msg.get("item", {})
                events.extend(self._parse_item_completed(item))

            case "turn.completed":
                usage = msg.get("usage", {})
                events.append(
                    LLMCallEnd(
                        duration_ms=0.0,  # Codex doesn't report duration
                        provider="codex",
                        model=self._model,
                        tokens_in=usage.get("input_tokens"),
                        tokens_out=usage.get("output_tokens"),
                        status="success",
                    )
                )
                events.append(StreamDone(finish_reason="stop"))

            case "turn.failed":
                error = msg.get("error", "Unknown error")
                events.append(StreamError(error=str(error)))

        return events

    def _parse_item_started(self, item: dict[str, Any]) -> list:
        """Parse an item.started event."""
        events = []
        item_id = item.get("id", "")
        item_type = item.get("type")

        if item_type == "command_execution":
            command = item.get("command", "")
            # Emit tool call start
            events.append(
                ToolCallStart(
                    content_index=self._content_index,
                    tool_call_id=item_id,
                    tool_name="shell",
                )
            )
            events.append(
                ToolExecutionStart(
                    tool_call_id=item_id,
                    tool_name="shell",
                )
            )
            self._content_index += 1

        elif item_type == "file_change":
            # File changes also get tool events
            path = item.get("path", "")
            events.append(
                ToolCallStart(
                    content_index=self._content_index,
                    tool_call_id=item_id,
                    tool_name="file_edit",
                )
            )
            events.append(
                ToolExecutionStart(
                    tool_call_id=item_id,
                    tool_name="file_edit",
                )
            )
            self._content_index += 1

        return events

    def _parse_item_completed(self, item: dict[str, Any]) -> list:
        """Parse an item.completed event."""
        events = []
        item_id = item.get("id", "")
        item_type = item.get("type")

        if item_type == "agent_message":
            # Text message from agent
            text = item.get("text", "")
            events.append(TextStart(content_index=self._content_index))
            events.append(TextDelta(content_index=self._content_index, delta=text))
            events.append(TextEnd(content_index=self._content_index, content=text))
            self._content_index += 1

        elif item_type == "command_execution":
            # Command completed
            command = item.get("command", "")
            output = item.get("aggregated_output", "")
            exit_code = item.get("exit_code", 0)
            is_error = exit_code != 0

            events.append(
                ToolExecutionEnd(
                    tool_call_id=item_id,
                    tool_name="shell",
                    duration_ms=0.0,
                    status="error" if is_error else "success",
                    is_error=is_error,
                    result_summary={"exit_code": exit_code, "command": command},
                )
            )
            events.append(
                ToolResultReceived(
                    tool_call_id=item_id,
                    content=output,
                    is_error=is_error,
                )
            )
            events.append(
                ToolCallEnd(
                    content_index=0,
                    tool_call=ToolCall(
                        id=item_id,
                        name="shell",
                        args={"command": command},
                    ),
                )
            )

        elif item_type == "file_change":
            # File change completed
            path = item.get("path", "")
            diff = item.get("diff", "")

            events.append(
                ToolExecutionEnd(
                    tool_call_id=item_id,
                    tool_name="file_edit",
                    duration_ms=0.0,
                    status="success",
                    is_error=False,
                    result_summary={"path": path},
                )
            )
            events.append(
                ToolResultReceived(
                    tool_call_id=item_id,
                    content=diff or f"Modified: {path}",
                    is_error=False,
                )
            )
            events.append(
                ToolCallEnd(
                    content_index=0,
                    tool_call=ToolCall(
                        id=item_id,
                        name="file_edit",
                        args={"path": path},
                    ),
                )
            )

        return events
