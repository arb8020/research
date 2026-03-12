"""run_cursor - Cursor Agent CLI as a run_agent-compatible backend.

Same signature as run_agent:
    async def run_cursor(state: AgentState, config: RunConfig) -> list[AgentState]

Uses `cursor-agent --print --output-format stream-json` for JSON output.
Each turn spawns a new cursor-agent process (no bidirectional mode).

The JSON format is nearly identical to Claude Code:
- {"type":"system","subtype":"init","session_id":"..."}
- {"type":"user","message":...}
- {"type":"assistant","message":...}
- {"type":"result","subtype":"success",...}
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

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
    Trajectory,
)

if TYPE_CHECKING:
    from ..agents import AgentState, RunConfig
    from ..store import SessionStore

logger = logging.getLogger(__name__)


async def run_cursor(
    state: AgentState,
    config: RunConfig,
    *,
    model: str | None = None,
    cwd: Path | None = None,
    api_key: str | None = None,
    autonomous: bool = False,
) -> list[AgentState]:
    """Run Cursor Agent CLI as the agent backend.

    Uses `cursor-agent --print --output-format stream-json` for each turn.
    Messages are accumulated into the rollouts session for cross-driver resume.

    Args:
        state: Initial agent state with trajectory
        config: Run configuration with callbacks (on_chunk, handle_no_tool)
        model: Cursor model (optional, uses default if not specified)
        cwd: Working directory (defaults to current)
        api_key: Cursor API key (can also use CURSOR_API_KEY env var)
        autonomous: If True, run without user input - agent runs to completion.
            Used for evals. Runs a single turn.

    Returns:
        List of agent states from the run
    """
    cwd = cwd or Path.cwd()
    states: list[AgentState] = []
    current_state = state

    # Session store for dual-write (rollouts native format)
    session_store: SessionStore | None = config.session_store

    # Find cursor-agent binary
    cursor_bin = shutil.which("cursor-agent")
    if cursor_bin is None:
        # Try common locations
        for path in [
            Path.home() / ".local/bin/cursor-agent",
            Path("/usr/local/bin/cursor-agent"),
        ]:
            if path.exists():
                cursor_bin = str(path)
                break

    if cursor_bin is None:
        await config.on_chunk(
            StreamError(error="Cursor Agent CLI not found. Install from: https://cursor.com/cli")
        )
        return [replace(current_state, stop=StopReason.PROVIDER_ERROR)]

    # Get API key from arg, env, or error
    api_key = api_key or os.environ.get("CURSOR_API_KEY")
    if not api_key:
        await config.on_chunk(
            StreamError(
                error="Cursor API key not provided. Set CURSOR_API_KEY env var or pass --cursor-api-key"
            )
        )
        return [replace(current_state, stop=StopReason.PROVIDER_ERROR)]

    logger.info(f"Starting Cursor Agent: {cursor_bin} --print --output-format stream-json")

    # Main loop: get user input, run cursor, repeat
    try:
        while True:
            # Get the last user message to send
            messages = list(current_state.actor.trajectory.messages)
            if not messages or messages[-1].role != "user":
                if autonomous:
                    # Autonomous mode requires a user message in trajectory
                    await config.on_chunk(
                        StreamError(error="Autonomous mode requires a user message in trajectory")
                    )
                    current_state = replace(current_state, stop=StopReason.ERROR)
                    break
                # Need user input
                new_state = await config.handle_no_tool(current_state, config)
                if new_state.stop:
                    current_state = new_state
                    break
                messages = list(new_state.actor.trajectory.messages)
                current_state = new_state

            # Get user message content
            last_msg = messages[-1]
            if last_msg.role != "user":
                break

            # Build prompt with conversation history for context
            prompt = _format_prompt_with_history(messages)

            # Run cursor-agent
            assistant_text, cursor_session_id = await _run_cursor_turn(
                cursor_bin=cursor_bin,
                prompt=prompt,
                model=model,
                cwd=cwd,
                api_key=api_key,
                config=config,
            )

            if assistant_text is None:
                # Error or cancellation
                current_state = replace(current_state, stop=StopReason.PROVIDER_ERROR)
                break

            # Store cursor session_id for potential resume
            if cursor_session_id:
                current_state = replace(current_state, driver_session_id=cursor_session_id)

            # Create assistant message and update trajectory
            assistant_msg = Message(role="assistant", content=assistant_text)
            new_messages = messages + [assistant_msg]
            new_trajectory = Trajectory(messages=new_messages)
            current_state = replace(
                current_state,
                actor=replace(current_state.actor, trajectory=new_trajectory),
            )

            # Dual-write: persist assistant message
            if session_store and current_state.session_id:
                await session_store.append_message(current_state.session_id, assistant_msg)

            states.append(current_state)

            # Autonomous mode: single turn, exit after response
            if autonomous:
                logger.info("Autonomous mode: completed single turn")
                break

            # Get next user input
            new_state = await config.handle_no_tool(current_state, config)
            if new_state.stop:
                current_state = new_state
                break

            # Check if user added a message
            if len(new_state.actor.trajectory.messages) > len(messages) + 1:
                last_user_msg = new_state.actor.trajectory.messages[-1]
                if last_user_msg.role == "user":
                    current_state = new_state
                    # Dual-write: persist user message
                    if session_store and current_state.session_id:
                        await session_store.append_message(current_state.session_id, last_user_msg)

    except trio.Cancelled:
        current_state = replace(current_state, stop=StopReason.ABORTED)
        if session_store and current_state.session_id:
            await session_store.update(current_state.session_id, status=SessionStatus.ABORTED)
        states.append(current_state)
        return states

    # Update session status
    if session_store and current_state.session_id:
        if current_state.stop == StopReason.PROVIDER_ERROR:
            status = SessionStatus.ABORTED
        else:
            status = SessionStatus.COMPLETED
        await session_store.update(current_state.session_id, status=status)

    states.append(current_state)
    return states


def _format_prompt_with_history(messages: list[Message]) -> str:
    """Format messages into a prompt with conversation history."""
    if len(messages) <= 1:
        # Just the user message
        msg = messages[-1]
        return msg.content if isinstance(msg.content, str) else str(msg.content)

    # Include history
    lines = ["[Previous conversation]"]
    for msg in messages[:-1]:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if content.strip() == "[interrupted]":
            continue
        if len(content) > 500:
            content = content[:500] + "..."
        role_label = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role_label}: {content}")

    lines.append("")
    lines.append("[Current message]")
    last_content = messages[-1].content
    lines.append(last_content if isinstance(last_content, str) else str(last_content))

    return "\n".join(lines)


async def _run_cursor_turn(
    cursor_bin: str,
    prompt: str,
    model: str | None,
    cwd: Path,
    api_key: str,
    config: RunConfig,
) -> tuple[str | None, str | None]:
    """Run a single cursor-agent turn and return (response_text, session_id)."""
    import subprocess

    cmd = [
        cursor_bin,
        "--print",
        "--output-format",
        "stream-json",
        "--api-key",
        api_key,
        "--workspace",
        str(cwd),
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)

    logger.info(f"Running cursor-agent: {' '.join(cmd[:7])} <prompt>")

    proc = await trio.lowlevel.open_process(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd),
    )

    # Emit stream start
    await config.on_chunk(StreamStart())
    await config.on_chunk(LLMCallStart())

    stdout_buffer = b""
    response_text = ""
    session_id = None
    model_used = None

    try:
        # Read and parse JSONL events
        while True:
            try:
                chunk = await proc.stdout.receive_some(4096)
            except trio.ClosedResourceError:
                break
            if not chunk:
                break
            stdout_buffer += chunk

            # Process complete lines
            while b"\n" in stdout_buffer:
                line, stdout_buffer = stdout_buffer.split(b"\n", 1)
                if not line.strip():
                    continue

                try:
                    event = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue

                # Parse cursor events (similar to Claude Code format)
                event_type = event.get("type", "")

                if event_type == "system" and event.get("subtype") == "init":
                    session_id = event.get("session_id")
                    model_used = event.get("model")
                    logger.debug(f"Cursor session: {session_id}, model: {model_used}")

                elif event_type == "assistant":
                    # Extract text from assistant message
                    message = event.get("message", {})
                    content = message.get("content", [])
                    for block in content:
                        if block.get("type") == "text":
                            text = block.get("text", "")
                            response_text += text
                            # Emit text events
                            await config.on_chunk(TextStart(content_index=0))
                            await config.on_chunk(TextDelta(content_index=0, delta=text))
                            await config.on_chunk(TextEnd(content_index=0, content=text))

                elif event_type == "result":
                    # Turn completed
                    duration_ms = event.get("duration_ms", 0)
                    is_error = event.get("is_error", False)

                    await config.on_chunk(
                        LLMCallEnd(
                            duration_ms=duration_ms,
                            provider="cursor",
                            model=model_used or model or "unknown",
                            tokens_in=None,
                            tokens_out=None,
                            status="error" if is_error else "success",
                        )
                    )

        # Wait for process to finish
        await proc.wait()

        await config.on_chunk(StreamDone(finish_reason="stop"))

        return (response_text if response_text else None, session_id)

    except trio.Cancelled:
        proc.terminate()
        raise
    finally:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
