"""run_codex - Codex CLI as a run_agent-compatible backend.

Same signature as run_agent:
    async def run_codex(state: AgentState, config: RunConfig) -> list[AgentState]

Uses `codex exec --json` for non-interactive JSON output.
Each turn spawns a new codex process (no bidirectional mode yet).
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import trio

from ..dtypes import (
    LLMCallEnd,
    LLMCallStart,
    Message,
    SessionStatus,
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
    from ..agents import AgentState
    from ..dtypes import RunConfig
    from ..store import SessionStore

logger = logging.getLogger(__name__)


async def run_codex(
    state: AgentState,
    config: RunConfig,
    *,
    model: str = "o3-mini",
    cwd: Path | None = None,
    autonomous: bool = False,
) -> list[AgentState]:
    """Run Codex CLI as the agent backend.

    Uses `codex exec --json` for each turn (non-interactive).
    Messages are accumulated into the rollouts session for cross-driver resume.

    Args:
        state: Initial agent state with trajectory
        config: Run configuration with callbacks (on_chunk, handle_no_tool)
        model: Codex model (o3-mini, o3, etc.)
        cwd: Working directory (defaults to current)
        autonomous: If True, run without user input - agent runs to completion.
            Used for evals. Runs a single turn with full-access sandbox.

    Returns:
        List of agent states from the run
    """
    cwd = cwd or Path.cwd()
    states: list[AgentState] = []
    current_state = state

    # Session store for dual-write (rollouts native format)
    # Note: Session creation is handled by the runner (orchestration layer)
    session_store: SessionStore | None = config.session_store

    # Find codex binary
    codex_bin = shutil.which("codex")
    if codex_bin is None:
        await config.on_chunk(
            StreamError(error="Codex CLI not found. Install from: https://github.com/openai/codex")
        )
        return [replace(current_state, stop=StopReason.PROVIDER_ERROR)]

    logger.info(f"Starting Codex: {codex_bin} exec --json --model {model}")

    # Main loop: get user input, run codex, repeat
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

            user_content = (
                last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
            )

            # Build prompt with conversation history for context
            prompt = _format_prompt_with_history(messages)

            # Run codex exec
            assistant_text = await _run_codex_turn(
                codex_bin=codex_bin,
                prompt=prompt,
                model=model,
                cwd=cwd,
                config=config,
                autonomous=autonomous,
            )

            if assistant_text is None:
                # Error or cancellation
                current_state = replace(current_state, stop=StopReason.PROVIDER_ERROR)
                break

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


async def _run_codex_turn(
    codex_bin: str,
    prompt: str,
    model: str,
    cwd: Path,
    config: RunConfig,
    autonomous: bool = False,
) -> str | None:
    """Run a single codex turn and return the response text."""
    import subprocess

    # Use danger-full-access sandbox in autonomous mode (for evals)
    sandbox_mode = "danger-full-access" if autonomous else "workspace-write"

    cmd = [
        codex_bin,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        sandbox_mode,
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)

    logger.info(f"Running codex: {' '.join(cmd[:6])} <prompt>")

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
    thread_id = None
    usage = {}

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

                # Parse codex events
                event_type = event.get("type", "")

                logger.debug(f"Codex event: {event_type} - {event}")

                if event_type == "thread.started":
                    thread_id = event.get("thread_id")

                elif event_type == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        response_text += text
                        # Emit text events
                        await config.on_chunk(TextStart(content_index=0))
                        await config.on_chunk(TextDelta(content_index=0, delta=text))
                        await config.on_chunk(TextEnd(content_index=0, content=text))

                elif event_type == "turn.completed":
                    usage = event.get("usage", {})

        # Wait for process to finish
        await proc.wait()

        logger.info(f"Codex finished, response_text={response_text!r}")

        # Emit completion events
        await config.on_chunk(
            LLMCallEnd(
                duration_ms=0,
                provider="codex",
                model=model,
                tokens_in=usage.get("input_tokens"),
                tokens_out=usage.get("output_tokens"),
                status="success",
            )
        )
        await config.on_chunk(StreamDone(finish_reason="stop"))

        return response_text if response_text else None

    except trio.Cancelled:
        proc.terminate()
        raise
    finally:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
