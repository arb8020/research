"""Runner for external agent drivers.

Connects a driver to a frontend, streaming events between them.

Usage:
    from rollouts.drivers import ClaudeDriver
    from rollouts.drivers.runner import run_external_agent, run_driver_to_trajectory
    from rollouts.frontends import MinimalFrontend

    driver = ClaudeDriver(cwd="/path/to/repo")
    frontend = MinimalFrontend()

    # Option 1: Stream to frontend
    await run_external_agent(driver, frontend, "Fix the bug in auth.py")

    # Option 2: Capture as trajectory (with logging)
    trajectory = await run_driver_to_trajectory(driver, "Fix the bug", sample_id="001")
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from ..dtypes import (
    ContentBlock,
    Message,
    StreamEvent,
    TextContent,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingContent,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallContent,
    ToolCallEnd,
    ToolCallStart,
    ToolResultReceived,
    Trajectory,
)

if TYPE_CHECKING:
    from ..frontends.protocol import Frontend
    from .protocol import ExternalAgentDriver

# Event logger for eval infrastructure — writes to events.jsonl and per-sample files
# when setup_eval_logging() has been called
_event_logger = logging.getLogger("rollouts.eval.events")


def _event_to_log_dict(event: StreamEvent) -> dict[str, object]:
    if is_dataclass(event):
        return asdict(event)
    raise TypeError(f"Unsupported stream event for eval logging: {type(event)!r}")


class _EventAccumulator:
    """Accumulate StreamEvents into Messages.

    Tracks active content blocks (text, thinking, tool calls) and builds
    complete Messages when blocks finish. Tool results create separate
    tool-role messages.
    """

    def __init__(self) -> None:
        self._messages: list[Message] = []

        # Active content being built for current assistant message
        self._active_text: dict[int, str] = {}  # content_index -> text
        self._active_thinking: dict[int, str] = {}  # content_index -> thinking
        self._active_tools: dict[str, dict] = {}  # tool_call_id -> {name, args, index}

        # Completed content blocks for current assistant message
        self._completed_blocks: list[tuple[int, ContentBlock]] = []  # (index, block)

    def handle(self, event: Any) -> None:
        """Process a StreamEvent, updating internal state."""
        # _FlushAssistantMessage is a Codex-internal sentinel, not a real StreamEvent.
        # It signals a turn boundary so we emit an assistant message immediately
        # rather than waiting for finalize() at the end.
        from .codex import _FlushAssistantMessage

        if isinstance(event, _FlushAssistantMessage):
            self._finalize_assistant_message()
            return

        match event:
            # Text content
            case TextStart(content_index=idx):
                self._active_text[idx] = ""

            case TextDelta(content_index=idx, delta=delta):
                if idx in self._active_text:
                    self._active_text[idx] += delta

            case TextEnd(content_index=idx, content=content):
                self._active_text.pop(idx, None)
                self._completed_blocks.append((idx, TextContent(text=content)))

            # Thinking content
            case ThinkingStart(content_index=idx):
                self._active_thinking[idx] = ""

            case ThinkingDelta(content_index=idx, delta=delta):
                if idx in self._active_thinking:
                    self._active_thinking[idx] += delta

            case ThinkingEnd(content_index=idx, content=content):
                self._active_thinking.pop(idx, None)
                self._completed_blocks.append((idx, ThinkingContent(thinking=content)))

            # Tool calls
            case ToolCallStart(content_index=idx, tool_call_id=tid, tool_name=name):
                self._active_tools[tid] = {"name": name, "index": idx}

            case ToolCallEnd(content_index=idx, tool_call=tc):
                self._active_tools.pop(tc.id, None)
                self._completed_blocks.append((
                    idx,
                    ToolCallContent(
                        id=tc.id,
                        name=tc.name,
                        arguments=tc.args,
                        parse_error=tc.parse_error,
                    ),
                ))

            # Tool results → separate message
            case ToolResultReceived(tool_call_id=tid, content=content, is_error=is_error):
                # Finalize any pending assistant message first
                self._finalize_assistant_message()
                # Add tool result as separate message
                self._messages.append(
                    Message(
                        role="tool",
                        content=content if isinstance(content, str) else str(content),
                        tool_call_id=tid,
                    )
                )

    def _finalize_assistant_message(self) -> None:
        """Finalize current assistant message from completed blocks."""
        if not self._completed_blocks:
            return

        # Sort by content_index to preserve order
        self._completed_blocks.sort(key=lambda x: x[0])
        blocks = [block for _, block in self._completed_blocks]

        self._messages.append(Message(role="assistant", content=blocks))
        self._completed_blocks = []

    def finalize(self) -> list[Message]:
        """Finalize and return all accumulated messages."""
        self._finalize_assistant_message()
        return self._messages


async def run_driver_to_trajectory(
    driver: ExternalAgentDriver,
    prompt: str,
    sample_id: str | None = None,
    on_event: Callable[[StreamEvent], Awaitable[None]] | None = None,
) -> Trajectory:
    """Run an external agent driver and capture the result as a Trajectory.

    Accumulates StreamEvents into Messages and optionally logs each event
    for the eval infrastructure (TUI dashboard, per-sample debug files).

    Args:
        driver: The external agent driver (ClaudeDriver, CodexDriver, etc.)
        prompt: The task/prompt to send to the agent
        sample_id: Optional sample ID for logging. If provided, events are
            logged to rollouts.eval.events logger with sample_id in extra.
        on_event: Optional live event sink. When provided, each StreamEvent is
            forwarded before being accumulated into the trajectory.

    Returns:
        Trajectory containing the accumulated messages

    Example:
        driver = ClaudeDriver(cwd="/path/to/repo")
        trajectory = await run_driver_to_trajectory(driver, "Fix the bug", sample_id="001")
        print(f"Got {len(trajectory.messages)} messages")
    """
    accumulator = _EventAccumulator()

    async for event in driver.run(prompt):
        # Log event if sample_id provided and eval logging is configured
        if sample_id is not None:
            _event_logger.debug(
                event.type,
                extra={"sample_id": sample_id, **_event_to_log_dict(event)},
            )

        if on_event is not None:
            await on_event(event)

        accumulator.handle(event)

    messages = accumulator.finalize()
    return Trajectory(messages=messages)


async def run_external_agent(
    driver: ExternalAgentDriver,
    frontend: Frontend,
    prompt: str,
) -> None:
    """Run an external agent through a frontend.

    Streams events from the driver to the frontend, handling the full
    lifecycle (start, events, stop).

    Args:
        driver: The external agent driver (ClaudeDriver, CodexDriver, etc.)
        frontend: The frontend to render events
        prompt: The task/prompt to send to the agent
    """
    await frontend.start()
    try:
        async for event in driver.run(prompt):
            await frontend.handle_event(event)
    finally:
        await frontend.stop()
