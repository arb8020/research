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
from dataclasses import asdict, dataclass, is_dataclass
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

_event_logger = logging.getLogger("rollouts.eval.events")


@dataclass
class _FlushAssistantMessage:
    """Internal accumulator control signal for assistant-message boundaries."""


def _event_to_log_dict(event: StreamEvent) -> dict[str, object]:
    if is_dataclass(event):
        return asdict(event)
    raise TypeError(f"Unsupported stream event for eval logging: {type(event)!r}")


class _EventAccumulator:
    """Accumulate StreamEvents into Messages."""

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._active_text: dict[int, str] = {}
        self._active_thinking: dict[int, str] = {}
        self._active_tools: dict[str, dict] = {}
        self._completed_blocks: list[tuple[int, ContentBlock]] = []

    def handle(self, event: Any) -> None:
        if isinstance(event, _FlushAssistantMessage):
            self._finalize_assistant_message()
            return

        match event:
            case TextStart(content_index=idx):
                self._active_text[idx] = ""
            case TextDelta(content_index=idx, delta=delta):
                if idx in self._active_text:
                    self._active_text[idx] += delta
            case TextEnd(content_index=idx, content=content):
                self._active_text.pop(idx, None)
                self._completed_blocks.append((idx, TextContent(text=content)))
            case ThinkingStart(content_index=idx):
                self._active_thinking[idx] = ""
            case ThinkingDelta(content_index=idx, delta=delta):
                if idx in self._active_thinking:
                    self._active_thinking[idx] += delta
            case ThinkingEnd(content_index=idx, content=content):
                self._active_thinking.pop(idx, None)
                self._completed_blocks.append((idx, ThinkingContent(thinking=content)))
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
            case ToolResultReceived(tool_call_id=tid, content=content, is_error=is_error):
                self._finalize_assistant_message()
                self._messages.append(
                    Message(
                        role="tool",
                        content=content if isinstance(content, str) else str(content),
                        tool_call_id=tid,
                    )
                )

    def _finalize_assistant_message(self) -> None:
        if not self._completed_blocks:
            return
        self._completed_blocks.sort(key=lambda x: x[0])
        blocks = [block for _, block in self._completed_blocks]
        self._messages.append(Message(role="assistant", content=blocks))
        self._completed_blocks = []

    def finalize(self) -> list[Message]:
        self._finalize_assistant_message()
        return self._messages


async def run_driver_to_trajectory(
    driver: ExternalAgentDriver,
    prompt: str,
    sample_id: str | None = None,
    on_event: Callable[[StreamEvent], Awaitable[None]] | None = None,
) -> Trajectory:
    accumulator = _EventAccumulator()

    async for event in driver.run(prompt):
        if isinstance(event, _FlushAssistantMessage):
            accumulator.handle(event)
            continue

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
    await frontend.start()
    try:
        async for event in driver.run(prompt):
            await frontend.handle_event(event)
    finally:
        await frontend.stop()
