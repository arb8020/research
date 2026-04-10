from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import trio

from ..dtypes import (
    JsonSerializable,
    StopReason,
    StreamEvent,
    Tool,
    ToolCall,
    ToolConfirmResult,
    ToolResult,
)

if TYPE_CHECKING:
    from ..dtypes import Endpoint, Environment, Trajectory


@dataclass(frozen=True)
class Actor(JsonSerializable):
    trajectory: Trajectory
    endpoint: Endpoint
    tools: list[Tool] = field(default_factory=list)


@dataclass(frozen=True)
class AgentState:
    actor: Actor
    environment: Environment | None
    stop: StopReason | None = None
    error: str | None = None
    turn_idx: int = 0
    pending_tool_calls: list[ToolCall] = field(default_factory=list)
    next_tool_idx: int = 0
    timestamp: str = datetime.now(timezone.utc).isoformat() + "Z"
    session_id: str | None = None
    parent_session_id: str | None = None
    branch_point: int | None = None
    confirm_tools: bool = False
    driver_session_id: str | None = None


async def default_stdin_handler(prompt: str) -> str:
    return await trio.to_thread.run_sync(input, prompt)


async def default_confirm_tool(
    tc: ToolCall, state: AgentState, run_config: RunConfig
) -> tuple[AgentState, ToolConfirmResult]:
    return state, ToolConfirmResult(proceed=True)


async def default_no_tool_handler(state: AgentState, run_config: RunConfig) -> AgentState:
    return replace(state, stop=StopReason.TASK_COMPLETED)


@dataclass(frozen=True)
class RunConfig:
    on_chunk: Callable[[StreamEvent], Awaitable[None]]
    on_input: Callable[[str], Awaitable[Any]] = field(default_factory=lambda: default_stdin_handler)
    confirm_tool: Callable[
        [ToolCall, AgentState, RunConfig], Awaitable[tuple[AgentState, ToolConfirmResult]]
    ] = field(default_factory=lambda: default_confirm_tool)
    handle_tool_error: Callable[[ToolResult, AgentState], AgentState] = lambda tr, s: s
    on_step_start: Callable[[AgentState], AgentState] = lambda s: s
    handle_stop: Callable[[AgentState], AgentState] = lambda s: s
    handle_no_tool: Callable[[AgentState, RunConfig], Awaitable[AgentState]] = field(
        default_factory=lambda: default_no_tool_handler
    )
    user_message_for_thinking: str | None = None
    inline_thinking: str | None = None
    show_progress: bool = False
    cancel_scope: trio.CancelScope | None = None
    interrupt_flag: list[bool] | None = None
    session_store: Any | None = None
    session_id: str | None = None
    api_limiter: trio.CapacityLimiter | None = None
    tool_limiter: trio.CapacityLimiter | None = None
