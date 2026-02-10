"""InteractiveRunner - frontend-agnostic agent loop.

This module provides the core interactive agent loop that works with any
frontend implementing the Frontend protocol. The runner handles:
- Agent state management
- Input/output coordination
- Tool confirmation
- Session persistence
- Interruption handling

The frontend is responsible for:
- Rendering stream events
- Collecting user input
- Displaying loading indicators

Design: Uses run_agent() with proper callbacks instead of wrapping it in
an outer loop. All control flow is handled via RunConfig callbacks:
- handle_no_tool: Get input and continue, or stop for detached/single_turn
- handle_stop: Check stop conditions
- on_input: Get user input via frontend
"""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from types import FrameType
from typing import TYPE_CHECKING

import trio

from ..agents import Actor, AgentState, run_agent
from ..dtypes import (
    Endpoint,
    Environment,
    Message,
    RunConfig,
    StopReason,
    StreamEvent,
    ToolCall,
    ToolConfirmResult,
    ToolResult,
    Trajectory,
)

if TYPE_CHECKING:
    from ..store import SessionStore
    from .protocol import Frontend


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for InteractiveRunner.

    Groups session management and behavior flags to reduce constructor arity.
    """

    # Session management
    session_store: SessionStore | None = None
    session_id: str | None = None
    parent_session_id: str | None = None
    branch_point: int | None = None

    # Behavior flags
    confirm_tools: bool = False
    initial_prompt: str | None = None
    single_turn: bool = False
    detached: bool = False


# ---------------------------------------------------------------------------
# Debug context for interrupt diagnostics
# ---------------------------------------------------------------------------


class _DebugContext:
    """Tracks agent state for debugging hangs/interrupts."""

    def __init__(self) -> None:
        self.phase: str = "initializing"
        self.turn: int = 0
        self.tool_name: str | None = None
        self.stream_start_time: float | None = None
        self.last_stream_event_time: float | None = None
        self.last_operation: str | None = None
        self.last_operation_time: float | None = None

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self._track_operation(f"phase:{phase}")

    def set_streaming(self) -> None:
        self.phase = "streaming"
        self.stream_start_time = time.time()
        self.last_stream_event_time = time.time()
        self._track_operation("streaming_start")

    def on_stream_event(self) -> None:
        self.last_stream_event_time = time.time()

    def set_tool(self, name: str) -> None:
        self.phase = "tool_execution"
        self.tool_name = name
        self._track_operation(f"tool:{name}")

    def _track_operation(self, op: str) -> None:
        now = time.time()
        if self.last_operation_time and self.last_operation:
            elapsed = now - self.last_operation_time
            if elapsed > 5.0:
                self._log_slow_operation(self.last_operation, elapsed)
        self.last_operation = op
        self.last_operation_time = now

    def _log_slow_operation(self, operation: str, elapsed: float) -> None:
        from datetime import datetime
        from pathlib import Path

        log_path = Path.home() / ".rollouts" / "tui-debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} SLOW: {operation} took {elapsed:.1f}s\n")

    def dump(self) -> str:
        lines = [f"Phase: {self.phase}", f"Turn: {self.turn}"]
        if self.tool_name and self.phase == "tool_execution":
            lines.append(f"Tool: {self.tool_name}")
        if self.stream_start_time and self.phase == "streaming":
            elapsed = time.time() - self.stream_start_time
            lines.append(f"Streaming for: {elapsed:.1f}s")
        return "\n".join(lines)


_debug_ctx = _DebugContext()


def get_debug_context() -> _DebugContext:
    """Get the global debug context for state tracking."""
    return _debug_ctx


# ---------------------------------------------------------------------------
# InteractiveRunner
# ---------------------------------------------------------------------------


class InteractiveRunner:
    """Frontend-agnostic interactive agent runner.

    Uses run_agent() with callbacks - no outer while loop needed.
    Control flow is handled via:
    - handle_no_tool: Gets input and returns updated state to continue
    - handle_stop: Checks single_turn/detached flags
    - Cancellation: SIGINT cancels the agent scope

    Example:
        frontend = NoneFrontend()
        runner = InteractiveRunner(trajectory, endpoint, frontend, env)
        states = await runner.run()
    """

    def __init__(
        self,
        trajectory: Trajectory,
        endpoint: Endpoint,
        frontend: Frontend,
        environment: Environment | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        self.trajectory = trajectory
        self.endpoint = endpoint
        self.frontend = frontend
        self.environment = environment

        cfg = config or RunnerConfig()
        self.session_store = cfg.session_store
        self.session_id = cfg.session_id
        self.parent_session_id = cfg.parent_session_id
        self.branch_point = cfg.branch_point
        self.confirm_tools = cfg.confirm_tools
        self.initial_prompt = cfg.initial_prompt
        self.single_turn = cfg.single_turn
        self.detached = cfg.detached

        self._cancel_scope: trio.CancelScope | None = None

    async def run(self) -> list[AgentState]:
        """Run interactive agent loop.

        Returns list of agent states from the run.
        """
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

        try:
            await self.frontend.start()
            self._render_history_if_resuming()
            self._update_frontend_status()

            initial_state = await self._create_initial_state()
            run_config = self._create_run_config()

            self._cancel_scope = trio.CancelScope()
            with self._cancel_scope:
                states = await run_agent(initial_state, run_config)

            self._update_session_id_from_states(states)
            return states

        finally:
            signal.signal(signal.SIGINT, original_handler)
            await self._cleanup()

    # -----------------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------------

    def _render_history_if_resuming(self) -> None:
        if self.trajectory.messages and hasattr(self.frontend, "render_history"):
            self.frontend.render_history(self.trajectory.messages)

    async def _create_initial_state(self) -> AgentState:
        """Create initial agent state with first user message."""
        first_input = self.initial_prompt
        if not first_input:
            first_input = await self.frontend.get_input()

        initial_trajectory = Trajectory(
            messages=self.trajectory.messages + [Message(role="user", content=first_input)]
        )

        return AgentState(
            actor=Actor(
                trajectory=initial_trajectory,
                endpoint=self.endpoint,
                tools=self.environment.get_tools() if self.environment else [],
            ),
            environment=self.environment,
            session_id=self.session_id,
            parent_session_id=self.parent_session_id,
            branch_point=self.branch_point,
            confirm_tools=self.confirm_tools,
        )

    def _create_run_config(self) -> RunConfig:
        """Create RunConfig with all callbacks."""
        return RunConfig(
            on_chunk=self._on_stream_event,
            on_input=self._on_input,
            confirm_tool=self._on_confirm_tool,
            handle_stop=self._on_stop,
            handle_no_tool=self._on_no_tool,
            session_store=self.session_store,
            cancel_scope=self._cancel_scope,
        )

    # -----------------------------------------------------------------------
    # RunConfig callbacks
    # -----------------------------------------------------------------------

    async def _on_stream_event(self, event: StreamEvent) -> None:
        """Route stream event to frontend."""
        await self.frontend.handle_event(event)

    async def _on_input(self, prompt: str) -> str:
        """Get user input via frontend."""
        return await self.frontend.get_input(prompt)

    async def _on_confirm_tool(
        self, tool_call: ToolCall, state: AgentState, config: RunConfig
    ) -> tuple[AgentState, ToolConfirmResult]:
        """Handle tool confirmation via frontend."""
        if not state.confirm_tools:
            return state, ToolConfirmResult(proceed=True)

        approved = await self.frontend.confirm_tool(tool_call)
        if approved:
            return state, ToolConfirmResult(proceed=True)

        return state, ToolConfirmResult(
            proceed=False,
            tool_result=ToolResult(
                tool_call_id=tool_call.id, is_error=True, error="Rejected by user"
            ),
        )

    def _on_stop(self, state: AgentState) -> AgentState:
        """Check stop conditions. No max turns in interactive mode."""
        return state

    async def _on_no_tool(self, state: AgentState, config: RunConfig) -> AgentState:
        """Handle response without tool calls.

        This is the key callback that controls interactive behavior:
        - single_turn: Stop immediately
        - detached: Write pending_input and stop
        - interactive: Get input and continue
        """
        self._update_frontend_status(state)

        if self.single_turn:
            return dc_replace(state, stop=StopReason.NO_TOOL_CALLED)

        if self.detached:
            await self._write_pending_input(state)
            return dc_replace(state, stop=StopReason.NEEDS_INPUT)

        # Interactive: get input and continue
        user_input = await config.on_input("Enter your message: ")
        new_trajectory = Trajectory(
            messages=state.actor.trajectory.messages + [Message(role="user", content=user_input)]
        )
        return dc_replace(state, actor=dc_replace(state.actor, trajectory=new_trajectory))

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    async def _write_pending_input(self, state: AgentState) -> None:
        """Write pending_input.json for detached mode."""
        session_id = state.session_id or self.session_id
        if not (self.session_store and session_id):
            return

        last_message = self._extract_last_assistant_message(state)
        await self.session_store.write_pending_input(
            session_id, {"type": "no_tools", "last_message": last_message}
        )

    def _extract_last_assistant_message(self, state: AgentState) -> str:
        """Extract text from the last assistant message for pending_input context."""
        for msg in reversed(state.actor.trajectory.messages):
            if msg.role == "assistant":
                content = msg.content
                if isinstance(content, str):
                    return content[:500]
                if isinstance(content, list):
                    texts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    return " ".join(texts)[:500]
        return ""

    def _update_session_id_from_states(self, states: list[AgentState]) -> None:
        """Update self.session_id from final state."""
        if states and states[-1].session_id:
            self.session_id = states[-1].session_id

    def _handle_sigint(self, signum: int, frame: FrameType | None) -> None:
        """Handle SIGINT - cancel agent and dump debug context."""
        print("\n[SIGINT] Interrupting agent...", file=sys.stderr)
        print(f"[DEBUG] {_debug_ctx.dump()}", file=sys.stderr)
        if self._cancel_scope:
            self._cancel_scope.cancel()

    def _update_frontend_status(self, state: AgentState | None = None) -> None:
        """Update frontend status bar if supported."""
        if not hasattr(self.frontend, "set_status"):
            return

        kwargs: dict = {
            "model": f"{self.endpoint.provider}/{self.endpoint.model}",
            "session_id": self.session_id,
        }

        if state:
            total_input, total_output, total_cost = 0, 0, 0.0
            for completion in state.actor.trajectory.completions:
                if completion.usage:
                    total_input += (
                        completion.usage.input_tokens + completion.usage.cache_read_tokens
                    )
                    total_output += (
                        completion.usage.output_tokens + completion.usage.reasoning_tokens
                    )
                    total_cost += completion.usage.cost.total
            kwargs.update(input_tokens=total_input, output_tokens=total_output, cost=total_cost)

        if self.environment and hasattr(self.environment, "get_status_info"):
            kwargs["env_info"] = self.environment.get_status_info()

        self.frontend.set_status(**kwargs)

    async def _cleanup(self) -> None:
        """Stop frontend and print session info."""
        await self.frontend.stop()

        if self.session_id:
            print(f"\nSession: {self.session_id}")
            print(f"Resume with: --session {self.session_id}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_interactive(
    trajectory: Trajectory,
    endpoint: Endpoint,
    frontend: Frontend,
    environment: Environment | None = None,
    config: RunnerConfig | None = None,
) -> list[AgentState]:
    """Run an interactive agent with any frontend.

    Args:
        trajectory: Initial conversation trajectory
        endpoint: LLM endpoint configuration
        frontend: Frontend implementation
        environment: Optional environment for tool execution
        config: Runner configuration (session management and behavior flags)

    Returns:
        List of agent states from the run
    """
    runner = InteractiveRunner(
        trajectory=trajectory,
        endpoint=endpoint,
        frontend=frontend,
        environment=environment,
        config=config,
    )
    return await runner.run()
