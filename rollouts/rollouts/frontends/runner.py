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
"""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import replace as dc_replace
from types import FrameType
from typing import TYPE_CHECKING

import trio

from rollouts.agents import Actor, AgentState, run_agent


# Global debug context for interrupt diagnostics
# Thread-safe since signal handlers run on main thread
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
        """Track operation timing and log if previous operation was slow."""
        now = time.time()
        if self.last_operation_time and self.last_operation:
            elapsed = now - self.last_operation_time
            if elapsed > 5.0:  # Log operations that took >5s
                self._log_slow_operation(self.last_operation, elapsed)
        self.last_operation = op
        self.last_operation_time = now

    def _log_slow_operation(self, operation: str, elapsed: float) -> None:
        """Log slow operation to debug file."""
        from datetime import datetime
        from pathlib import Path

        log_path = Path.home() / ".rollouts" / "tui-debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} SLOW_OPERATION: {operation} took {elapsed:.1f}s\n"
            )
            f.write(f"  turn={self.turn}, phase={self.phase}\n")

    def dump(self) -> str:
        """Return debug info string for interrupt diagnostics."""
        lines = [f"Phase: {self.phase}", f"Turn: {self.turn}"]
        if self.tool_name and self.phase == "tool_execution":
            lines.append(f"Tool: {self.tool_name}")
        if self.stream_start_time and self.phase == "streaming":
            elapsed = time.time() - self.stream_start_time
            lines.append(f"Streaming for: {elapsed:.1f}s")
            if self.last_stream_event_time:
                since_last = time.time() - self.last_stream_event_time
                lines.append(f"Time since last event: {since_last:.1f}s")
        if self.last_operation and self.last_operation_time:
            elapsed = time.time() - self.last_operation_time
            lines.append(f"Last operation: {self.last_operation} ({elapsed:.1f}s ago)")
        return "\n".join(lines)


_debug_ctx = _DebugContext()


def get_debug_context() -> _DebugContext:
    """Get the global debug context for state tracking."""
    return _debug_ctx


from rollouts.dtypes import (
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
    from rollouts.store import SessionStore

    from .protocol import Frontend


class InteractiveRunner:
    """Frontend-agnostic interactive agent runner.

    This handles the core agent loop logic, delegating all UI to the frontend.

    Example usage:
        frontend = NoneFrontend()  # or TUIFrontend(), TextualFrontend(), etc.
        runner = InteractiveRunner(
            trajectory=trajectory,
            endpoint=endpoint,
            frontend=frontend,
            environment=env,
        )
        states = await runner.run()
    """

    def __init__(
        self,
        trajectory: Trajectory,
        endpoint: Endpoint,
        frontend: Frontend,
        environment: Environment | None = None,
        session_store: SessionStore | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        branch_point: int | None = None,
        confirm_tools: bool = False,
        initial_prompt: str | None = None,
        single_turn: bool = False,
    ) -> None:
        """Initialize runner.

        Args:
            trajectory: Initial conversation trajectory
            endpoint: LLM endpoint configuration
            frontend: Frontend implementation for UI
            environment: Optional environment for tool execution
            session_store: Optional session store for persistence
            session_id: Optional session ID for resumption
            parent_session_id: Parent session ID when forking
            branch_point: Message index where forking from parent
            confirm_tools: Require confirmation before executing tools
            initial_prompt: Optional initial prompt to send immediately
            single_turn: If True, exit after agent responds (no interactive loop)
        """
        self.trajectory = trajectory
        self.endpoint = endpoint
        self.frontend = frontend
        self.environment = environment
        self.session_store = session_store
        self.session_id = session_id
        self.parent_session_id = parent_session_id
        self.branch_point = branch_point
        self.confirm_tools = confirm_tools
        self.initial_prompt = initial_prompt
        self.single_turn = single_turn

        # Cancellation scopes
        self._cancel_scope: trio.CancelScope | None = None
        self._agent_cancel_scope: trio.CancelScope | None = None
        self._interrupted = False

        # Message queuing for batch input
        self._pending_messages: list[str] = []
        self._is_first_message = True

    async def run(self) -> list[AgentState]:
        """Run interactive agent loop.

        Returns:
            List of agent states from the run
        """
        agent_states: list[AgentState] = []

        # Set up signal handler
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

        try:
            await self.frontend.start()

            # Render history if resuming
            if self.trajectory.messages:
                if hasattr(self.frontend, "render_history"):
                    self.frontend.render_history(self.trajectory.messages)
                self._is_first_message = False

            # Update status if frontend supports it
            self._update_frontend_status()

            try:
                async with trio.open_nursery() as nursery:
                    self._cancel_scope = nursery.cancel_scope

                    # Start frontend input loop if it has one
                    if hasattr(self.frontend, "run_input_loop"):
                        await self.frontend.run_input_loop(nursery)

                    # Queue initial prompt if provided
                    first_input = self.initial_prompt

                    # Get first user input
                    if not first_input:
                        first_input = await self.frontend.get_input()

                    # Create initial state
                    initial_trajectory = Trajectory(
                        messages=self.trajectory.messages
                        + [Message(role="user", content=first_input)]
                    )

                    current_state = AgentState(
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

                    # Main agent loop
                    while True:
                        self._agent_cancel_scope = trio.CancelScope()

                        run_config = RunConfig(
                            on_chunk=self._handle_stream_event,
                            on_input=self._handle_input,
                            confirm_tool=self._handle_tool_confirm,
                            handle_stop=self._handle_stop,
                            handle_no_tool=self._handle_no_tool,
                            session_store=self.session_store,
                            cancel_scope=self._agent_cancel_scope,
                        )

                        with self._agent_cancel_scope:
                            agent_states = await run_agent(current_state, run_config)

                        # Check for abort
                        if agent_states and agent_states[-1].stop == StopReason.ABORTED:
                            latest_state = agent_states[-1]
                            self.session_id = latest_state.session_id or self.session_id

                            if not self._interrupted:
                                # Hard exit (Ctrl+C)
                                break

                            # Soft interrupt (Escape) - continue
                            self._interrupted = False
                            self.frontend.hide_loader()

                            # Get partial response
                            partial_response = None
                            if hasattr(self.frontend, "get_partial_response"):
                                partial_response = self.frontend.get_partial_response()
                            if hasattr(self.frontend, "finalize_partial_response"):
                                self.frontend.finalize_partial_response()
                            if hasattr(self.frontend, "add_system_message"):
                                self.frontend.add_system_message("Interrupted")

                            # Build new messages
                            new_messages = list(latest_state.actor.trajectory.messages)
                            if partial_response:
                                new_messages.append(
                                    Message(
                                        role="assistant",
                                        content=partial_response + "\n\n[interrupted]",
                                    )
                                )

                            # Get next user input
                            user_input = await self.frontend.get_input()
                            new_messages.append(Message(role="user", content=user_input))

                            # Continue with new state
                            current_state = dc_replace(
                                latest_state,
                                actor=dc_replace(
                                    latest_state.actor,
                                    trajectory=Trajectory(messages=new_messages),
                                ),
                                stop=None,
                            )
                        elif agent_states and agent_states[-1].stop == StopReason.TASK_COMPLETED:
                            # Task completed - in interactive mode, show result and continue
                            latest_state = agent_states[-1]
                            self.session_id = latest_state.session_id or self.session_id
                            self.frontend.hide_loader()

                            # Check if environment has a final_answer to display
                            if latest_state.environment and hasattr(
                                latest_state.environment, "_final_answer"
                            ):
                                final_answer = getattr(
                                    latest_state.environment, "_final_answer", None
                                )
                                if final_answer:
                                    # Use add_final_answer if available (TUI), otherwise add_system_message
                                    if hasattr(self.frontend, "add_final_answer"):
                                        self.frontend.add_final_answer(final_answer)
                                    elif hasattr(self.frontend, "add_system_message"):
                                        self.frontend.add_system_message(
                                            f"final_answer()\n\n{final_answer}"
                                        )

                            # In single_turn mode, exit after completion
                            if self.single_turn:
                                break

                            # Get next user input
                            user_input = await self.frontend.get_input()
                            new_messages = list(latest_state.actor.trajectory.messages)
                            new_messages.append(Message(role="user", content=user_input))

                            current_state = dc_replace(
                                latest_state,
                                actor=dc_replace(
                                    latest_state.actor,
                                    trajectory=Trajectory(messages=new_messages),
                                ),
                                stop=None,
                            )
                        else:
                            # Other stop reasons (MAX_TURNS, etc.) - exit
                            if agent_states:
                                self.session_id = agent_states[-1].session_id or self.session_id
                            break

                        self._agent_cancel_scope = None
            except BaseException:
                # Collect feedback even on errors (e.g., EOF, exception groups)
                pass

            return agent_states

        finally:
            signal.signal(signal.SIGINT, original_handler)
            await self.frontend.stop()

            # Collect exit feedback
            if agent_states:
                final_state = agent_states[-1]
                exit_reason = "unknown"
                if final_state.stop:
                    exit_reason = str(final_state.stop).split(".")[-1].lower()

                try:
                    from rollouts.feedback import run_exit_survey

                    await run_exit_survey(
                        final_state,
                        self.endpoint,
                        exit_reason,
                        session_id=self.session_id,
                        skip_check=True,
                    )
                except Exception as e:
                    import sys

                    print(f"[DEBUG] Feedback error: {e}", file=sys.stderr)

            # Print session info
            if self.session_id:
                print(f"\nSession: {self.session_id}")
                print(f"Resume with: --session {self.session_id}")

    async def _handle_stream_event(self, event: StreamEvent) -> None:
        """Route stream event to frontend."""
        await self.frontend.handle_event(event)

    async def _handle_input(self, prompt: str) -> str:
        """Get user input via frontend."""
        return await self.frontend.get_input(prompt)

    async def _handle_tool_confirm(
        self, tool_call: ToolCall, state: AgentState, config: RunConfig
    ) -> tuple[AgentState, ToolConfirmResult]:
        """Handle tool confirmation."""
        if not state.confirm_tools:
            return state, ToolConfirmResult(proceed=True)

        approved = await self.frontend.confirm_tool(tool_call)

        if approved:
            return state, ToolConfirmResult(proceed=True)
        else:
            return state, ToolConfirmResult(
                proceed=False,
                tool_result=ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    error="Rejected by user",
                ),
            )

    async def _handle_no_tool(self, state: AgentState, config: RunConfig) -> AgentState:
        """Handle response without tool calls - get next user input or stop."""
        # Update session_id from state (session may have been created in run_agent)
        if state.session_id:
            self.session_id = state.session_id
            # Also update frontend's session_id if it has one
            if hasattr(self.frontend, "session_id"):
                self.frontend.session_id = state.session_id

        # Update status
        self._update_frontend_status(state)

        # In single_turn mode, stop after first response without tools
        if self.single_turn:
            return dc_replace(state, stop=StopReason.NO_TOOL)

        # Get next input
        user_input = await config.on_input("Enter your message: ")

        # Build new trajectory
        new_messages = [Message(role="user", content=user_input)]
        for pending in self._pending_messages:
            new_messages.append(Message(role="user", content=pending))
        self._pending_messages = []

        new_trajectory = Trajectory(messages=state.actor.trajectory.messages + new_messages)

        return dc_replace(
            state,
            actor=dc_replace(state.actor, trajectory=new_trajectory),
        )

    def _handle_stop(self, state: AgentState) -> AgentState:
        """Check stop conditions. No max turns limit in interactive mode."""
        return state

    def _handle_sigint(self, signum: int, frame: FrameType | None) -> None:
        """Handle SIGINT - cancel and dump debug context."""
        # Dump debug info to help diagnose hangs
        print("\n[SIGINT] Interrupting agent...", file=sys.stderr)
        print(f"[DEBUG] {_debug_ctx.dump()}", file=sys.stderr)

        if self._cancel_scope:
            self._cancel_scope.cancel()

    def _update_frontend_status(self, state: AgentState | None = None) -> None:
        """Update frontend status if supported."""
        if not hasattr(self.frontend, "set_status"):
            return

        model = f"{self.endpoint.provider}/{self.endpoint.model}"
        kwargs = {"model": model, "session_id": self.session_id}

        if state:
            # Calculate token counts
            total_input = 0
            total_output = 0
            total_cost = 0.0
            for completion in state.actor.trajectory.completions:
                if completion.usage:
                    total_input += (
                        completion.usage.input_tokens + completion.usage.cache_read_tokens
                    )
                    total_output += (
                        completion.usage.output_tokens + completion.usage.reasoning_tokens
                    )
                    total_cost += completion.usage.cost.total
            kwargs["input_tokens"] = total_input
            kwargs["output_tokens"] = total_output
            kwargs["cost"] = total_cost

        if self.environment and hasattr(self.environment, "get_status_info"):
            kwargs["env_info"] = self.environment.get_status_info()

        self.frontend.set_status(**kwargs)


async def run_interactive(
    trajectory: Trajectory,
    endpoint: Endpoint,
    frontend: Frontend,
    environment: Environment | None = None,
    session_store: SessionStore | None = None,
    session_id: str | None = None,
    parent_session_id: str | None = None,
    branch_point: int | None = None,
    confirm_tools: bool = False,
    initial_prompt: str | None = None,
    single_turn: bool = False,
) -> list[AgentState]:
    """Run an interactive agent with any frontend.

    Args:
        trajectory: Initial conversation trajectory
        endpoint: LLM endpoint configuration
        frontend: Frontend implementation
        environment: Optional environment for tool execution
        session_store: Optional session store for persistence
        session_id: Optional session ID for resumption
        parent_session_id: Parent session ID when forking
        branch_point: Message index where forking from parent
        confirm_tools: Require confirmation before executing tools
        initial_prompt: Optional initial prompt to send immediately
        single_turn: If True, exit after agent responds (no interactive loop)

    Returns:
        List of agent states from the run
    """
    runner = InteractiveRunner(
        trajectory=trajectory,
        endpoint=endpoint,
        frontend=frontend,
        environment=environment,
        session_store=session_store,
        session_id=session_id,
        parent_session_id=parent_session_id,
        branch_point=branch_point,
        confirm_tools=confirm_tools,
        initial_prompt=initial_prompt,
        single_turn=single_turn,
    )
    return await runner.run()
