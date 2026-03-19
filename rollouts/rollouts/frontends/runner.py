"""InteractiveRunner - frontend-agnostic agent loop.

This module provides the core interactive agent loop that works with any
frontend implementing the Frontend protocol. The runner handles:
- Agent state management
- Input/output coordination
- Tool confirmation
- Session persistence
- Interruption handling
- Hot-swap to external drivers (Claude Code, Codex)

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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING

import trio

from ..agents import Actor, AgentState, RunConfig, run_agent
from ..agents.session_runtime import ensure_persisted_session
from ..core import (
    Endpoint,
    Environment,
    Message,
    StopReason,
    ToolCall,
    ToolConfirmResult,
    ToolResult,
    Trajectory,
)
from ..dtypes import StreamEvent
from .tui.slash_commands import SlashCommandResult, handle_slash_command

# Type alias for run functions (run_agent, run_claude, run_codex)
RunFn = Callable[[AgentState, RunConfig], Awaitable[list[AgentState]]]


class _SwapBackend(Exception):
    """Internal exception to swap backend mid-session.

    Raised by /swap command, caught by InteractiveRunner.run() to switch run_fn.
    """

    def __init__(self, target: str, new_run_fn: RunFn) -> None:
        self.target = target
        self.new_run_fn = new_run_fn
        super().__init__(f"Swap to {target}")


from .protocol import InputExit, InputInterrupt, InputResult, SlashCommand, UserMessage

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
    bootstrap_input: str | None = None
    single_turn: bool = False
    detached: bool = False

    # Hot-swap support
    cwd: Path | None = None  # Working directory for driver swaps
    enable_swap: bool = True  # Enable /swap command

    # Backend function (default: run_agent from SDK)
    # Can be swapped to run_claude, run_codex for external drivers
    run_fn: RunFn | None = None  # None means use run_agent


@dataclass(frozen=True)
class _SlashHandlingResult:
    """Normalized result for slash command handling in the runner."""

    state: AgentState
    handled: bool
    expanded_text: str | None = None


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
        self.bootstrap_input = cfg.bootstrap_input
        self.single_turn = cfg.single_turn
        self.detached = cfg.detached
        self.cwd = cfg.cwd or Path.cwd()
        self.enable_swap = cfg.enable_swap
        self.run_fn: RunFn = cfg.run_fn or run_agent

        self._cancel_scope: trio.CancelScope | None = None

    async def run(self) -> list[AgentState]:
        """Run interactive agent loop.

        Returns list of agent states from the run.
        Handles /swap internally by switching run_fn and continuing.

        Control flow is explicit via InputResult matching:
        - InputExit: User wants to exit (Ctrl+C double-tap or 'exit')
        - InputInterrupt: User pressed Escape to interrupt current operation
        - UserMessage: Regular text to send to LLM
        - SlashCommand: Command for runner to execute
        """
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

        all_states: list[AgentState] = []
        self._all_states = all_states  # Store for access in _cleanup

        try:
            await self.frontend.start()
            self._render_history_if_resuming()
            self._update_frontend_status()
            current_state: AgentState | None = None

            while True:
                self._cancel_scope = trio.CancelScope()
                run_config = self._create_run_config()
                swap_request: _SwapBackend | None = None
                states: list[AgentState] = []
                should_exit = False

                async with trio.open_nursery() as nursery:
                    # Start TUI input loop (handles key events in background)
                    if hasattr(self.frontend, "run_input_loop"):
                        await self.frontend.run_input_loop(nursery)

                    # Get initial state if we don't have one yet
                    if current_state is None:
                        try:
                            current_state = await self._create_initial_state()
                            self.bootstrap_input = None
                        except _SwapBackend as e:
                            swap_request = e
                            nursery.cancel_scope.cancel()

                        if current_state is None and swap_request is None:
                            should_exit = True
                            nursery.cancel_scope.cancel()

                    # Run agent if we have state
                    if current_state is not None and swap_request is None and not should_exit:
                        current_state = await self._ensure_session(current_state)

                        try:
                            with self._cancel_scope:
                                states = await self.run_fn(current_state, run_config)
                        except _SwapBackend as e:
                            swap_request = e
                        finally:
                            nursery.cancel_scope.cancel()
                    else:
                        nursery.cancel_scope.cancel()

                if should_exit:
                    return all_states

                all_states.extend(states)

                # Handle swap request
                if swap_request is not None:
                    self.run_fn = swap_request.new_run_fn
                    self._show_message(f"Swapped to {swap_request.target}")
                    if states:
                        current_state = states[-1]
                        current_state = dc_replace(current_state, stop=None)
                    continue

                # Check if agent was interrupted or exited
                if states and states[-1].stop in (StopReason.INTERRUPTED, StopReason.ABORTED):
                    # Hide loader and wait for new input
                    if hasattr(self.frontend, "hide_loader"):
                        self.frontend.hide_loader()
                    current_state = None
                    self.trajectory = states[-1].actor.trajectory
                    continue

                # Normal completion - exit loop
                break

            self._update_session_id_from_states(all_states)
            return all_states

        finally:
            signal.signal(signal.SIGINT, original_handler)
            self._update_session_id_from_states(all_states)
            await self._cleanup()

    # -----------------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------------

    def _render_history_if_resuming(self) -> None:
        if self.trajectory.messages and hasattr(self.frontend, "render_history"):
            self.frontend.render_history(self.trajectory.messages)

    async def _create_initial_state(self) -> AgentState | None:
        """Create initial agent state with first user message.

        Returns None if user exits or interrupts before providing input.
        Raises _SwapBackend if user issues /swap command.
        """
        first_input = self.bootstrap_input

        # Check if bootstrap input is a slash command
        if first_input and first_input.startswith("/"):
            space_idx = first_input.find(" ")
            if space_idx == -1:
                name = first_input[1:]
                args = ""
            else:
                name = first_input[1:space_idx]
                args = first_input[space_idx + 1 :].strip()

            if name == "swap" and args.lower() == "claude":
                from functools import partial

                from ..drivers.run_claude import run_claude

                # Clear bootstrap input before raising to prevent infinite loop
                self.bootstrap_input = None
                new_run_fn = partial(run_claude, model="sonnet", cwd=self.cwd)
                raise _SwapBackend(target="claude", new_run_fn=new_run_fn)
            # Other slash commands don't make sense as bootstrap input
            # (no session context yet), so pass them through to LLM

        if not first_input:
            # Get first input, handling slash commands
            while True:
                input_result = await self.frontend.get_input()

                match input_result:
                    case InputExit() | InputInterrupt():
                        return None

                    case SlashCommand(name=name, args=args):
                        # Handle swap at initial state
                        if name == "swap" and args.lower() == "claude":
                            from functools import partial

                            from ..drivers.run_claude import run_claude

                            new_run_fn = partial(run_claude, model="sonnet", cwd=self.cwd)
                            raise _SwapBackend(target="claude", new_run_fn=new_run_fn)
                        # Other commands don't make sense before first message
                        print(f"Cannot use /{name} before sending a message")
                        continue

                    case UserMessage(text=text):
                        first_input = text
                        break

                    case _:
                        # Legacy string (for backwards compat)
                        first_input = str(input_result)
                        break

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

    async def _ensure_session(self, state: AgentState) -> AgentState:
        """Create session if needed, return state with session_id set.

        Session creation happens HERE in the runner (orchestration layer),
        not inside run_fn. This ensures session_id is available even if
        run_fn is cancelled or crashes.
        """
        if not self.session_store:
            return state

        if state.session_id:
            self.session_id = state.session_id

        ensured_state = await ensure_persisted_session(state, self.session_store)
        self.session_id = ensured_state.session_id
        persisted_trajectory = dc_replace(
            ensured_state.actor.trajectory,
            session=dc_replace(
                ensured_state.actor.trajectory.session,
                session_id=ensured_state.session_id,
                endpoint=ensured_state.actor.endpoint,
            ),
        )
        self.trajectory = persisted_trajectory
        return dc_replace(
            ensured_state,
            actor=dc_replace(ensured_state.actor, trajectory=persisted_trajectory),
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

    async def _on_input(self, prompt: str) -> InputResult:
        """Get user input via frontend. Returns InputResult type."""
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
        - detached: Stop and leave the session resumable
        - interactive: Get input, handle slash commands, continue
        """
        self._update_frontend_status(state)

        if self.single_turn:
            return dc_replace(state, stop=StopReason.NO_TOOL_CALLED)

        if self.detached:
            return dc_replace(state, stop=StopReason.NO_TOOL_CALLED)

        # Interactive: get input and handle it
        while True:
            input_result = await config.on_input("Enter your message: ")

            match input_result:
                case InputExit():
                    return dc_replace(state, stop=StopReason.NO_TOOL_CALLED)

                case InputInterrupt():
                    # User pressed Escape - signal interrupted so main loop handles it
                    return dc_replace(state, stop=StopReason.INTERRUPTED)

                case SlashCommand(name=name, args=args):
                    slash_result = await self._handle_slash_command(name, args, state)
                    state = slash_result.state
                    if slash_result.handled:
                        continue  # Get next input
                    user_text = slash_result.expanded_text or f"/{name} {args}".strip()

                case UserMessage(text=user_text):
                    pass  # Fall through to add message

                case _:
                    # Legacy string return (for backwards compatibility during transition)
                    user_text = str(input_result)

            # Add user message and continue
            new_trajectory = Trajectory(
                messages=state.actor.trajectory.messages + [Message(role="user", content=user_text)]
            )
            return dc_replace(state, actor=dc_replace(state.actor, trajectory=new_trajectory))

    async def _handle_slash_command(
        self, name: str, args: str, state: AgentState
    ) -> _SlashHandlingResult:
        """Handle slash commands through the shared slash-command contract."""
        command_text = f"/{name} {args}".strip()
        result = await handle_slash_command(self, command_text)
        return await self._apply_slash_command_result(state, result)

    async def _apply_slash_command_result(
        self,
        state: AgentState,
        result: SlashCommandResult,
    ) -> _SlashHandlingResult:
        """Apply a shared slash-command result to runner and agent state."""
        if result.message:
            self._show_message(result.message)

        if result.swap_target:
            await self._raise_swap_backend(result.swap_target)

        if not result.handled:
            return _SlashHandlingResult(
                state=state,
                handled=False,
                expanded_text=result.expanded_text,
            )

        next_state = state
        if result.new_session_id:
            switched = await self.switch_session(
                result.new_session_id,
                environment=result.new_environment,
            )
            if not switched:
                return _SlashHandlingResult(state=state, handled=True)

            next_state = self._build_state_for_active_session(state)
        elif result.new_endpoint or result.new_environment:
            next_state = self._apply_in_place_runtime_update(
                state,
                endpoint=result.new_endpoint,
                environment=result.new_environment,
                trajectory=result.new_trajectory,
            )

        return _SlashHandlingResult(state=next_state, handled=True)

    async def _raise_swap_backend(self, target: str) -> None:
        """Translate slash-command swap requests into runner backend swaps."""
        if target == "claude":
            from functools import partial

            from ..drivers.run_claude import run_claude

            new_run_fn = partial(run_claude, model="sonnet", cwd=self.cwd)
            raise _SwapBackend(target="claude", new_run_fn=new_run_fn)
        if target == "codex":
            self._show_message("Codex swap not yet implemented")
            return
        if target == "rollouts":
            raise _SwapBackend(target="rollouts", new_run_fn=run_agent)
        self._show_message("Usage: /swap <claude|codex|rollouts>")

    async def switch_session(
        self,
        new_session_id: str,
        *,
        environment: Environment | None = None,
    ) -> bool:
        """Switch the runner to a persisted child session/trajectory."""
        if not self.session_store:
            return False

        session, err = await self.session_store.get(new_session_id)
        if err or not session:
            return False

        self.session_id = new_session_id
        self.parent_session_id = session.parent_id
        self.branch_point = session.branch_point
        self.endpoint = session.endpoint
        self.trajectory = session
        if environment is not None:
            self.environment = environment
        return True

    def _build_state_for_active_session(self, state: AgentState) -> AgentState:
        """Rebuild agent state from the runner's active session handle."""
        if not self.trajectory.session_id:
            return state

        tools = self.environment.get_tools() if self.environment else []
        return dc_replace(
            state,
            actor=dc_replace(
                state.actor,
                trajectory=self.trajectory,
                endpoint=self.endpoint,
                tools=tools,
            ),
            environment=self.environment,
            session_id=self.session_id,
            parent_session_id=self.parent_session_id,
            branch_point=self.branch_point,
        )

    def _apply_in_place_runtime_update(
        self,
        state: AgentState,
        *,
        endpoint: Endpoint | None = None,
        environment: Environment | None = None,
        trajectory: Trajectory | None = None,
    ) -> AgentState:
        """Apply non-forking runtime updates directly to the active state."""
        current_trajectory = trajectory or state.actor.trajectory
        if endpoint is not None:
            self.endpoint = endpoint
        if environment is not None:
            self.environment = environment

        if endpoint is not None:
            current_trajectory = dc_replace(
                current_trajectory,
                session=dc_replace(current_trajectory.session, endpoint=endpoint),
            )

        self.trajectory = current_trajectory

        tools = self.environment.get_tools() if self.environment else []
        return dc_replace(
            state,
            actor=dc_replace(
                state.actor,
                trajectory=current_trajectory,
                endpoint=self.endpoint,
                tools=tools,
            ),
            environment=self.environment,
        )

    def _show_message(self, text: str) -> None:
        """Show a message to the user (via frontend if possible, else print)."""
        if hasattr(self.frontend, "add_system_message"):
            self.frontend.add_system_message(text)
        else:
            print(text)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _update_session_id_from_states(self, states: list[AgentState]) -> None:
        """Update self.session_id from final state."""
        if states:
            # Prefer session_id, fall back to driver_session_id (for Claude/Codex drivers)
            final_state = states[-1]
            if final_state.session_id:
                self.session_id = final_state.session_id
            elif final_state.driver_session_id:
                self.session_id = final_state.driver_session_id

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
        """Stop frontend and print session info with token usage."""
        await self.frontend.stop()

        if self.session_id:
            # Calculate total token usage from all states
            total_input = 0
            total_output = 0
            total_reasoning = 0
            total_cache_read = 0
            total_cache_write = 0
            total_cost = 0.0

            for state in getattr(self, "_all_states", []):
                for completion in state.actor.trajectory.completions:
                    if completion.usage:
                        total_input += completion.usage.input_tokens
                        total_output += completion.usage.output_tokens
                        total_reasoning += completion.usage.reasoning_tokens
                        total_cache_read += completion.usage.cache_read_tokens
                        total_cache_write += completion.usage.cache_write_tokens
                        total_cost += completion.usage.cost.total

            # Format token counts (similar to codex format)
            def fmt_tokens(n: int) -> str:
                if n < 1000:
                    return f"{n:,}"
                if n < 1_000_000:
                    return f"{n / 1000:.1f}k"
                return f"{n / 1_000_000:.2f}M"

            # Build token usage line
            parts = []
            if total_input > 0:
                parts.append(f"input={fmt_tokens(total_input)}")
            if total_output > 0:
                parts.append(f"output={fmt_tokens(total_output)}")
            if total_reasoning > 0:
                parts.append(f"reasoning={fmt_tokens(total_reasoning)}")
            if total_cache_read > 0:
                parts.append(f"cached={fmt_tokens(total_cache_read)}")
            if total_cache_write > 0:
                parts.append(f"cache_write={fmt_tokens(total_cache_write)}")

            print(
                f"\nToken usage: total={fmt_tokens(total_input + total_output + total_reasoning + total_cache_read)} "
                + f"input={fmt_tokens(total_input)} (+ {fmt_tokens(total_cache_read)} cached) "
                + f"output={fmt_tokens(total_output)}"
                + (f" (reasoning {fmt_tokens(total_reasoning)})" if total_reasoning > 0 else "")
                + f" cost=${total_cost:.4f}"
            )
            print(f"\nTo continue this session, run: rollouts resume {self.session_id}")


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

    Supports hot-swap to external drivers via /swap command:
        /swap claude   - Switch to Claude Code
        /swap codex    - Switch to Codex

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
