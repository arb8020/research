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


from .protocol import InputResult

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

    # Hot-swap support
    cwd: Path | None = None  # Working directory for driver swaps
    enable_swap: bool = True  # Enable /swap command

    # Backend function (default: run_agent from SDK)
    # Can be swapped to run_claude, run_codex for external drivers
    run_fn: RunFn | None = None  # None means use run_agent


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
        self.cwd = cfg.cwd or Path.cwd()
        self.enable_swap = cfg.enable_swap
        self.run_fn: RunFn = cfg.run_fn or run_agent

        self._cancel_scope: trio.CancelScope | None = None

    async def run(self) -> list[AgentState]:
        """Run interactive agent loop.

        Returns list of agent states from the run.
        Handles /swap internally by switching run_fn and continuing.
        """
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

        try:
            await self.frontend.start()
            self._render_history_if_resuming()
            self._update_frontend_status()

            all_states: list[AgentState] = []
            current_state: AgentState | None = None

            # Outer loop handles /swap by switching run_fn
            while True:
                run_config = self._create_run_config()
                self._cancel_scope = trio.CancelScope()
                swap_request: _SwapBackend | None = None
                user_exited = False
                states: list[AgentState] = []

                async with trio.open_nursery() as nursery:
                    # Set up Ctrl+C handler for TUI
                    if hasattr(self.frontend, "set_on_cancel"):

                        def handle_ctrl_c() -> None:
                            print("\n[Ctrl+C] Cancelling...", file=sys.stderr)
                            if self._cancel_scope:
                                self._cancel_scope.cancel()

                        self.frontend.set_on_cancel(handle_ctrl_c)

                    # Start TUI input loop FIRST (before getting initial state)
                    if hasattr(self.frontend, "run_input_loop"):
                        await self.frontend.run_input_loop(nursery)

                    # Get initial state if we don't have one yet
                    if current_state is None:
                        try:
                            current_state = await self._create_initial_state()
                            # Clear initial_prompt after use to prevent re-processing
                            self.initial_prompt = None
                        except _SwapBackend as e:
                            # Swap requested before first message
                            swap_request = e
                            nursery.cancel_scope.cancel()

                        if current_state is None and swap_request is None:
                            # User exited before sending first message
                            user_exited = True
                            nursery.cancel_scope.cancel()

                    # Only run agent if we have a state and no swap pending
                    if current_state is not None and swap_request is None and not user_exited:
                        try:
                            with self._cancel_scope:
                                states = await self.run_fn(current_state, run_config)
                        except _SwapBackend as e:
                            # Capture swap request to handle outside nursery
                            swap_request = e
                        finally:
                            # Cancel background tasks when agent finishes
                            nursery.cancel_scope.cancel()
                    else:
                        nursery.cancel_scope.cancel()

                # Handle user exit
                if user_exited:
                    return []

                all_states.extend(states)

                # Handle swap: update run_fn and continue with current state
                if swap_request is not None:
                    self.run_fn = swap_request.new_run_fn
                    self._show_message(f"Swapped to {swap_request.target}")
                    # Continue from last state (or initial if no states yet)
                    if states:
                        current_state = states[-1]
                        # Clear stop so we continue
                        current_state = dc_replace(current_state, stop=None)
                    continue

                # Normal exit
                break

            self._update_session_id_from_states(all_states)
            return all_states

        finally:
            signal.signal(signal.SIGINT, original_handler)
            await self._cleanup()

    # -----------------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------------

    def _render_history_if_resuming(self) -> None:
        if self.trajectory.messages and hasattr(self.frontend, "render_history"):
            self.frontend.render_history(self.trajectory.messages)

    async def _create_initial_state(self) -> AgentState | None:
        """Create initial agent state with first user message.

        Returns None if user exits before providing input.
        Raises _SwapBackend if user issues /swap command.
        """
        from .protocol import InputExit, SlashCommand, UserMessage

        first_input = self.initial_prompt

        # Check if initial_prompt is a slash command
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

                # Clear initial_prompt before raising to prevent infinite loop
                self.initial_prompt = None
                new_run_fn = partial(run_claude, model="sonnet", cwd=self.cwd)
                raise _SwapBackend(target="claude", new_run_fn=new_run_fn)
            # Other slash commands don't make sense as initial prompt
            # (no session context yet), so pass them through to LLM

        if not first_input:
            # Get first input, handling slash commands
            while True:
                input_result = await self.frontend.get_input()

                match input_result:
                    case InputExit():
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
        - detached: Write pending_input and stop
        - interactive: Get input, handle slash commands, continue
        """
        from .protocol import InputExit, SlashCommand, UserMessage

        self._update_frontend_status(state)

        if self.single_turn:
            return dc_replace(state, stop=StopReason.NO_TOOL_CALLED)

        if self.detached:
            await self._write_pending_input(state)
            return dc_replace(state, stop=StopReason.NEEDS_INPUT)

        # Interactive: get input and handle it
        while True:
            input_result = await config.on_input("Enter your message: ")

            match input_result:
                case InputExit():
                    return dc_replace(state, stop=StopReason.NO_TOOL_CALLED)

                case SlashCommand(name=name, args=args):
                    # Handle slash command - runner has access to session/endpoint/etc
                    handled = await self._handle_slash_command(name, args, state)
                    if handled:
                        continue  # Get next input
                    # Unknown command - pass to LLM as regular message
                    user_text = f"/{name} {args}".strip()

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

    async def _handle_slash_command(self, name: str, args: str, state: AgentState) -> bool:
        """Handle a slash command. Returns True if handled, False to pass to LLM.

        Slash commands are handled here because the runner has access to:
        - self.endpoint (for /model)
        - self.session_store, self.session_id (for /slice)
        - self.environment (for /env)
        - self.run_fn (for /swap)
        """
        if name == "swap":
            target = args.lower() if args else ""
            if target == "claude":
                from functools import partial

                from ..drivers.run_claude import run_claude

                new_run_fn = partial(run_claude, model="sonnet", cwd=self.cwd)
                raise _SwapBackend(target="claude", new_run_fn=new_run_fn)
            if target == "codex":
                # TODO: implement run_codex
                self._show_message("Codex swap not yet implemented")
                return True
            elif target == "rollouts":
                # Swap back to SDK
                raise _SwapBackend(target="rollouts", new_run_fn=run_agent)
            else:
                self._show_message("Usage: /swap <claude|codex|rollouts>")
            return True

        if name == "model":
            return await self._handle_model_command(args, state)

        if name == "thinking":
            return await self._handle_thinking_command(args, state)

        # Unknown command
        self._show_message(f"Unknown command: /{name}\nAvailable: /model, /thinking, /swap")
        return True

    async def _handle_model_command(self, args: str, state: AgentState) -> bool:
        """Handle /model command to switch models."""
        from ..models import get_model

        if not args:
            # Show current model
            self._show_message(f"Current model: {self.endpoint.model}")
            return True

        # Try to parse and switch model
        try:
            new_endpoint = get_model(args, api_key=self.endpoint.api_key)
            self.endpoint = new_endpoint
            # Update actor with new endpoint
            self._show_message(f"Switched to: {new_endpoint.model}")
        except Exception as e:
            self._show_message(f"Cannot switch to {args}: {e}")

        return True

    async def _handle_thinking_command(self, args: str, state: AgentState) -> bool:
        """Handle /thinking command to toggle extended thinking."""
        if not args:
            # Show current state
            thinking = getattr(self.endpoint, "thinking", None)
            if thinking:
                self._show_message(
                    f"Thinking: enabled (budget: {thinking.get('budget_tokens', 'default')})"
                )
            else:
                self._show_message("Thinking: disabled")
            return True

        if args.lower() == "off":
            self.endpoint = dc_replace(self.endpoint, thinking=None)
            self._show_message("Thinking: disabled")
        elif args.lower() == "on":
            self.endpoint = dc_replace(
                self.endpoint, thinking={"type": "enabled", "budget_tokens": 10000}
            )
            self._show_message("Thinking: enabled (budget: 10000)")
        else:
            # Try to parse as budget
            try:
                budget = int(args)
                self.endpoint = dc_replace(
                    self.endpoint, thinking={"type": "enabled", "budget_tokens": budget}
                )
                self._show_message(f"Thinking: enabled (budget: {budget})")
            except ValueError:
                self._show_message("Usage: /thinking [on|off|<budget>]")

        return True

    def _show_message(self, text: str) -> None:
        """Show a message to the user (via frontend if possible, else print)."""
        if hasattr(self.frontend, "add_system_message"):
            self.frontend.add_system_message(text)
        else:
            print(text)

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

    Raises:
        SwapRequest: When user requests /swap to another driver
    """
    runner = InteractiveRunner(
        trajectory=trajectory,
        endpoint=endpoint,
        frontend=frontend,
        environment=environment,
        config=config,
    )
    return await runner.run()
