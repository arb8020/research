"""
Interactive TUI agent runner.

Provides a complete interactive agent loop with TUI rendering.
"""

from __future__ import annotations

import signal
import sys
from typing import Optional

import trio

from rollouts.agents import AgentState, Actor, run_agent
from rollouts.dtypes import (
    ContentBlock,
    Endpoint,
    Environment,
    Message,
    RunConfig,
    StreamDone,
    StreamEvent,
    TextContent,
    TextDelta,
    TextEnd,
    ThinkingContent,
    ThinkingDelta,
    ThinkingEnd,
    ToolCallContent,
    ToolCallEnd,
    ToolResultReceived,
    Trajectory,
    ToolCall,
    ToolConfirmResult,
    StopReason,
)

from .terminal import ProcessTerminal
from .tui import TUI
from .agent_renderer import AgentRenderer
from .components.input import Input
from .components.spacer import Spacer
from .sessions import Session, append_message


class InteractiveAgentRunner:
    """Interactive agent runner with TUI."""

    def __init__(
        self,
        initial_trajectory: Trajectory,
        endpoint: Endpoint,
        environment: Optional[Environment] = None,
        max_turns: int = 50,
        session: Optional[Session] = None,
        theme_name: str = "dark",
    ) -> None:
        """Initialize interactive agent runner.

        Args:
            initial_trajectory: Initial conversation trajectory
            endpoint: LLM endpoint configuration
            environment: Optional environment for tool execution
            max_turns: Maximum number of turns
            session: Optional session for persistence
            theme_name: Theme name (dark or rounded)
        """
        self.initial_trajectory = initial_trajectory
        self.endpoint = endpoint
        self.theme_name = theme_name
        self.environment = environment
        self.max_turns = max_turns
        self.session = session

        # TUI components
        self.terminal: Optional[ProcessTerminal] = None
        self.tui: Optional[TUI] = None
        self.renderer: Optional[AgentRenderer] = None
        self.input_component: Optional[Input] = None

        # Input coordination - use Trio memory channels instead of asyncio.Queue
        self.input_send: Optional[trio.MemorySendChannel[str]] = None
        self.input_receive: Optional[trio.MemoryReceiveChannel[str]] = None
        self.input_pending: bool = False
        self.is_first_user_message = True

        # Cancellation
        self.cancel_scope: Optional[trio.CancelScope] = None

        # Message accumulation for session persistence
        self._current_text: str = ""
        self._current_thinking: str = ""
        self._current_thinking_signature: str | None = None
        self._current_tool_calls: list[dict] = []

    def _handle_input_submit(self, text: str) -> None:
        """Handle input submission from TUI (sync wrapper for trio channel send).

        This is called synchronously from the Input component, but we need to
        send to an async channel. We use send_nowait which works because the
        channel is unbuffered (capacity 0) and there's always a receiver waiting.
        """
        if text.strip() and self.input_send:
            self.input_send.send_nowait(text.strip())

    async def _tui_input_handler(self, prompt: str) -> str:
        """Async input handler for RunConfig.on_input.

        Args:
            prompt: Prompt string (not used in TUI, but required by signature)

        Returns:
            User input string
        """
        # Show input component if not already visible
        if not self.input_pending:
            self.input_pending = True
            if self.input_component and self.tui:
                self.tui.set_focus(self.input_component)
                self.tui.request_render()

        # Wait for input from channel
        if self.input_receive is None:
            raise RuntimeError("Input channel not initialized")
        user_input = await self.input_receive.receive()
        self.input_pending = False

        # Clear input component
        if self.input_component:
            self.input_component.set_text("")
            if self.tui:
                self.tui.set_focus(None)
                self.tui.request_render()

        # Add user message to chat
        if self.renderer:
            self.renderer.add_user_message(user_input, is_first=self.is_first_user_message)
            self.is_first_user_message = False

        # Persist user message to session
        if self.session:
            append_message(self.session, Message(role="user", content=user_input))

        return user_input

    async def _handle_stream_event(self, event: StreamEvent) -> None:
        """Handle streaming event - render and accumulate for persistence."""
        # Pass to renderer
        if self.renderer:
            await self.renderer.handle_event(event)

        # Accumulate for session persistence
        if isinstance(event, ThinkingDelta):
            self._current_thinking += event.delta
        elif isinstance(event, ThinkingEnd):
            # Capture thinking signature if available
            # Note: signature comes from the aggregate_anthropic_stream function
            pass  # Signature is accumulated in the stream aggregator
        elif isinstance(event, TextDelta):
            self._current_text += event.delta
        elif isinstance(event, ToolCallEnd):
            self._current_tool_calls.append({
                "id": event.tool_call.id,
                "name": event.tool_call.name,
                "arguments": event.tool_call.args,
            })
        elif isinstance(event, StreamDone):
            # Persist accumulated assistant message
            if self.session and (self._current_thinking or self._current_text or self._current_tool_calls):
                # Build content blocks using proper dataclass types
                content: list[ContentBlock] = []
                if self._current_thinking:
                    content.append(ThinkingContent(
                        thinking=self._current_thinking,
                        thinking_signature=self._current_thinking_signature
                    ))
                if self._current_text:
                    content.append(TextContent(text=self._current_text))
                for tc in self._current_tool_calls:
                    content.append(ToolCallContent(
                        id=tc["id"],
                        name=tc["name"],
                        arguments=dict(tc["arguments"]),
                    ))

                assistant_msg = Message(role="assistant", content=content)
                append_message(self.session, assistant_msg)

            # Reset accumulators
            self._current_thinking = ""
            self._current_thinking_signature = None
            self._current_text = ""
            self._current_tool_calls = []
        elif isinstance(event, ToolResultReceived):
            # Persist tool result as separate message
            if self.session:
                tool_msg = Message(
                    role="tool",
                    content=event.content,
                    tool_call_id=event.tool_call_id,
                )
                append_message(self.session, tool_msg)

    def _handle_sigint(self, signum, frame) -> None:
        """Handle SIGINT (Ctrl+C) - cancel agent.

        Note: In raw terminal mode, SIGINT is not generated by Ctrl+C.
        Ctrl+C is handled as input data (ASCII 3) in the input_reading_loop.
        """
        if self.cancel_scope:
            self.cancel_scope.cancel()

    async def run(self) -> list[AgentState]:
        """Run interactive agent loop.

        Returns:
            List of agent states from the run
        """
        # Create terminal and TUI with selected theme
        from .theme import DARK_THEME, ROUNDED_THEME
        theme = ROUNDED_THEME if self.theme_name == "rounded" else DARK_THEME

        self.terminal = ProcessTerminal()
        self.tui = TUI(self.terminal, theme=theme, debug=True)  # TODO: make debug configurable

        # Create renderer
        self.renderer = AgentRenderer(self.tui)

        # Render history from initial trajectory (for resumed sessions)
        # Skip system messages, render user/assistant/tool messages
        if self.initial_trajectory.messages:
            self.renderer.render_history(self.initial_trajectory.messages)
            # Mark that we've already shown messages, so next user message isn't "first"
            self.is_first_user_message = False

            # Debug: dump chat state after loading history
            import os
            if os.environ.get("ROLLOUTS_DEBUG_CHAT"):
                self.renderer.debug_dump_chat()

        # Create input component with theme
        # Add spacer above input box for visual separation
        self.tui.add_child(Spacer(1))
        self.input_component = Input(theme=self.tui.theme)
        self.input_component.set_on_submit(self._handle_input_submit)
        self.tui.add_child(self.input_component)
        # Add spacers below input box to push it up from the bottom
        self.tui.add_child(Spacer(3))

        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._handle_sigint)

        # Start TUI
        self.tui.start()

        try:
            # Create Trio memory channel for input coordination
            self.input_send, self.input_receive = trio.open_memory_channel[str](0)

            # Set up terminal input reading loop
            # Terminal is in raw mode, so we need to poll for input
            async def input_reading_loop():
                """Read terminal input and route to TUI."""
                while True:
                    if self.terminal and self.terminal._running:
                        # Read input (non-blocking)
                        input_data = self.terminal.read_input()
                        if input_data:
                            # Check for Ctrl+C (ASCII 3) first
                            if len(input_data) > 0 and ord(input_data[0]) == 3:
                                # Cancel the agent
                                if self.cancel_scope:
                                    self.cancel_scope.cancel()
                                return
                            # Route to TUI's input handler
                            if self.tui:
                                self.tui._handle_input(input_data)
                    await trio.sleep(0.01)  # Small delay to avoid busy-waiting

            async with trio.open_nursery() as nursery:
                self.cancel_scope = nursery.cancel_scope

                # Start input reading loop in background
                nursery.start_soon(input_reading_loop)

                # Start animation loop in background
                # Why: Loader spinner needs periodic re-renders during blocking operations
                # (e.g. API call before streaming starts). The loop calls request_render()
                # every 80ms when loader is active.
                nursery.start_soon(self.tui.run_animation_loop)

                # Wait for first user message before starting agent
                # This ensures we don't send empty messages to the LLM
                if self.input_component and self.tui:
                    self.tui.set_focus(self.input_component)
                    self.tui.request_render()

                first_message = await self._tui_input_handler("Enter your message: ")

                # Now create initial state with user message in trajectory
                initial_trajectory_with_user = Trajectory(
                    messages=self.initial_trajectory.messages + [
                        Message(role="user", content=first_message)
                    ]
                )

                initial_state = AgentState(
                    actor=Actor(
                        trajectory=initial_trajectory_with_user,
                        endpoint=self.endpoint,
                        tools=self.environment.get_tools() if self.environment else [],
                    ),
                    environment=self.environment,
                )

                # Create run config
                # Auto-confirm all tools (no interactive confirmation in TUI for now)
                async def auto_confirm_tool(tc: ToolCall, state: AgentState, rcfg: RunConfig) -> tuple[AgentState, ToolConfirmResult]:
                    return state, ToolConfirmResult(proceed=True)

                # Handle no-tool response: wait for user input before continuing
                async def handle_no_tool_interactive(state: AgentState, rcfg: RunConfig) -> AgentState:
                    """Wait for user input when LLM responds without tool calls."""
                    from dataclasses import replace as dc_replace

                    # Get user input via the TUI
                    user_input = await rcfg.on_input("Enter your message: ")

                    # Append user message to trajectory
                    new_trajectory = Trajectory(
                        messages=state.actor.trajectory.messages + [
                            Message(role="user", content=user_input)
                        ]
                    )
                    new_actor = dc_replace(state.actor, trajectory=new_trajectory)
                    return dc_replace(state, actor=new_actor)

                run_config = RunConfig(
                    on_chunk=self._handle_stream_event,
                    on_input=self._tui_input_handler,
                    confirm_tool=auto_confirm_tool,
                    handle_stop=self._handle_stop,
                    handle_no_tool=handle_no_tool_interactive,
                )

                # Store agent result
                agent_states = []

                # Run agent in foreground
                try:
                    agent_states = await run_agent(initial_state, run_config)
                except trio.Cancelled:
                    # Agent was cancelled - this is expected
                    agent_states = []

            return agent_states

        finally:
            # Stop TUI
            if self.tui:
                self.tui.stop()
            if self.terminal:
                self.terminal.stop()

    def _handle_stop(self, state: AgentState) -> AgentState:
        """Handle stop condition - check max turns."""
        from dataclasses import replace

        if state.turn_idx >= self.max_turns:
            return replace(state, stop=StopReason.MAX_TURNS)
        return state


async def run_interactive_agent(
    initial_trajectory: Trajectory,
    endpoint: Endpoint,
    environment: Optional[Environment] = None,
    max_turns: int = 50,
    session: Optional[Session] = None,
    theme_name: str = "dark",
) -> list[AgentState]:
    """Run an interactive agent with TUI.

    Args:
        initial_trajectory: Initial conversation trajectory
        endpoint: LLM endpoint configuration
        environment: Optional environment for tool execution
        max_turns: Maximum number of turns
        session: Optional session for persistence
        theme_name: Theme name (dark or rounded)

    Returns:
        List of agent states from the run
    """
    runner = InteractiveAgentRunner(
        initial_trajectory=initial_trajectory,
        endpoint=endpoint,
        environment=environment,
        max_turns=max_turns,
        session=session,
        theme_name=theme_name,
    )
    return await runner.run()

