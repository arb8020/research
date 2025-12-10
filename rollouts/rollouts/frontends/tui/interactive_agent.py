"""
Interactive TUI agent runner.

Provides a complete interactive agent loop with TUI rendering.
Session persistence is handled by run_agent() via RunConfig.session_store.
"""

from __future__ import annotations

import signal
import sys
from typing import TYPE_CHECKING

import trio

from rollouts.agents import AgentState, Actor, run_agent
from rollouts.dtypes import (
    Endpoint,
    Environment,
    Message,
    RunConfig,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
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
from .components.loader_container import LoaderContainer

if TYPE_CHECKING:
    from rollouts.store import SessionStore


class InteractiveAgentRunner:
    """Interactive agent runner with TUI."""

    def __init__(
        self,
        initial_trajectory: Trajectory,
        endpoint: Endpoint,
        environment: Environment | None = None,
        max_turns: int = 50,
        session_store: SessionStore | None = None,
        session_id: str | None = None,
        theme_name: str = "dark",
        debug: bool = False,
        debug_layout: bool = False,
    ) -> None:
        """Initialize interactive agent runner.

        Args:
            initial_trajectory: Initial conversation trajectory
            endpoint: LLM endpoint configuration
            environment: Optional environment for tool execution
            max_turns: Maximum number of turns
            session_store: Optional session store for persistence
            session_id: Optional session ID (required if session_store is set)
            theme_name: Theme name (dark or rounded)
            debug: Enable debug logging and chat state dumps
            debug_layout: Show component boundaries and spacing
        """
        self.initial_trajectory = initial_trajectory
        self.endpoint = endpoint
        self.theme_name = theme_name
        self.environment = environment
        self.max_turns = max_turns
        self.session_store = session_store
        self.session_id = session_id
        self.debug = debug
        self.debug_layout = debug_layout

        # TUI components
        self.terminal: ProcessTerminal | None = None
        self.tui: TUI | None = None
        self.renderer: AgentRenderer | None = None
        self.input_component: Input | None = None
        self.loader_container: LoaderContainer | None = None

        # Input coordination - use Trio memory channels instead of asyncio.Queue
        self.input_send: trio.MemorySendChannel[str] | None = None
        self.input_receive: trio.MemoryReceiveChannel[str] | None = None
        self.input_pending: bool = False
        self.is_first_user_message = True

        # Cancellation - separate scope for agent vs entire TUI
        self.cancel_scope: trio.CancelScope | None = None  # Outer nursery scope
        self.agent_cancel_scope: trio.CancelScope | None = None  # Inner agent scope

        # Store for passing multiple messages from input handler to no_tool handler
        self._pending_user_messages: list[str] = []

    def _handle_input_submit(self, text: str) -> None:
        """Handle input submission from TUI (sync wrapper for trio channel send).

        This is called synchronously from the Input component. With a buffered
        channel, messages can be queued while the agent is working.
        """
        if text.strip() and self.input_send:
            try:
                self.input_send.send_nowait(text.strip())
                # Add to visual queue display (only if not currently waiting for input)
                if not self.input_pending and self.input_component:
                    self.input_component.add_queued_message(text.strip())
                    if self.tui:
                        self.tui.request_render()
            except trio.WouldBlock:
                # Buffer full (10 messages) - silently drop
                # Could show a "queue full" indicator in the future
                pass

    def _handle_open_editor(self, current_text: str) -> None:
        """Handle Ctrl+G to open external editor for message composition."""
        if not self.terminal:
            return

        # Run editor (this temporarily exits raw mode)
        edited_content = self.terminal.run_external_editor(current_text)

        # If user saved content, update input and optionally submit
        if edited_content:
            if self.input_component:
                self.input_component.set_text(edited_content)
            # Auto-submit the edited content
            self._handle_input_submit(edited_content)
            # Clear input after submit
            if self.input_component:
                self.input_component.set_text("")

        # Force full redraw
        if self.tui:
            self.tui.request_render()

    async def _handle_slash_command(self, command: str) -> bool:
        """Handle slash commands.

        Args:
            command: The slash command string

        Returns:
            True if command was handled, False if it should be passed to LLM
        """
        # For now, no built-in slash commands
        # User should use --continue with different flags instead
        # Return False to pass to LLM (so /commands become regular messages)
        return False

    async def _tui_input_handler(self, prompt: str) -> str:
        """Async input handler for RunConfig.on_input.

        Args:
            prompt: Prompt string (not used in TUI, but required by signature)

        Returns:
            User input string
        """
        if self.input_receive is None:
            raise RuntimeError("Input channel not initialized")

        # Drain all queued messages (non-blocking)
        queued_messages: list[str] = []
        while True:
            try:
                msg = self.input_receive.receive_nowait()
                queued_messages.append(msg)
                # Remove from visual queue display
                if self.input_component:
                    self.input_component.pop_queued_message()
            except trio.WouldBlock:
                break

        if queued_messages:
            # Store all messages - first one returned, rest stored for handle_no_tool
            user_input = queued_messages[0]
            self._pending_user_messages = queued_messages[1:]
            if self.tui:
                self.tui.request_render()
        else:
            user_input = None
            self._pending_user_messages = []

        if user_input is None:
            # No queued message, show input and wait
            self.input_pending = True
            if self.input_component and self.tui:
                self.tui.set_focus(self.input_component)
                self.tui.request_render()

            user_input = await self.input_receive.receive()
            self.input_pending = False

            # Clear input component
            if self.input_component:
                self.input_component.set_text("")

        # Handle slash commands
        if user_input.startswith("/"):
            handled = await self._handle_slash_command(user_input)
            if handled:
                # Command was handled, request new input
                return await self._tui_input_handler(prompt)

        # Add user message to chat
        if self.renderer:
            self.renderer.add_user_message(user_input, is_first=self.is_first_user_message)
            self.is_first_user_message = False

        # Session persistence is handled by run_agent() via RunConfig.session_store

        return user_input

    async def _handle_stream_event(self, event: StreamEvent) -> None:
        """Handle streaming event - render to TUI.

        Session persistence is handled by run_agent() via RunConfig.session_store.
        """
        if self.renderer:
            await self.renderer.handle_event(event)

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
        from .theme import DARK_THEME, ROUNDED_THEME, MINIMAL_THEME
        if self.theme_name == "rounded":
            theme = ROUNDED_THEME
        elif self.theme_name == "minimal":
            theme = MINIMAL_THEME
        else:
            theme = DARK_THEME

        self.terminal = ProcessTerminal()
        self.tui = TUI(self.terminal, theme=theme, debug=self.debug, debug_layout=self.debug_layout)

        # Create renderer with environment for custom tool formatters
        self.renderer = AgentRenderer(self.tui, environment=self.environment, debug_layout=self.debug_layout)

        # Render history from initial trajectory (for resumed sessions)
        # Render all messages including system messages
        if self.initial_trajectory.messages:
            self.renderer.render_history(self.initial_trajectory.messages, skip_system=False)
            # Mark that we've already shown messages, so next user message isn't "first"
            self.is_first_user_message = False

            # Debug: dump chat state after loading history
            if self.debug:
                self.renderer.debug_dump_chat()

        # Create loader container (for spinner during LLM calls)
        # Loader brings its own "before" spacer when active
        self.loader_container = LoaderContainer(
            spinner_color_fn=self.tui.theme.accent_fg,
            text_color_fn=self.tui.theme.muted_fg,
        )
        self.tui.set_loader_container(self.loader_container)
        self.tui.add_child(self.loader_container)

        # Spacer before input box (always present)
        self.tui.add_child(Spacer(1, debug_label="before-input"))

        # Create input component with theme
        self.input_component = Input(theme=self.tui.theme)
        self.input_component.set_on_submit(self._handle_input_submit)
        self.input_component.set_on_editor(self._handle_open_editor)
        self.tui.add_child(self.input_component)

        # Add spacer after input box to keep it 6 lines from bottom
        self.tui.add_child(Spacer(6, debug_label="after-input"))

        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._handle_sigint)

        # Start TUI
        self.tui.start()

        try:
            # Create Trio memory channel for input coordination
            # Buffered channel allows queuing messages while agent is working
            # TODO: Consider injecting queued messages as system reminders into the
            # streaming context (like Claude Code does) so the agent is aware of
            # pending user input while generating a response, rather than only
            # processing them after the current turn completes.
            self.input_send, self.input_receive = trio.open_memory_channel[str](10)

            # Set up terminal input reading loop
            # Terminal is in raw mode, so we need to poll for input
            async def input_reading_loop():
                """Read terminal input and route to TUI."""
                while True:
                    if self.terminal and self.terminal._running:
                        # Read input (non-blocking)
                        input_data = self.terminal.read_input()
                        if input_data:
                            # Check for Ctrl+C (ASCII 3) - exit TUI entirely
                            if len(input_data) > 0 and ord(input_data[0]) == 3:
                                if self.cancel_scope:
                                    self.cancel_scope.cancel()
                                return
                            # Check for Escape (ASCII 27) - interrupt current agent run
                            if len(input_data) == 1 and ord(input_data[0]) == 27:
                                if self.agent_cancel_scope:
                                    self.agent_cancel_scope.cancel()
                                continue
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

                    # Get user input via the TUI (may drain multiple queued messages)
                    user_input = await rcfg.on_input("Enter your message: ")

                    # Build list of all user messages (first one + any pending)
                    user_messages = [Message(role="user", content=user_input)]
                    for pending_msg in self._pending_user_messages:
                        # Render each pending message in chat
                        if self.renderer:
                            self.renderer.add_user_message(pending_msg, is_first=False)
                        user_messages.append(Message(role="user", content=pending_msg))
                    self._pending_user_messages = []

                    # Append all user messages to trajectory
                    new_trajectory = Trajectory(
                        messages=state.actor.trajectory.messages + user_messages
                    )

                    # Update actor with new trajectory
                    new_actor = dc_replace(
                        state.actor,
                        trajectory=new_trajectory,
                    )

                    return dc_replace(state, actor=new_actor)

                # Store agent result
                agent_states = []
                current_state = initial_state

                # Main agent loop - handles interrupts and continues
                while True:
                    # Create a new cancel scope for this agent run
                    self.agent_cancel_scope = trio.CancelScope()

                    run_config = RunConfig(
                        on_chunk=self._handle_stream_event,
                        on_input=self._tui_input_handler,
                        confirm_tool=auto_confirm_tool,
                        handle_stop=self._handle_stop,
                        handle_no_tool=handle_no_tool_interactive,
                        session_store=self.session_store,
                        session_id=self.session_id,
                        cancel_scope=self.agent_cancel_scope,
                    )

                    # Run agent with cancellation support
                    with self.agent_cancel_scope:
                        agent_states = await run_agent(current_state, run_config)

                    # Check if agent was cancelled
                    if self.agent_cancel_scope.cancelled_caught:
                        # Agent was interrupted - hide loader and show message
                        if self.tui:
                            self.tui.hide_loader()

                        # Get any partial response that was being streamed
                        partial_response = None
                        if self.renderer:
                            partial_response = self.renderer.get_partial_response()
                            self.renderer.finalize_partial_response()
                            self.renderer.add_system_message("Interrupted")

                        # Build new messages list
                        new_messages = list(current_state.actor.trajectory.messages)

                        # If there was a partial response, add it with [interrupted] marker
                        # TODO: Consider whether to include partial response in trajectory.
                        # Pros: Agent knows what it said before being cut off
                        # Cons: Partial text may be confusing, user might not want to see it,
                        #       could be mid-word/mid-thought garbage
                        # Maybe add a flag to control this behavior?
                        if partial_response:
                            new_messages.append(Message(
                                role="assistant",
                                content=partial_response + "\n\n[interrupted]"
                            ))

                        # Wait for new user input
                        user_input = await self._tui_input_handler("Enter your message: ")
                        new_messages.append(Message(role="user", content=user_input))

                        # Update state with new trajectory
                        from dataclasses import replace as dc_replace
                        new_trajectory = Trajectory(messages=new_messages)
                        current_state = dc_replace(
                            current_state,
                            actor=dc_replace(current_state.actor, trajectory=new_trajectory),
                            stop=None,  # Clear any stop reason
                        )
                        # Loop continues with new state
                    else:
                        # Agent completed normally - exit loop
                        break

                    self.agent_cancel_scope = None

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
    environment: Environment | None = None,
    max_turns: int = 50,
    session_store: SessionStore | None = None,
    session_id: str | None = None,
    theme_name: str = "dark",
    debug: bool = False,
    debug_layout: bool = False,
) -> list[AgentState]:
    """Run an interactive agent with TUI.

    Args:
        initial_trajectory: Initial conversation trajectory
        endpoint: LLM endpoint configuration
        environment: Optional environment for tool execution
        max_turns: Maximum number of turns
        session_store: Optional session store for persistence
        session_id: Optional session ID (required if session_store is set)
        theme_name: Theme name (dark or rounded)
        debug: Enable debug logging and chat state dumps
        debug_layout: Show component boundaries and spacing

    Returns:
        List of agent states from the run
    """
    runner = InteractiveAgentRunner(
        initial_trajectory=initial_trajectory,
        endpoint=endpoint,
        environment=environment,
        max_turns=max_turns,
        session_store=session_store,
        session_id=session_id,
        theme_name=theme_name,
        debug=debug,
        debug_layout=debug_layout,
    )
    return await runner.run()

