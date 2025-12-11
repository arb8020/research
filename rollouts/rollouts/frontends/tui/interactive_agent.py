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
    ToolResult,
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
        parent_session_id: str | None = None,
        branch_point: int | None = None,
        confirm_tools: bool = False,
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
            parent_session_id: Parent session ID when forking
            branch_point: Message index where forking from parent
            confirm_tools: Require confirmation before executing tools
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
        self.parent_session_id = parent_session_id
        self.branch_point = branch_point
        self.confirm_tools = confirm_tools

        # TUI components
        self.terminal: ProcessTerminal | None = None
        self.tui: TUI | None = None
        self.renderer: AgentRenderer | None = None
        self.input_component: Input | None = None
        self.loader_container: LoaderContainer | None = None
        self.status_line: "StatusLine | None" = None

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

    def _update_token_counts(self, state: AgentState) -> None:
        """Update status line with cumulative token counts and cost from trajectory."""
        import logging
        logger = logging.getLogger(__name__)

        if not self.status_line:
            logger.debug("_update_token_counts: no status_line")
            return

        total_input = 0
        total_output = 0
        total_cost = 0.0
        completions = state.actor.trajectory.completions
        logger.debug(f"_update_token_counts: {len(completions)} completions")
        for completion in completions:
            if completion.usage:
                logger.debug(f"  usage: in={completion.usage.input_tokens} out={completion.usage.output_tokens} cost={completion.usage.cost.total}")
                total_input += completion.usage.input_tokens + completion.usage.cache_read_tokens
                total_output += completion.usage.output_tokens + completion.usage.reasoning_tokens
                total_cost += completion.usage.cost.total

        logger.debug(f"_update_token_counts: setting tokens {total_input}/{total_output} cost={total_cost}")
        self.status_line.set_tokens(total_input, total_output, total_cost)

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

        # Create status line below input
        from .components.status_line import StatusLine
        self.status_line = StatusLine(theme=self.tui.theme)
        self.status_line.set_session_id(self.session_id)
        self.status_line.set_model(f"{self.endpoint.provider}/{self.endpoint.model}")
        self.tui.add_child(self.status_line)

        # Add spacer after status line to keep it from bottom
        self.tui.add_child(Spacer(5, debug_label="after-status"))

        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._handle_sigint)

        # Start TUI
        self.tui.start()

        # Track agent states across the run (for session_id extraction on Ctrl+C)
        agent_states: list[AgentState] = []

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
                    session_id=self.session_id,  # Set for resumption, None for new session
                    parent_session_id=self.parent_session_id,  # For forking
                    branch_point=self.branch_point,  # For forking
                    confirm_tools=self.confirm_tools,  # Tool confirmation setting
                )

                # Create run config
                # Tool confirmation handlers
                async def auto_confirm_tool(tc: ToolCall, state: AgentState, rcfg: RunConfig) -> tuple[AgentState, ToolConfirmResult]:
                    return state, ToolConfirmResult(proceed=True)

                async def confirm_tool_tui(tc: ToolCall, state: AgentState, rcfg: RunConfig) -> tuple[AgentState, ToolConfirmResult]:
                    """Interactive tool confirmation in TUI."""
                    # Show confirmation prompt
                    if self.renderer:
                        self.renderer.add_system_message(f"⚠️  Tool: {tc.name}({tc.args})\n   [y] execute  [n] reject  [s] skip")

                    resp = await rcfg.on_input("Confirm tool? ")
                    resp = resp.strip().lower()

                    if resp in ('y', 'yes', ''):
                        return state, ToolConfirmResult(proceed=True)
                    elif resp in ('n', 'no'):
                        # Get feedback
                        feedback = await rcfg.on_input("Feedback for LLM: ")
                        return state, ToolConfirmResult(
                            proceed=False,
                            tool_result=ToolResult(
                                tool_call_id=tc.id,
                                is_error=True,
                                error="Rejected by user"
                            ),
                            user_message=feedback.strip() if feedback.strip() else None
                        )
                    else:  # skip
                        return state, ToolConfirmResult(
                            proceed=False,
                            tool_result=ToolResult(
                                tool_call_id=tc.id,
                                is_error=True,
                                error="Skipped by user"
                            )
                        )

                confirm_handler = confirm_tool_tui if self.confirm_tools else auto_confirm_tool

                # Handle no-tool response: wait for user input before continuing
                async def handle_no_tool_interactive(state: AgentState, rcfg: RunConfig) -> AgentState:
                    """Wait for user input when LLM responds without tool calls."""
                    from dataclasses import replace as dc_replace

                    # Update token counts after LLM response
                    self._update_token_counts(state)
                    if self.tui:
                        self.tui.request_render()

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

                current_state = initial_state

                # Main agent loop - handles interrupts and continues
                while True:
                    # Create a new cancel scope for this agent run
                    self.agent_cancel_scope = trio.CancelScope()

                    run_config = RunConfig(
                        on_chunk=self._handle_stream_event,
                        on_input=self._tui_input_handler,
                        confirm_tool=confirm_handler,
                        handle_stop=self._handle_stop,
                        handle_no_tool=handle_no_tool_interactive,
                        session_store=self.session_store,
                        cancel_scope=self.agent_cancel_scope,
                    )

                    # Run agent with cancellation support
                    with self.agent_cancel_scope:
                        agent_states = await run_agent(current_state, run_config)

                    # Check if Ctrl+C was pressed (agent returned with ABORTED status)
                    if agent_states and agent_states[-1].stop == StopReason.ABORTED:
                        # Extract session_id before exiting
                        if agent_states[-1].session_id:
                            self.session_id = agent_states[-1].session_id
                        break  # Exit the while loop

                    # Check if agent was cancelled via Escape key
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

                        # Use latest state from agent (has session_id, latest trajectory)
                        latest_state = agent_states[-1] if agent_states else current_state

                        # Update session_id if it was created during this run
                        if latest_state.session_id and latest_state.session_id != self.session_id:
                            self.session_id = latest_state.session_id
                            if self.status_line:
                                self.status_line.set_session_id(self.session_id)

                        # Update token counts from completions
                        if self.status_line:
                            self._update_token_counts(latest_state)
                            self.tui.request_render()

                        # Build new messages list from latest state
                        new_messages = list(latest_state.actor.trajectory.messages)

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

                        # Update state with new trajectory, preserving session_id
                        from dataclasses import replace as dc_replace
                        new_trajectory = Trajectory(messages=new_messages)
                        current_state = dc_replace(
                            latest_state,
                            actor=dc_replace(latest_state.actor, trajectory=new_trajectory),
                            stop=None,  # Clear any stop reason
                        )
                        # Loop continues with new state
                    else:
                        # Agent completed normally - exit loop
                        # Update session_id and token counts from final state
                        if agent_states:
                            final_state = agent_states[-1]
                            if final_state.session_id and final_state.session_id != self.session_id:
                                self.session_id = final_state.session_id
                                if self.status_line:
                                    self.status_line.set_session_id(self.session_id)
                            # Update token counts
                            if self.status_line:
                                self._update_token_counts(final_state)
                                self.tui.request_render()
                        break

                    self.agent_cancel_scope = None

            # After nursery exits (normal or cancelled), extract session_id from agent states
            if agent_states and agent_states[-1].session_id:
                self.session_id = agent_states[-1].session_id

            return agent_states

        finally:
            # Stop TUI
            if self.tui:
                self.tui.stop()
            if self.terminal:
                self.terminal.stop()

            # Print session ID for easy resume
            if self.session_id:
                # Use \r\n for proper newlines after raw terminal mode
                import sys
                sys.stdout.write(f"\r\nSession: {self.session_id}\r\n")
                sys.stdout.write(f"Resume with: --session {self.session_id}\r\n")
                sys.stdout.flush()

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
    parent_session_id: str | None = None,
    branch_point: int | None = None,
    confirm_tools: bool = False,
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
        parent_session_id: Parent session ID when forking
        branch_point: Message index where forking from parent
        confirm_tools: Require confirmation before executing tools

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
        parent_session_id=parent_session_id,
        branch_point=branch_point,
        confirm_tools=confirm_tools,
    )
    return await runner.run()

