"""
Interactive TUI agent runner.

Provides a complete interactive agent loop with TUI rendering.
Session persistence is handled by run_agent() via RunConfig.session_store.
"""

from __future__ import annotations

import signal
import sys
from types import FrameType
from typing import TYPE_CHECKING

import trio

from rollouts.agents import Actor, AgentState, run_agent
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
from rollouts.models import get_model

from .agent_renderer import AgentRenderer
from .components.input import Input
from .components.loader_container import LoaderContainer
from .components.spacer import Spacer
from .slash_commands import handle_slash_command
from .terminal import ProcessTerminal, set_active_session_id
from .tui import TUI

if TYPE_CHECKING:
    from rollouts.store import SessionStore

    from .components.status_line import StatusLine


class InteractiveAgentRunner:
    """Interactive agent runner with TUI."""

    def __init__(
        self,
        initial_trajectory: Trajectory,
        endpoint: Endpoint,
        environment: Environment | None = None,
        session_store: SessionStore | None = None,
        session_id: str | None = None,
        theme_name: str = "dark",
        debug: bool = False,
        debug_layout: bool = False,
        parent_session_id: str | None = None,
        branch_point: int | None = None,
        confirm_tools: bool = False,
        initial_prompt: str | None = None,
    ) -> None:
        """Initialize interactive agent runner.

        Args:
            initial_trajectory: Initial conversation trajectory
            endpoint: LLM endpoint configuration
            environment: Optional environment for tool execution
            session_store: Optional session store for persistence
            session_id: Optional session ID (required if session_store is set)
            theme_name: Theme name (dark or rounded)
            debug: Enable debug logging and chat state dumps
            debug_layout: Show component boundaries and spacing
            parent_session_id: Parent session ID when forking
            branch_point: Message index where forking from parent
            confirm_tools: Require confirmation before executing tools
            initial_prompt: Optional initial prompt to send immediately
        """
        self.initial_trajectory = initial_trajectory
        self.endpoint = endpoint
        self.theme_name = theme_name
        self.environment = environment
        self.session_store = session_store
        self.session_id = session_id
        if session_id:
            set_active_session_id(session_id)  # For crash reporting
        self.debug = debug
        self.debug_layout = debug_layout
        self.parent_session_id = parent_session_id
        self.branch_point = branch_point
        self.confirm_tools = confirm_tools
        self.initial_prompt = initial_prompt

        # TUI components
        self.terminal: ProcessTerminal | None = None
        self.tui: TUI | None = None
        self.renderer: AgentRenderer | None = None
        self.input_component: Input | None = None
        self.loader_container: LoaderContainer | None = None
        self.status_line: StatusLine | None = None

        # Input coordination - use Trio memory channels instead of asyncio.Queue
        self.input_send: trio.MemorySendChannel[str] | None = None
        self.input_receive: trio.MemoryReceiveChannel[str] | None = None
        self.input_pending: bool = False
        self.is_first_user_message = True

        # Cancellation - separate scope for agent vs entire TUI
        self.cancel_scope: trio.CancelScope | None = None  # Outer nursery scope
        self.agent_cancel_scope: trio.CancelScope | None = None  # Inner agent scope
        self.escape_pressed: bool = False  # Track if Escape (not Ctrl+C) triggered cancel

        # Store for passing multiple messages from input handler to no_tool handler
        self._pending_user_messages: list[str] = []

        # Track current trajectory for slash commands (updated during agent loop)
        self._current_trajectory: Trajectory | None = None

        # Flag set by /env when environment changes - signals main loop to update current_state
        self._environment_changed: bool = False

        # Flag set by /slice when session switches - signals main loop to rebuild state
        # from self.initial_trajectory and self.endpoint
        self._session_switched: bool = False

        # Tab completion cycling state
        self._tab_cycle_matches: list[str] = []  # Current list of matches
        self._tab_cycle_index: int = 0  # Current position in cycle
        self._tab_cycle_prefix: str = ""  # Original prefix before cycling started

    @property
    def trajectory(self) -> Trajectory:
        """Get the current trajectory (for slash commands like /slice)."""
        # Debug: log which trajectory we're returning
        if self._current_trajectory:
            result = self._current_trajectory
            source = "_current_trajectory"
        else:
            result = self.initial_trajectory
            source = "initial_trajectory"

        # Log to help debug /slice issue
        import sys

        print(
            f"[DEBUG trajectory] source={source}, messages={len(result.messages)}", file=sys.stderr
        )

        return result

    async def switch_session(self, new_session_id: str) -> bool:
        """Switch to a different session (e.g., after /slice).

        Args:
            new_session_id: ID of the session to switch to

        Returns:
            True if switch succeeded, False otherwise
        """
        if not self.session_store:
            return False

        # Load the new session
        session, err = await self.session_store.get(new_session_id)
        if err or not session:
            return False

        # Update session tracking
        self.session_id = new_session_id
        set_active_session_id(new_session_id)

        # Update endpoint from the loaded session
        # This is critical - /slice and /env create child sessions with specific endpoints
        self.endpoint = session.endpoint

        # Update trajectory
        self.initial_trajectory = Trajectory(messages=session.messages)
        self._current_trajectory = self.initial_trajectory

        # Signal main loop to rebuild state from self.initial_trajectory/endpoint
        # This is critical - without this, the next agent loop iteration uses stale state
        self._session_switched = True

        # Update status line
        if self.status_line:
            self.status_line.set_session_id(new_session_id)

        # Clear and re-render chat with new session's messages
        if self.renderer:
            self.renderer.clear_chat()
            self.renderer.render_history(session.messages, skip_system=False)

        if self.tui:
            # Reset render state to force complete re-render after session switch
            # This clears cached previous_lines that may be stale
            self.tui.reset_render_state()
            self.tui.request_render()

        return True

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
        from .utils import strip_terminal_control_sequences

        if not self.terminal:
            return

        # Run editor (this temporarily exits raw mode)
        edited_content = self.terminal.run_external_editor(current_text)

        # Reset TUI state before redrawing - this clears cached render state
        # that may be invalid after returning from the external editor
        if self.tui:
            self.tui.reset_render_state()

        # If user saved content, update input and optionally submit
        if edited_content:
            # Strip any terminal control sequences that may have leaked in
            # (e.g., bracketed paste sequences from the editor session)
            edited_content = strip_terminal_control_sequences(edited_content)
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

    def _handle_input_change(self, text: str) -> None:
        """Handle input text change - update ghost text for autocomplete preview.

        This is called whenever the input text changes. We use it to show
        ghost text (dimmed completion hint) for slash commands.

        Ghost text types:
        1. Command name completion: /mo → "del" (completes to /model)
        2. Argument hint: /model → " [provider/model]" (shows expected args)
        3. Model completion: /model anth → "ropic/claude-sonnet-4-20250514"
        """
        from rollouts.models import get_models, get_providers

        from .slash_commands import get_all_commands, get_command_arg_hint

        if not self.input_component:
            return

        # Reset tab cycling when user types (unless we're in the middle of cycling)
        # We detect cycling by checking if current text matches a cycled completion
        is_cycling = False
        if self._tab_cycle_matches and self._tab_cycle_index < len(self._tab_cycle_matches):
            current_match = self._tab_cycle_matches[self._tab_cycle_index]
            # Check for command cycling: /{command}
            if text == f"/{current_match} ":
                is_cycling = True
            # Check for model cycling: /model {model}
            elif self._tab_cycle_prefix.startswith("/model ") and text == f"/model {current_match}":
                is_cycling = True

        if not is_cycling:
            self._tab_cycle_matches = []
            self._tab_cycle_index = 0
            self._tab_cycle_prefix = ""

        ghost_text = ""

        if text.startswith("/"):
            if " " not in text:
                # No space yet - show command name completion
                prefix = text[1:]  # Remove /
                if prefix:  # Don't show ghost for just "/"
                    commands = get_all_commands()
                    matches = [c.name for c in commands if c.name.startswith(prefix)]

                    if matches:
                        # Show the remainder of the first match as ghost text
                        first_match = matches[0]
                        ghost_text = first_match[len(prefix) :]

                        # If exact match, also show arg hint
                        if prefix == first_match:
                            arg_hint = get_command_arg_hint(first_match)
                            if arg_hint:
                                ghost_text = f" {arg_hint}"
            else:
                # Space present - check for model completion or show arg hint
                space_idx = text.index(" ")
                cmd_name = text[1:space_idx]  # Extract command name
                arg_text = text[space_idx + 1 :]

                if cmd_name == "model" and arg_text:
                    # Model argument - show completion for provider/model
                    all_models: list[str] = []
                    for provider in get_providers():
                        for model in get_models(provider):
                            all_models.append(f"{provider}/{model.id}")

                    # Find matches (prefix match)
                    matches = [m for m in all_models if m.startswith(arg_text)]
                    if matches:
                        first_match = matches[0]
                        ghost_text = first_match[len(arg_text) :]
                elif not arg_text:
                    # No args typed yet - show hint
                    arg_hint = get_command_arg_hint(cmd_name)
                    if arg_hint:
                        ghost_text = arg_hint

        self.input_component.set_ghost_text(ghost_text)

        if self.tui:
            self.tui.request_render()

    def _handle_tab_complete(self, text: str) -> str | None:
        """Handle tab completion for slash commands and model names.

        Supports tab cycling: pressing Tab multiple times cycles through matches.

        Args:
            text: Current input text

        Returns:
            Completed text, or None if no completion
        """
        from rollouts.models import get_models, get_providers

        from .slash_commands import get_all_commands

        if not text.startswith("/"):
            return None

        # Check for /model completion with tab cycling
        if text.startswith("/model "):
            # FIRST: Check if we're continuing a tab cycle for models
            # (must check before finding new matches, otherwise we lose the cycle)
            if (
                self._tab_cycle_matches
                and self._tab_cycle_prefix.startswith("/model ")
                and f"/model {self._tab_cycle_matches[self._tab_cycle_index]}" == text.rstrip()
            ):
                # Continue cycling - move to next match
                self._tab_cycle_index = (self._tab_cycle_index + 1) % len(self._tab_cycle_matches)
                return f"/model {self._tab_cycle_matches[self._tab_cycle_index]}"

            arg = text[7:]  # After "/model "

            # Build list of all provider/model combinations
            all_models: list[str] = []
            for provider in get_providers():
                for model in get_models(provider):
                    all_models.append(f"{provider}/{model.id}")

            # Find matches
            matches = [m for m in all_models if m.startswith(arg)]

            if not matches:
                return None

            # Start new cycle
            self._tab_cycle_matches = matches
            self._tab_cycle_index = 0
            self._tab_cycle_prefix = text

            return f"/model {matches[0]}"

        # Check for /env completion with tab cycling
        if text.startswith("/env "):
            from .slash_commands import _get_available_envs

            # Check if we're continuing a tab cycle for envs
            if (
                self._tab_cycle_matches
                and self._tab_cycle_prefix.startswith("/env ")
                and f"/env {self._tab_cycle_matches[self._tab_cycle_index]}" == text.rstrip()
            ):
                # Continue cycling - move to next match
                self._tab_cycle_index = (self._tab_cycle_index + 1) % len(self._tab_cycle_matches)
                return f"/env {self._tab_cycle_matches[self._tab_cycle_index]}"

            arg = text[5:]  # After "/env "

            # Get available environments
            all_envs = _get_available_envs()
            # Also add "list" as a completable option
            all_options = ["list"] + all_envs

            # Find matches
            matches = [e for e in all_options if e.startswith(arg)]

            if not matches:
                return None

            # Start new cycle
            self._tab_cycle_matches = matches
            self._tab_cycle_index = 0
            self._tab_cycle_prefix = text

            return f"/env {matches[0]}"

        # Check for command name completion with tab cycling
        if " " not in text:
            prefix = text[1:]  # Remove /
            commands = get_all_commands()
            matches = [c.name for c in commands if c.name.startswith(prefix)]

            if not matches:
                return None

            # Check if we're continuing a tab cycle
            # (text matches a previously completed command)
            if (
                self._tab_cycle_matches
                and text.rstrip() == f"/{self._tab_cycle_matches[self._tab_cycle_index]}"
            ):
                # Continue cycling - move to next match
                self._tab_cycle_index = (self._tab_cycle_index + 1) % len(self._tab_cycle_matches)
                next_match = self._tab_cycle_matches[self._tab_cycle_index]
                return f"/{next_match} "

            # Start new cycle if we have matches
            if len(matches) >= 1:
                # Initialize cycling state
                self._tab_cycle_matches = matches
                self._tab_cycle_index = 0
                self._tab_cycle_prefix = prefix

                # Complete to first match
                return f"/{matches[0]} "

        return None

    async def _handle_slash_command(self, command: str) -> tuple[bool, str | None]:
        """Handle slash commands.

        Args:
            command: The slash command string

        Returns:
            (handled, expanded_text):
            - handled=True, expanded_text=None: command was handled, don't send to LLM
            - handled=False, expanded_text=None: unknown command, pass original to LLM
            - handled=False, expanded_text=str: file command, send expanded_text to LLM
        """
        result = await handle_slash_command(self, command)

        # Display any message from the command as a ghost message
        # (shows in chat but not part of conversation history)
        # Note: add_ghost_message internally calls request_render()
        if result.message and self.renderer:
            self.renderer.add_ghost_message(result.message)

        if result.handled:
            return True, None
        elif result.expanded_text:
            return False, result.expanded_text
        else:
            return False, None

    async def _tui_input_handler(self, prompt: str) -> str:
        """Async input handler for RunConfig.on_input.

        Uses a loop pattern instead of recursion for slash commands.
        This provides cleaner state management and avoids render timing issues.

        Args:
            prompt: Prompt string (not used in TUI, but required by signature)

        Returns:
            User input string (either direct input or expanded file command)
        """
        from .utils import strip_terminal_control_sequences

        if self.input_receive is None:
            raise RuntimeError("Input channel not initialized")

        # Loop until we get a message to send to LLM
        while True:
            user_input = await self._get_next_input()

            # Strip any terminal control sequences that may have leaked in
            # (e.g., bracketed paste sequences from vim mode / Ctrl+G)
            user_input = strip_terminal_control_sequences(user_input)

            # Handle slash commands
            if user_input.startswith("/"):
                handled, expanded_text = await self._handle_slash_command(user_input)
                if handled:
                    # Command was handled (e.g., /model, /thinking)
                    # Loop back to wait for next input
                    continue
                if expanded_text:
                    # File-based command expanded, use expanded text
                    user_input = expanded_text
                # else: unknown command, pass through to LLM as-is

            # We have a message to send to LLM
            break

        # Add user message to chat
        if self.renderer:
            self.renderer.add_user_message(user_input, is_first=self.is_first_user_message)
            self.is_first_user_message = False

        # Session persistence is handled by run_agent() via RunConfig.session_store
        return user_input

    async def _get_next_input(self) -> str:
        """Get the next user input, either from queue or by waiting.

        Returns:
            User input string
        """
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
            return user_input

        # No queued message - clear pending and wait for input
        self._pending_user_messages = []
        self.input_pending = True

        if self.input_component and self.tui:
            self.tui.set_focus(self.input_component)
            self.tui.request_render()

        user_input = await self.input_receive.receive()
        self.input_pending = False

        # Clear input component
        if self.input_component:
            self.input_component.set_text("")

        return user_input

    async def _handle_stream_event(self, event: StreamEvent) -> None:
        """Handle streaming event - render to TUI.

        Session persistence is handled by run_agent() via RunConfig.session_store.
        """
        if self.renderer:
            await self.renderer.handle_event(event)

    def _handle_sigint(self, signum: int, frame: FrameType | None) -> None:
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
                logger.debug(
                    f"  usage: in={completion.usage.input_tokens} out={completion.usage.output_tokens} cost={completion.usage.cost.total}"
                )
                total_input += completion.usage.input_tokens + completion.usage.cache_read_tokens
                total_output += completion.usage.output_tokens + completion.usage.reasoning_tokens
                total_cost += completion.usage.cost.total

        logger.debug(
            f"_update_token_counts: setting tokens {total_input}/{total_output} cost={total_cost}"
        )
        self.status_line.set_tokens(total_input, total_output, total_cost)

    def _update_env_status_info(self) -> None:
        """Update status line with environment info."""
        if self.status_line and self.environment:
            if hasattr(self.environment, "get_status_info"):
                env_info = self.environment.get_status_info()
                if env_info:
                    self.status_line.set_env_info(env_info)

    async def run(self) -> list[AgentState]:
        """Run interactive agent loop.

        Returns:
            List of agent states from the run
        """
        self._setup_tui()
        agent_states: list[AgentState] = []

        try:
            agent_states = await self._run_agent_loop()
        finally:
            await self._cleanup_and_print_session(agent_states)

        return agent_states

    async def _run_agent_loop(self) -> list[AgentState]:
        """Main agent loop with input handling."""
        self.input_send, self.input_receive = trio.open_memory_channel[str](10)

        if self.initial_prompt:
            self.input_send.send_nowait(self.initial_prompt)

        agent_states: list[AgentState] = []

        async with trio.open_nursery() as nursery:
            self.cancel_scope = nursery.cancel_scope

            nursery.start_soon(self._input_reading_loop)
            nursery.start_soon(self.tui.run_animation_loop)

            if self.input_component and self.tui:
                self.tui.set_focus(self.input_component)
                self.tui.request_render()

            first_message = await self._tui_input_handler("Enter your message: ")
            current_state = self._create_initial_state(first_message)

            # Main agent loop - handles interrupts and continues
            while True:
                self.agent_cancel_scope = trio.CancelScope()
                run_config = self._create_run_config()

                try:
                    with self.agent_cancel_scope:
                        agent_states = await run_agent(current_state, run_config)
                    # Update trajectory for slash commands (e.g., /slice)
                    if agent_states:
                        self._current_trajectory = agent_states[-1].actor.trajectory
                except Exception as e:
                    # Check for context too long error
                    from rollouts.providers.base import ContextTooLongError

                    if isinstance(e, ContextTooLongError):
                        current_state = await self._handle_context_too_long(e, current_state)
                        continue

                    # Check for OAuth expired error - prompt for re-login
                    from rollouts.frontends.tui.oauth import OAuthExpiredError

                    if isinstance(e, OAuthExpiredError):
                        current_state = await self._handle_oauth_expired(e, current_state)
                        continue

                    raise  # Re-raise other exceptions

                if agent_states and agent_states[-1].stop == StopReason.ABORTED:
                    if agent_states[-1].session_id:
                        self.session_id = agent_states[-1].session_id
                        set_active_session_id(self.session_id)

                    if not self.escape_pressed:
                        break  # Ctrl+C - exit the TUI

                    # Escape key - interrupt but continue
                    current_state = await self._handle_agent_interrupt(agent_states, current_state)
                elif agent_states and agent_states[-1].stop == StopReason.TASK_COMPLETED:
                    # Task completed - in interactive mode, show result and continue
                    current_state = await self._handle_task_completed(agent_states)
                else:
                    # Other stop reasons (MAX_TURNS, etc.) - update state and exit
                    self._update_final_state(agent_states)
                    break

                self.agent_cancel_scope = None

        if agent_states and agent_states[-1].session_id:
            self.session_id = agent_states[-1].session_id
            set_active_session_id(self.session_id)

        return agent_states

    async def _input_reading_loop(self) -> None:
        """Read terminal input and route to TUI."""
        while True:
            if self.terminal and self.terminal._running:
                input_data = self.terminal.read_input()
                if input_data:
                    # Check for Ctrl+C (ASCII 3) - exit TUI entirely
                    if len(input_data) > 0 and ord(input_data[0]) == 3:
                        if self.cancel_scope:
                            self.cancel_scope.cancel()
                        return

                    # Check for standalone Escape - interrupt current agent run
                    # But if there's a focused component that handles escape (like question selector),
                    # route escape to it instead. The Input component doesn't handle escape,
                    # so we skip routing to it to allow the interrupt to work.
                    if input_data == "\x1b":
                        if (
                            self.tui
                            and self.tui._focused_component is not None
                            and self.tui._focused_component is not self.input_component
                        ):
                            # Route to focused component (e.g., question selector review)
                            self.tui._handle_input(input_data)
                            continue

                        if self.agent_cancel_scope:
                            self.escape_pressed = True
                            self.agent_cancel_scope.cancel()
                            # Show visual feedback that interrupt was received
                            if self.tui:
                                self.tui.show_loader(
                                    "Interrupting...",
                                    spinner_color_fn=self.tui.theme.accent_fg,
                                    text_color_fn=self.tui.theme.accent_fg,
                                )
                                self.tui.request_render()
                        else:
                            # No active agent to interrupt - show message
                            if self.renderer:
                                self.renderer.add_system_message(
                                    "Nothing to interrupt (no active operation)"
                                )
                            if self.tui:
                                self.tui.request_render()
                        continue

                    if self.tui:
                        self.tui._handle_input(input_data)
            await trio.sleep(0.01)

    def _update_final_state(self, agent_states: list[AgentState]) -> None:
        """Update session_id and token counts from final agent state."""
        if not agent_states:
            return

        final_state = agent_states[-1]
        if final_state.session_id and final_state.session_id != self.session_id:
            self.session_id = final_state.session_id
            set_active_session_id(self.session_id)
            if self.status_line:
                self.status_line.set_session_id(self.session_id)
            self._update_env_status_info()

        if self.status_line and self.tui:
            self._update_token_counts(final_state)
            self.tui.request_render()

    def _handle_stop(self, state: AgentState) -> AgentState:
        """Handle stop condition. No max turns limit in interactive mode."""
        return state

    def _setup_tui(self) -> None:
        """Initialize terminal, TUI, and all UI components."""
        from .components.status_line import StatusLine
        from .theme import DARK_THEME, MINIMAL_THEME, ROUNDED_THEME

        if self.theme_name == "rounded":
            theme = ROUNDED_THEME
        elif self.theme_name == "minimal":
            theme = MINIMAL_THEME
        else:
            theme = DARK_THEME

        self.terminal = ProcessTerminal()
        self.tui = TUI(self.terminal, theme=theme, debug=self.debug, debug_layout=self.debug_layout)

        # Create renderer with environment for custom tool formatters
        self.renderer = AgentRenderer(
            self.tui, environment=self.environment, debug_layout=self.debug_layout
        )

        # Render history from initial trajectory (for resumed sessions)
        if self.initial_trajectory.messages:
            self.renderer.render_history(self.initial_trajectory.messages, skip_system=False)
            self.is_first_user_message = False
            if self.debug:
                self.renderer.debug_dump_chat()

        # Create loader container (for spinner during LLM calls)
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
        self.input_component.set_on_tab_complete(self._handle_tab_complete)
        self.input_component.set_on_change(self._handle_input_change)
        self.tui.add_child(self.input_component)

        # Create status line below input
        self.status_line = StatusLine(theme=self.tui.theme)
        self.status_line.set_session_id(self.session_id)
        model_meta = get_model(self.endpoint.provider, self.endpoint.model)  # type: ignore[arg-type]
        context_window = model_meta.context_window if model_meta else None
        self.status_line.set_model(
            f"{self.endpoint.provider}/{self.endpoint.model}", context_window=context_window
        )
        if self.environment and hasattr(self.environment, "get_status_info"):
            env_info = self.environment.get_status_info()
            if env_info:
                self.status_line.set_env_info(env_info)
        self.tui.add_child(self.status_line)

        # Add spacer after status line
        self.tui.add_child(Spacer(5, debug_label="after-status"))

        # Inject TUI question handler into AskUserQuestionEnvironment if present
        self._inject_question_handler()

        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._handle_sigint)

        # Start TUI
        self.tui.start()

    def _inject_question_handler(self) -> None:
        """No longer needed - ask_user_question is now intercepted via confirm_tool.

        Kept as no-op for backwards compatibility in case it's called elsewhere.
        """
        pass

    async def _handle_ask_user_question_tool(
        self, tc: ToolCall, state: AgentState
    ) -> tuple[AgentState, ToolConfirmResult]:
        """Handle ask_user_question tool via TUI confirm_tool interception.

        Instead of letting the environment execute this tool, we intercept it
        in confirm_tool and provide the result directly. This avoids the issue
        of environment deserialization losing injected handlers.
        """
        import json

        from .components.question_selector import MultiQuestionSelector

        questions = tc.args.get("questions", [])

        # Validate questions
        if not questions:
            return state, ToolConfirmResult(
                proceed=False,
                tool_result=ToolResult(
                    tool_call_id=tc.id,
                    is_error=True,
                    error="No questions provided",
                ),
            )

        if not self.tui:
            # Fallback - let environment handle it
            return state, ToolConfirmResult(proceed=True)

        # Hide the loader while asking questions
        self.tui.hide_loader()

        # Use the interactive selector
        selector = MultiQuestionSelector(
            questions=questions,
            tui=self.tui,
            theme=self.tui.theme,
        )

        answers = await selector.ask_all()

        # Return the result directly, bypassing environment execution
        return state, ToolConfirmResult(
            proceed=False,  # Don't proceed to env.exec_tool
            tool_result=ToolResult(
                tool_call_id=tc.id,
                is_error=False,
                content=json.dumps(answers),
            ),
        )

    def _create_initial_state(self, first_message: str) -> AgentState:
        """Create initial agent state with first user message."""
        initial_trajectory_with_user = Trajectory(
            messages=self.initial_trajectory.messages
            + [Message(role="user", content=first_message)]
        )

        return AgentState(
            actor=Actor(
                trajectory=initial_trajectory_with_user,
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
        """Create RunConfig with all handlers."""

        async def auto_confirm_tool(
            tc: ToolCall, state: AgentState, rcfg: RunConfig
        ) -> tuple[AgentState, ToolConfirmResult]:
            # Intercept ask_user_question to handle it via TUI
            if tc.name == "ask_user_question":
                return await self._handle_ask_user_question_tool(tc, state)
            return state, ToolConfirmResult(proceed=True)

        async def confirm_tool_tui(
            tc: ToolCall, state: AgentState, rcfg: RunConfig
        ) -> tuple[AgentState, ToolConfirmResult]:
            """Interactive tool confirmation in TUI."""
            # Intercept ask_user_question to handle it via TUI (no confirmation needed)
            if tc.name == "ask_user_question":
                return await self._handle_ask_user_question_tool(tc, state)

            if self.renderer:
                self.renderer.add_system_message(
                    f"⚠️  Tool: {tc.name}({tc.args})\n   [y] execute  [n] reject  [s] skip"
                )

            resp = await rcfg.on_input("Confirm tool? ")
            resp = resp.strip().lower()

            if resp in ("y", "yes", ""):
                return state, ToolConfirmResult(proceed=True)
            elif resp in ("n", "no"):
                feedback = await rcfg.on_input("Feedback for LLM: ")
                return state, ToolConfirmResult(
                    proceed=False,
                    tool_result=ToolResult(
                        tool_call_id=tc.id, is_error=True, error="Rejected by user"
                    ),
                    user_message=feedback.strip() if feedback.strip() else None,
                )
            else:
                return state, ToolConfirmResult(
                    proceed=False,
                    tool_result=ToolResult(
                        tool_call_id=tc.id, is_error=True, error="Skipped by user"
                    ),
                )

        async def handle_no_tool_interactive(state: AgentState, rcfg: RunConfig) -> AgentState:
            """Wait for user input when LLM responds without tool calls."""
            from dataclasses import replace as dc_replace

            # Update session_id from state (session created on first message)
            if state.session_id and state.session_id != self.session_id:
                self.session_id = state.session_id
                set_active_session_id(self.session_id)
                if self.status_line:
                    self.status_line.set_session_id(self.session_id)

            self._update_token_counts(state)
            if self.tui:
                self.tui.request_render()

            user_input = await rcfg.on_input("Enter your message: ")

            # Check if session was switched by /slice command
            # If so, rebuild state completely from self.initial_trajectory/endpoint
            if self._session_switched:
                self._session_switched = False  # Reset flag
                new_trajectory = Trajectory(
                    messages=list(self.initial_trajectory.messages)
                    + [Message(role="user", content=user_input)]
                )
                new_environment = self.environment
                new_tools = self.environment.get_tools() if self.environment else []
                return AgentState(
                    actor=Actor(
                        trajectory=new_trajectory,
                        endpoint=self.endpoint,
                        tools=new_tools,
                    ),
                    environment=new_environment,
                    session_id=self.session_id,
                )

            user_messages = [Message(role="user", content=user_input)]
            for pending_msg in self._pending_user_messages:
                if self.renderer:
                    self.renderer.add_user_message(pending_msg, is_first=False)
                user_messages.append(Message(role="user", content=pending_msg))
            self._pending_user_messages = []

            new_trajectory = Trajectory(messages=state.actor.trajectory.messages + user_messages)

            # Check if environment was changed by /env command
            new_environment = state.environment
            new_tools = state.actor.tools
            if self._environment_changed and self.environment:
                new_environment = self.environment
                new_tools = self.environment.get_tools()
                self._environment_changed = False  # Reset flag

            # Use self.endpoint to pick up any /model changes
            # Use self.session_id to pick up session changes from /env
            new_actor = dc_replace(
                state.actor,
                trajectory=new_trajectory,
                endpoint=self.endpoint,
                tools=new_tools,
            )
            result = dc_replace(
                state,
                actor=new_actor,
                environment=new_environment,
                session_id=self.session_id,
            )
            return result

        confirm_handler = confirm_tool_tui if self.confirm_tools else auto_confirm_tool

        return RunConfig(
            on_chunk=self._handle_stream_event,
            on_input=self._tui_input_handler,
            confirm_tool=confirm_handler,
            handle_stop=self._handle_stop,
            handle_no_tool=handle_no_tool_interactive,
            session_store=self.session_store,
            cancel_scope=self.agent_cancel_scope,
        )

    async def _handle_agent_interrupt(
        self, agent_states: list[AgentState], current_state: AgentState
    ) -> AgentState:
        """Handle agent interruption (Escape key). Returns new state to continue."""
        self.escape_pressed = False
        if self.tui:
            self.tui.hide_loader()

        partial_response = None
        if self.renderer:
            partial_response = self.renderer.get_partial_response()
            self.renderer.finalize_partial_response()
            self.renderer.add_system_message("Interrupted")

        latest_state = agent_states[-1] if agent_states else current_state

        if latest_state.session_id and latest_state.session_id != self.session_id:
            self.session_id = latest_state.session_id
            set_active_session_id(self.session_id)
            if self.status_line:
                self.status_line.set_session_id(self.session_id)
            self._update_env_status_info()

        if self.status_line and self.tui:
            self._update_token_counts(latest_state)
            self.tui.request_render()

        # Load messages from session store - this includes any assistant messages
        # that were persisted during the interrupted turn but aren't in the in-memory state
        new_messages = []
        if self.session_store and latest_state.session_id:
            try:
                session, _ = await self.session_store.get(latest_state.session_id)
                if session and session.messages:
                    new_messages = list(session.messages)
            except Exception:
                pass

        # Fall back to in-memory trajectory if session store load failed
        if not new_messages:
            new_messages = list(latest_state.actor.trajectory.messages)

        # Add partial response if we interrupted during streaming
        if partial_response:
            new_messages.append(
                Message(role="assistant", content=partial_response + "\n\n[interrupted]")
            )

        try:
            from rollouts.feedback import run_exit_survey

            await run_exit_survey(
                latest_state, self.endpoint, "yield", session_id=self.session_id, skip_check=True
            )
        except Exception:
            pass

        user_input = await self._tui_input_handler("Enter your message: ")

        from dataclasses import replace as dc_replace

        # Check if session was switched by /slice command
        # If so, rebuild state completely from self.initial_trajectory/endpoint
        if self._session_switched:
            self._session_switched = False  # Reset flag
            new_trajectory = Trajectory(
                messages=list(self.initial_trajectory.messages)
                + [Message(role="user", content=user_input)]
            )
            new_environment = self.environment
            new_tools = self.environment.get_tools() if self.environment else []
            return AgentState(
                actor=Actor(
                    trajectory=new_trajectory,
                    endpoint=self.endpoint,
                    tools=new_tools,
                ),
                environment=new_environment,
                session_id=self.session_id,
            )

        new_messages.append(Message(role="user", content=user_input))

        new_trajectory = Trajectory(messages=new_messages)

        # Check if environment was changed by /env command
        new_environment = latest_state.environment
        new_tools = latest_state.actor.tools
        if self._environment_changed and self.environment:
            new_environment = self.environment
            new_tools = self.environment.get_tools()
            self._environment_changed = False  # Reset flag

        return dc_replace(
            latest_state,
            actor=dc_replace(
                latest_state.actor,
                trajectory=new_trajectory,
                tools=new_tools,
            ),
            environment=new_environment,
            stop=None,
        )

    async def _handle_task_completed(self, agent_states: list[AgentState]) -> AgentState:
        """Handle TASK_COMPLETED in interactive mode - show result and continue.

        Unlike batch mode where TASK_COMPLETED exits, interactive mode should
        display the result and wait for more user input.
        """
        if self.tui:
            self.tui.hide_loader()

        latest_state = agent_states[-1]

        # Update session tracking
        if latest_state.session_id and latest_state.session_id != self.session_id:
            self.session_id = latest_state.session_id
            set_active_session_id(self.session_id)
            if self.status_line:
                self.status_line.set_session_id(self.session_id)
            self._update_env_status_info()

        if self.status_line and self.tui:
            self._update_token_counts(latest_state)
            self.tui.request_render()

        # Check if environment has a final_answer to display
        if latest_state.environment and hasattr(latest_state.environment, "_final_answer"):
            final_answer = getattr(latest_state.environment, "_final_answer", None)
            if final_answer and self.renderer:
                self.renderer.add_final_answer(final_answer)
                if self.tui:
                    self.tui.request_render()

        # Wait for next user input
        user_input = await self._tui_input_handler("Enter your message: ")

        from dataclasses import replace as dc_replace

        # Check if session was switched by /slice command
        # If so, rebuild state completely from self.initial_trajectory/endpoint
        if self._session_switched:
            self._session_switched = False  # Reset flag
            new_trajectory = Trajectory(
                messages=list(self.initial_trajectory.messages)
                + [Message(role="user", content=user_input)]
            )
            new_environment = self.environment
            new_tools = self.environment.get_tools() if self.environment else []
            return AgentState(
                actor=Actor(
                    trajectory=new_trajectory,
                    endpoint=self.endpoint,
                    tools=new_tools,
                ),
                environment=new_environment,
                session_id=self.session_id,
            )

        # Build new trajectory with user message
        new_messages = list(latest_state.actor.trajectory.messages)
        new_messages.append(Message(role="user", content=user_input))

        new_trajectory = Trajectory(messages=new_messages)

        # Check if environment was changed by /env command
        new_environment = latest_state.environment
        new_tools = latest_state.actor.tools
        if self._environment_changed and self.environment:
            new_environment = self.environment
            new_tools = self.environment.get_tools()
            self._environment_changed = False  # Reset flag

        return dc_replace(
            latest_state,
            actor=dc_replace(
                latest_state.actor,
                trajectory=new_trajectory,
                tools=new_tools,
            ),
            environment=new_environment,
            stop=None,  # Clear stop so agent continues
        )

    async def _handle_context_too_long(
        self, error: Exception, current_state: AgentState
    ) -> AgentState:
        """Handle context too long error gracefully.

        Shows a friendly error message and waits for user input.
        """
        from rollouts.providers.base import ContextTooLongError

        if self.tui:
            self.tui.hide_loader()

        # Display error message
        error_msg = "⚠️  Context too long"
        if isinstance(error, ContextTooLongError) and error.current_tokens and error.max_tokens:
            error_msg += f" ({error.current_tokens:,} tokens, max {error.max_tokens:,})"

        if self.renderer:
            self.renderer.add_user_message(
                f"{error_msg}\n\n"
                "The conversation has grown too long for the model's context window.\n"
                "Please start a new conversation."
            )
            if self.tui:
                self.tui.request_render()

        # Wait for user input
        user_input = await self._tui_input_handler("Enter your message: ")

        from dataclasses import replace as dc_replace

        # Check if session was switched by /slice command
        # If so, rebuild state completely from self.initial_trajectory/endpoint
        if self._session_switched:
            self._session_switched = False  # Reset flag
            new_trajectory = Trajectory(
                messages=list(self.initial_trajectory.messages)
                + [Message(role="user", content=user_input)]
            )
            new_environment = self.environment
            new_tools = self.environment.get_tools() if self.environment else []
            return AgentState(
                actor=Actor(
                    trajectory=new_trajectory,
                    endpoint=self.endpoint,
                    tools=new_tools,
                ),
                environment=new_environment,
                session_id=self.session_id,
            )

        # Check if environment was changed by /env command
        new_environment = current_state.environment
        new_tools = current_state.actor.tools
        if self._environment_changed and self.environment:
            new_environment = self.environment
            new_tools = self.environment.get_tools()
            self._environment_changed = False  # Reset flag

        # Start fresh with user's new message
        new_trajectory = Trajectory(messages=[Message(role="user", content=user_input)])
        return dc_replace(
            current_state,
            actor=dc_replace(
                current_state.actor,
                trajectory=new_trajectory,
                tools=new_tools,
            ),
            environment=new_environment,
            stop=None,
        )

    async def _handle_oauth_expired(
        self, error: Exception, current_state: AgentState
    ) -> AgentState:
        """Handle OAuth token expiration gracefully.

        Shows an error message and prompts user to re-login via /login command.
        After login, retries the last user message.
        """
        if self.tui:
            self.tui.hide_loader()

        # Display error message with login instructions
        if self.renderer:
            self.renderer.add_system_message(
                "🔐 OAuth token expired and refresh failed.\n"
                "   Run /login to re-authenticate, then your message will be retried."
            )
            if self.tui:
                self.tui.request_render()

        # Wait for user to run /login (or any other input)
        # The _tui_input_handler will process /login and loop back
        user_input = await self._tui_input_handler("Run /login to continue: ")

        # Return current state - the user's original message is still in trajectory
        # so when they re-authenticate and hit enter, it will retry
        return current_state

    async def _cleanup_and_print_session(self, agent_states: list[AgentState]) -> None:
        """Stop TUI, run exit survey, and print session info."""
        if self.tui:
            self.tui.stop()
        if self.terminal:
            self.terminal.stop()

        sys.stdout.flush()

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
            except Exception:
                pass

        if self.session_id:
            print(f"\nSession: {self.session_id}")
            print(f"Resume with: --session {self.session_id}")
            # Clear active session so atexit handler doesn't double-print
            set_active_session_id(None)

            from rollouts.environments.git_worktree import GitWorktreeEnvironment

            if (
                isinstance(self.environment, GitWorktreeEnvironment)
                and self.environment._worktree_path
            ):
                self._print_git_worktree_info()

    def _print_git_worktree_info(self) -> None:
        """Print git worktree information after session ends."""
        import subprocess

        from rollouts.environments.git_worktree import GitWorktreeEnvironment

        if not isinstance(self.environment, GitWorktreeEnvironment):
            return
        env = self.environment
        worktree = env._worktree_path

        try:
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=str(worktree),
                capture_output=True,
                text=True,
                timeout=5,
            )
            commit_count = int(result.stdout.strip()) if result.returncode == 0 else 0
        except Exception:
            commit_count = env._commit_count

        print(f"\nChanges in: {worktree}")
        if commit_count > 1:
            print(f"  {commit_count - 1} file operations committed")
        print(f"\nTo view:  cd {worktree} && git log --oneline")
        print(f"To diff:  diff -r {worktree} . --exclude=.rollouts")
        print(f"To apply: cp -r {worktree}/* .")


async def run_interactive_agent(
    initial_trajectory: Trajectory,
    endpoint: Endpoint,
    environment: Environment | None = None,
    session_store: SessionStore | None = None,
    session_id: str | None = None,
    theme_name: str = "dark",
    debug: bool = False,
    debug_layout: bool = False,
    parent_session_id: str | None = None,
    branch_point: int | None = None,
    confirm_tools: bool = False,
    initial_prompt: str | None = None,
) -> list[AgentState]:
    """Run an interactive agent with TUI.

    Args:
        initial_trajectory: Initial conversation trajectory
        endpoint: LLM endpoint configuration
        environment: Optional environment for tool execution
        session_store: Optional session store for persistence
        session_id: Optional session ID (required if session_store is set)
        theme_name: Theme name (dark or rounded)
        debug: Enable debug logging and chat state dumps
        debug_layout: Show component boundaries and spacing
        parent_session_id: Parent session ID when forking
        branch_point: Message index where forking from parent
        confirm_tools: Require confirmation before executing tools
        initial_prompt: Optional initial prompt to send immediately

    Returns:
        List of agent states from the run
    """
    runner = InteractiveAgentRunner(
        initial_trajectory=initial_trajectory,
        endpoint=endpoint,
        environment=environment,
        session_store=session_store,
        session_id=session_id,
        theme_name=theme_name,
        debug=debug,
        debug_layout=debug_layout,
        parent_session_id=parent_session_id,
        branch_point=branch_point,
        confirm_tools=confirm_tools,
        initial_prompt=initial_prompt,
    )
    return await runner.run()
