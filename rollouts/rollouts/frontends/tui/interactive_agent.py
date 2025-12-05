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
from .components.loader_container import LoaderContainer
from .sessions import Session, append_message, append_config_change


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
        debug: bool = False,
        debug_layout: bool = False,
    ) -> None:
        """Initialize interactive agent runner.

        Args:
            initial_trajectory: Initial conversation trajectory
            endpoint: LLM endpoint configuration
            environment: Optional environment for tool execution
            max_turns: Maximum number of turns
            session: Optional session for persistence
            theme_name: Theme name (dark or rounded)
            debug: Enable debug logging and chat state dumps
            debug_layout: Show component boundaries and spacing
        """
        self.initial_trajectory = initial_trajectory
        self.endpoint = endpoint
        self.theme_name = theme_name
        self.environment = environment
        self.max_turns = max_turns
        self.session = session
        self.debug = debug
        self.debug_layout = debug_layout

        # TUI components
        self.terminal: Optional[ProcessTerminal] = None
        self.tui: Optional[TUI] = None
        self.renderer: Optional[AgentRenderer] = None
        self.input_component: Optional[Input] = None
        self.loader_container: Optional[LoaderContainer] = None

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

    async def _handle_slash_command(self, command: str) -> bool:
        """Handle slash commands like /model, /thinking, /tools.

        Args:
            command: The slash command string

        Returns:
            True if command was handled, False if it should be passed to LLM
        """
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "/model":
            if not args:
                if self.renderer:
                    self.renderer.add_system_message(f"Current model: {self.endpoint.provider}/{self.endpoint.model}\nUsage: /model <provider/model> or /model list")
                return True

            # Handle /model list
            if args.lower() == "list":
                try:
                    from rollouts.models import MODELS
                    lines = ["Available models:"]
                    for provider, models in MODELS.items():
                        lines.append(f"\n{provider}:")
                        for model_id in models.keys():
                            lines.append(f"  - {provider}/{model_id}")
                    if self.renderer:
                        self.renderer.add_system_message("\n".join(lines))
                except Exception as e:
                    if self.renderer:
                        self.renderer.add_system_message(f"Error listing models: {e}")
                return True

            # Parse and create new endpoint
            try:
                from .cli import parse_model_string, create_endpoint
                provider, model = parse_model_string(args)

                # Preserve thinking config from current endpoint
                thinking_str = "enabled" if self.endpoint.thinking else "disabled"
                new_endpoint = create_endpoint(args, thinking=thinking_str)

                # Update endpoint
                self.endpoint = new_endpoint

                # Log config change to session
                if self.session:
                    import json
                    env_type = type(self.environment).__name__ if self.environment else "none"
                    append_config_change(
                        self.session,
                        json.loads(new_endpoint.to_json()),
                        env_type
                    )

                if self.renderer:
                    self.renderer.add_system_message(f"Switched to model: {provider}/{model}")
            except Exception as e:
                if self.renderer:
                    self.renderer.add_system_message(f"Error switching model: {e}")
            return True

        elif cmd == "/thinking":
            if not args:
                current = "enabled" if self.endpoint.thinking else "disabled"
                if self.renderer:
                    self.renderer.add_system_message(f"Current thinking: {current}\nUsage: /thinking <enabled|disabled>")
                return True

            # Toggle thinking
            try:
                from .cli import create_endpoint
                args_lower = args.lower()
                if args_lower not in ["enabled", "disabled"]:
                    if self.renderer:
                        self.renderer.add_system_message("Invalid thinking level. Use: enabled or disabled")
                    return True

                # Create new endpoint with updated thinking
                model_str = f"{self.endpoint.provider}/{self.endpoint.model}"
                new_endpoint = create_endpoint(model_str, thinking=args_lower)

                # Update endpoint
                self.endpoint = new_endpoint

                # Log config change to session
                if self.session:
                    import json
                    env_type = type(self.environment).__name__ if self.environment else "none"
                    append_config_change(
                        self.session,
                        json.loads(new_endpoint.to_json()),
                        env_type
                    )

                if self.renderer:
                    self.renderer.add_system_message(f"Thinking set to: {args_lower}")
            except Exception as e:
                if self.renderer:
                    self.renderer.add_system_message(f"Error changing thinking: {e}")
            return True

        elif cmd == "/tools":
            if not args:
                current_name = self.environment.get_name() if hasattr(self.environment, 'get_name') else "unknown"
                if self.renderer:
                    from ...environments.tool_presets import get_preset_names
                    available = ", ".join(get_preset_names())
                    self.renderer.add_system_message(f"Current preset: {current_name}\nAvailable: {available}\nUsage: /tools <preset>")
                return True

            # Switch tool preset
            try:
                from ...environments.tool_presets import create_preset
                from pathlib import Path

                # Get working dir from current environment
                working_dir = self.environment.working_dir if hasattr(self.environment, 'working_dir') else Path.cwd()

                # Create new environment with preset
                new_environment = create_preset(args, working_dir)

                # Update environment
                self.environment = new_environment

                # Update renderer's environment reference
                if self.renderer:
                    self.renderer.environment = new_environment

                # Log config change to session
                if self.session:
                    import json
                    env_type = new_environment.get_name()
                    append_config_change(
                        self.session,
                        json.loads(self.endpoint.to_json()),
                        env_type
                    )

                if self.renderer:
                    desc = new_environment.get_description() if hasattr(new_environment, 'get_description') else args
                    self.renderer.add_system_message(f"Switched to preset: {args} ({desc})")
            except Exception as e:
                if self.renderer:
                    self.renderer.add_system_message(f"Error switching tools: {e}")
            return True

        elif cmd == "/help":
            help_text = """Available commands:
  /model [name]     - Switch model (e.g., /model anthropic/claude-sonnet-4)
  /model list       - List all available models
  /thinking [level] - Change thinking level (enabled/disabled)
  /tools [preset]   - Switch tool preset (full/readonly/no-write)
  /help             - Show this help message

Examples:
  /model list
  /model anthropic/claude-3-5-haiku-20241022
  /model openai/gpt-4
  /thinking disabled
  /tools readonly"""
            if self.renderer:
                self.renderer.add_system_message(help_text)
            return True

        # Unknown command - pass to LLM
        return False

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
        self.tui.add_child(self.input_component)

        # Add spacer after input box to keep it 6 lines from bottom
        self.tui.add_child(Spacer(6, debug_label="after-input"))

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
    debug: bool = False,
    debug_layout: bool = False,
) -> list[AgentState]:
    """Run an interactive agent with TUI.

    Args:
        initial_trajectory: Initial conversation trajectory
        endpoint: LLM endpoint configuration
        environment: Optional environment for tool execution
        max_turns: Maximum number of turns
        session: Optional session for persistence
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
        session=session,
        theme_name=theme_name,
        debug=debug,
        debug_layout=debug_layout,
    )
    return await runner.run()

