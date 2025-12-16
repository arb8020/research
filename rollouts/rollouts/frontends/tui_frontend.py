"""TUIFrontend - wrapper around existing Python TUI.

This adapts the existing terminal UI (ProcessTerminal, TUI, AgentRenderer)
to the Frontend protocol, enabling it to be used interchangeably with
other frontends like NoneFrontend or TextualFrontend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import trio

if TYPE_CHECKING:
    from rollouts.dtypes import Environment, StreamEvent, ToolCall


class TUIFrontend:
    """Wrapper around existing Python TUI implementation.

    Adapts ProcessTerminal + TUI + AgentRenderer to the Frontend protocol.

    Example usage:
        frontend = TUIFrontend(theme="dark", environment=env)
        await runner.run(frontend=frontend)
    """

    def __init__(
        self,
        theme: str = "dark",
        environment: Environment | None = None,
        debug: bool = False,
        debug_layout: bool = False,
    ):
        """Initialize TUIFrontend.

        Args:
            theme: Theme name (dark, rounded, minimal)
            environment: Optional environment for custom tool formatters
            debug: Enable debug logging
            debug_layout: Show component boundaries
        """
        self.theme_name = theme
        self.environment = environment
        self.debug = debug
        self.debug_layout = debug_layout

        # Components (initialized in start())
        self._terminal: Any | None = None
        self._tui: Any | None = None
        self._renderer: Any | None = None
        self._input_component: Any | None = None
        self._loader_container: Any | None = None
        self._status_line: Any | None = None

        # Input coordination
        self._input_send: trio.MemorySendChannel[str] | None = None
        self._input_receive: trio.MemoryReceiveChannel[str] | None = None
        self._input_pending: bool = False
        self._is_first_user_message = True

    async def start(self) -> None:
        """Initialize TUI components and enter raw mode."""
        from .tui.agent_renderer import AgentRenderer
        from .tui.components.input import Input
        from .tui.components.loader_container import LoaderContainer
        from .tui.components.spacer import Spacer
        from .tui.components.status_line import StatusLine
        from .tui.terminal import ProcessTerminal
        from .tui.theme import DARK_THEME, MINIMAL_THEME, ROUNDED_THEME
        from .tui.tui import TUI

        # Select theme
        if self.theme_name == "rounded":
            theme = ROUNDED_THEME
        elif self.theme_name == "minimal":
            theme = MINIMAL_THEME
        else:
            theme = DARK_THEME

        # Create terminal and TUI
        self._terminal = ProcessTerminal()
        self._tui = TUI(
            self._terminal,
            theme=theme,
            debug=self.debug,
            debug_layout=self.debug_layout,
        )

        # Create renderer with environment for custom tool formatters
        self._renderer = AgentRenderer(
            self._tui,
            environment=self.environment,
            debug_layout=self.debug_layout,
        )

        # Create loader container
        self._loader_container = LoaderContainer(
            spinner_color_fn=self._tui.theme.accent_fg,
            text_color_fn=self._tui.theme.muted_fg,
        )
        self._tui.set_loader_container(self._loader_container)
        self._tui.add_child(self._loader_container)

        # Spacer before input
        self._tui.add_child(Spacer(1, debug_label="before-input"))

        # Create input component
        self._input_component = Input(theme=self._tui.theme)
        self._input_component.set_on_submit(self._handle_input_submit)
        self._tui.add_child(self._input_component)

        # Create status line
        self._status_line = StatusLine(theme=self._tui.theme)
        self._tui.add_child(self._status_line)

        # Spacer after status
        self._tui.add_child(Spacer(5, debug_label="after-status"))

        # Create input channel
        self._input_send, self._input_receive = trio.open_memory_channel[str](10)

        # Start TUI
        self._tui.start()

    async def stop(self) -> None:
        """Stop TUI and restore terminal."""
        import sys

        if self._tui:
            self._tui.stop()
        if self._terminal:
            self._terminal.stop()

        # Ensure output buffer is clean
        sys.stdout.flush()

    async def handle_event(self, event: StreamEvent) -> None:
        """Route event to AgentRenderer.

        Args:
            event: StreamEvent to handle
        """
        if self._renderer:
            await self._renderer.handle_event(event)

    async def get_input(self, prompt: str = "") -> str:
        """Get user input via TUI input component.

        Args:
            prompt: Ignored (TUI has its own prompt)

        Returns:
            User's input string
        """
        if self._input_receive is None:
            raise RuntimeError("Input channel not initialized")

        # Try to get queued message first
        try:
            msg = self._input_receive.receive_nowait()
            if self._input_component:
                self._input_component.pop_queued_message()
            return msg
        except trio.WouldBlock:
            pass

        # No queued message, show input and wait
        self._input_pending = True
        if self._input_component and self._tui:
            self._tui.set_focus(self._input_component)
            self._tui.request_render()

        user_input = await self._input_receive.receive()
        self._input_pending = False

        # Clear input component
        if self._input_component:
            self._input_component.set_text("")

        # Add user message to chat
        if self._renderer:
            self._renderer.add_user_message(user_input, is_first=self._is_first_user_message)
            self._is_first_user_message = False

        return user_input

    async def confirm_tool(self, tool_call: ToolCall) -> bool:
        """Confirm tool execution via TUI.

        Args:
            tool_call: Tool call to confirm

        Returns:
            True if approved, False if rejected
        """
        if self._renderer:
            args_str = ", ".join(f"{k}={v!r}" for k, v in tool_call.args.items())
            if len(args_str) > 80:
                args_str = args_str[:77] + "..."
            self._renderer.add_system_message(
                f"⚠️  Tool: {tool_call.name}({args_str})\n   [y] execute  [n] reject  [s] skip"
            )

        response = await self.get_input("Confirm tool? ")
        return response.strip().lower() in ("y", "yes", "")

    def show_loader(self, text: str) -> None:
        """Show loading indicator.

        Args:
            text: Loading status text
        """
        if self._tui:
            self._tui.show_loader(
                text,
                spinner_color_fn=self._tui.theme.accent_fg,
                text_color_fn=self._tui.theme.muted_fg,
            )

    def hide_loader(self) -> None:
        """Hide loading indicator."""
        if self._tui:
            self._tui.hide_loader()

    def set_status(
        self,
        *,
        model: str | None = None,
        session_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost: float | None = None,
        env_info: dict[str, str] | None = None,
    ) -> None:
        """Update status line information.

        Args:
            model: Model name
            session_id: Session ID
            input_tokens: Cumulative input tokens
            output_tokens: Cumulative output tokens
            cost: Cumulative cost
            env_info: Environment info
        """
        if not self._status_line:
            return

        if model is not None:
            self._status_line.set_model(model)
        if session_id is not None:
            self._status_line.set_session_id(session_id)
        if input_tokens is not None and output_tokens is not None:
            cost_val = cost if cost is not None else 0.0
            self._status_line.set_tokens(input_tokens, output_tokens, cost_val)
        if env_info is not None:
            self._status_line.set_env_info(env_info)

    def _handle_input_submit(self, text: str) -> None:
        """Handle input submission from TUI component.

        Args:
            text: Submitted text
        """
        if text.strip() and self._input_send:
            try:
                self._input_send.send_nowait(text.strip())
                # Add to visual queue if not waiting for input
                if not self._input_pending and self._input_component:
                    self._input_component.add_queued_message(text.strip())
                    if self._tui:
                        self._tui.request_render()
            except trio.WouldBlock:
                pass  # Buffer full

    def render_history(self, messages: list) -> None:
        """Render historical messages from resumed session.

        Args:
            messages: List of Message objects
        """
        if self._renderer:
            self._renderer.render_history(messages, skip_system=False)
            self._is_first_user_message = False

    def add_system_message(self, text: str) -> None:
        """Add a system message to the chat.

        Args:
            text: Message text
        """
        if self._renderer:
            self._renderer.add_system_message(text)

    def get_partial_response(self) -> str | None:
        """Get any partial assistant response being streamed.

        Returns:
            Partial text or None
        """
        if self._renderer:
            return self._renderer.get_partial_response()
        return None

    def finalize_partial_response(self) -> None:
        """Mark any partial response as complete."""
        if self._renderer:
            self._renderer.finalize_partial_response()

    async def run_input_loop(self, nursery: trio.Nursery) -> None:
        """Run terminal input reading loop.

        Must be called in a nursery to handle keyboard input.

        Args:
            nursery: Trio nursery to spawn tasks in
        """

        async def input_reading_loop():
            while True:
                if self._terminal and self._terminal._running:
                    input_data = self._terminal.read_input()
                    if input_data and self._tui:
                        self._tui._handle_input(input_data)
                await trio.sleep(0.01)

        nursery.start_soon(input_reading_loop)

        # Also start animation loop
        if self._tui:
            nursery.start_soon(self._tui.run_animation_loop)

    def request_render(self) -> None:
        """Request a TUI render."""
        if self._tui:
            self._tui.request_render()
