"""TUIFrontend - wrapper around existing Python TUI.

This adapts the existing terminal UI (ProcessTerminal, TUI, AgentRenderer)
to the Frontend protocol, enabling it to be used interchangeably with
other frontends like NoneFrontend or TextualFrontend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import trio

from pytui.input import KeyPress

from .protocol import InputResult

if TYPE_CHECKING:
    from ..dtypes import Environment, StreamEvent, ToolCall


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
        driver: str | None = None,
    ) -> None:
        """Initialize TUIFrontend.

        Args:
            theme: Theme name (dark, rounded, minimal)
            environment: Optional environment for custom tool formatters
            debug: Enable debug logging
            debug_layout: Show component boundaries
            driver: Driver name (sdk, claude, codex, cursor)
        """
        self.theme_name = theme
        self.environment = environment
        self.debug = debug
        self.debug_layout = debug_layout
        self.driver = driver

        # Components (initialized in start())
        self._terminal: Any | None = None
        self._tui: Any | None = None
        self._renderer: Any | None = None
        self._input_component: Any | None = None
        self._loader_container: Any | None = None
        self._pending_messages: Any | None = None
        self._status_line: Any | None = None

        # Input coordination - channel carries InputResult (text, interrupt, exit)
        self._input_send: trio.MemorySendChannel[InputResult] | None = None
        self._input_receive: trio.MemoryReceiveChannel[InputResult] | None = None
        self._input_pending: bool = False
        self._is_first_user_message = True

        # Double-tap Ctrl+C tracking
        self._ctrl_c_pending: float | None = None  # Timestamp of first Ctrl+C

        # Interrupt state - set when Escape is pressed during agent execution
        self._interrupt_pending: bool = False

    async def start(self) -> None:
        """Initialize TUI components and enter raw mode."""
        from .tui.agent_renderer import AgentRenderer
        from .tui.components.input import Input
        from .tui.components.loader_container import LoaderContainer
        from .tui.components.pending_messages import PendingMessages
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

        # Spacer before input
        self._tui.add_child(Spacer(1, debug_label="before-input"))

        # Create input component
        self._input_component = Input(theme=self._tui.theme)
        self._input_component.set_on_submit(self._handle_input_submit)
        self._input_component.set_on_editor(self._handle_open_editor)

        # Pending messages (queued while agent is busy)
        self._pending_messages = PendingMessages(
            theme=self._tui.theme,
            get_blocked_reason=lambda: self._tui.get_loader_text()
            or (
                f"Waiting on {type(self._tui._focused_component).__name__}"
                if self._tui
                and self._tui._focused_component
                and self._tui._focused_component is not self._input_component
                else None
            ),
            restore_hint="↑ restore to edit",
        )
        self._tui.add_child(self._pending_messages)
        self._tui.add_child(self._input_component)

        # Create loader container (keep near the input so it's visible in the viewport)
        self._loader_container = LoaderContainer(
            spinner_color_fn=self._tui.theme.accent_fg,
            text_color_fn=self._tui.theme.muted_fg,
        )
        self._tui.set_loader_container(self._loader_container)
        self._tui.add_child(self._loader_container)

        # Create status line
        self._status_line = StatusLine(theme=self._tui.theme)
        if self.driver:
            self._status_line.set_driver(self.driver)
        self._tui.add_child(self._status_line)

        # Connect status_line to renderer for tok/s updates
        self._renderer.set_status_line(self._status_line)

        # Spacer after status
        self._tui.add_child(Spacer(1, debug_label="after-status"))

        # Create input channel
        self._input_send, self._input_receive = trio.open_memory_channel[InputResult](10)

        # Start TUI
        self._tui.start()

    async def stop(self) -> None:
        """Stop TUI and restore terminal."""
        import subprocess
        import sys

        if self._tui:
            self._tui.stop()
        if self._terminal:
            self._terminal.stop()

        # Ensure output buffer is clean
        sys.stdout.flush()
        sys.stderr.flush()

        # Run stty sane to ensure terminal is fully restored
        # (handles edge cases where termios restoration is incomplete)
        try:
            subprocess.run(["stty", "sane"], stdin=open("/dev/tty"), check=False)
        except Exception:
            pass

    async def handle_event(self, event: StreamEvent) -> None:
        """Route event to AgentRenderer.

        Args:
            event: StreamEvent to handle
        """
        if self._renderer:
            await self._renderer.handle_event(event)

    async def get_input(self, prompt: str = "") -> InputResult:
        """Get user input via TUI input component.

        Returns InputResult:
        - UserMessage: Regular text to send to LLM
        - SlashCommand: Parsed command for runner to execute
        - InputExit: User wants to quit
        - InputInterrupt: User pressed Escape to interrupt

        Args:
            prompt: Ignored (TUI has its own prompt)

        Returns:
            InputResult indicating what the user wants to do
        """
        from .protocol import InputExit, InputInterrupt, SlashCommand, UserMessage

        if self._input_receive is None:
            raise RuntimeError("Input channel not initialized")

        # Try to get queued message first
        try:
            result = self._input_receive.receive_nowait()
            if self._pending_messages and isinstance(result, UserMessage):
                self._pending_messages.pop_left()
            if self._tui:
                self._tui.request_render()
        except trio.WouldBlock:
            # No queued message, show input and wait
            self._input_pending = True
            if self._tui:
                # Hide loader - we're now waiting for user input
                self._tui.hide_loader()
                if self._input_component:
                    self._tui.set_focus(self._input_component)
                self._tui.request_render()

            result = await self._input_receive.receive()
            self._input_pending = False

        # Clear input component
        if self._input_component:
            self._input_component.set_text("")

        # If we got InputInterrupt or InputExit, return directly
        if isinstance(result, (InputInterrupt, InputExit)):
            return result

        # Extract text from UserMessage
        assert isinstance(result, UserMessage), f"Unexpected result type: {type(result)}"
        user_input = result.text

        # Check for exit commands
        if user_input.strip().lower() in ("exit", "quit", "q"):
            return InputExit()

        # Parse slash commands - frontend parses, runner executes
        if user_input.startswith("/"):
            space_idx = user_input.find(" ")
            if space_idx == -1:
                name = user_input[1:]
                args = ""
            else:
                name = user_input[1:space_idx]
                args = user_input[space_idx + 1 :].strip()
            return SlashCommand(name=name, args=args)

        # Regular message - add to chat display
        if self._renderer:
            self._renderer.add_user_message(user_input, is_first=self._is_first_user_message)
            self._is_first_user_message = False

        return UserMessage(text=user_input)

    def _restore_queued_messages_to_input(self) -> None:
        """Move all queued messages back into the editor for editing."""
        from .protocol import UserMessage

        if not self._input_receive or not self._input_component:
            return

        drained: list[str] = []
        while True:
            try:
                result = self._input_receive.receive_nowait()
                # Only restore UserMessage text, skip interrupts/exits
                if isinstance(result, UserMessage):
                    drained.append(result.text)
            except trio.WouldBlock:
                break

        if not drained:
            return

        current = self._input_component.get_text()
        queued_text = "\n\n".join(drained)
        combined = "\n\n".join([t for t in (queued_text, current) if t.strip()])
        self._input_component.set_text(combined)

        if self._pending_messages:
            self._pending_messages.clear()

        if self._tui:
            self._tui.set_focus(self._input_component)
            self._tui.request_render()

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

        from .protocol import UserMessage

        response = await self.get_input("Confirm tool? ")
        # Handle InputResult: only UserMessage contains confirmable text
        if isinstance(response, UserMessage):
            return response.text.strip().lower() in ("y", "yes", "")
        # InputExit/InputInterrupt -> reject the tool
        return False

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
        from .protocol import UserMessage

        if text.strip() and self._input_send:
            try:
                self._input_send.send_nowait(UserMessage(text=text.strip()))
                # Add to visual queue if not waiting for input
                if not self._input_pending and self._pending_messages:
                    self._pending_messages.add(text.strip())
                    if self._tui:
                        if not self._tui.is_loader_active():
                            queued = self._pending_messages.count()
                            reason = self._tui.get_loader_text() or "Working..."
                            if (
                                hasattr(self._tui, "_focused_component")
                                and self._tui._focused_component is not None
                                and self._tui._focused_component is not self._input_component
                            ):
                                reason = (
                                    f"Waiting on {type(self._tui._focused_component).__name__}..."
                                )
                            self._tui.show_loader(
                                f"{queued} queued. {reason} (Esc to interrupt)",
                                spinner_color_fn=self._tui.theme.accent_fg,
                                text_color_fn=self._tui.theme.muted_fg,
                            )
                        self._tui.request_render()
            except trio.WouldBlock:
                if self._renderer:
                    self._renderer.add_system_message(
                        "Queue full (10) - dropped message. Wait for the current operation to finish."
                    )
                if self._tui:
                    self._tui.request_render()

    def _handle_open_editor(self, current_text: str) -> None:
        """Handle Ctrl+G to open external editor for message composition."""
        if not self._terminal:
            return

        # Run editor (this temporarily exits raw mode)
        edited_content = self._terminal.run_external_editor(current_text)

        # Reset TUI state before redrawing
        if self._tui:
            self._tui.reset_render_state()

        # If user saved content, update input and optionally submit
        if edited_content:
            # Strip any terminal control sequences that may have leaked in
            from .tui.utils import strip_terminal_control_sequences

            edited_content = strip_terminal_control_sequences(edited_content)
            if self._input_component:
                self._input_component.set_text(edited_content)
            # Auto-submit the edited content
            self._handle_input_submit(edited_content)
            # Clear input after submit
            if self._input_component:
                self._input_component.set_text("")

        # Force full redraw
        if self._tui:
            self._tui.request_render()

    def render_history(self, messages: list) -> None:
        """Render historical messages from resumed session.

        Args:
            messages: List of Message objects
        """
        if self._renderer:
            self._renderer.render_history(messages, skip_system=False)
            self._is_first_user_message = False

    def replace_history(self, messages: list) -> None:
        """Replace visible chat history after a session/context switch."""
        if self._renderer:
            self._renderer.clear_chat()
            self._renderer.render_history(messages, skip_system=False)
            self._is_first_user_message = False
        if self._tui:
            self._tui.request_render()

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

        async def input_reading_loop() -> None:
            import time

            from .protocol import InputExit, InputInterrupt

            CTRL_C_TIMEOUT = 1.5  # Seconds to wait for second Ctrl+C

            while True:
                if self._terminal and self._terminal._running:
                    msg = self._terminal.read_message()
                    if msg is not None:
                        key = msg.key if isinstance(msg, KeyPress) else None
                        # Check for Ctrl+C (ASCII 3) - double-tap to exit
                        if key == "\x03":
                            now = time.time()
                            if (
                                self._ctrl_c_pending
                                and (now - self._ctrl_c_pending) < CTRL_C_TIMEOUT
                            ):
                                # Second Ctrl+C within timeout - send exit
                                self._ctrl_c_pending = None
                                if self._input_send:
                                    try:
                                        self._input_send.send_nowait(InputExit())
                                    except trio.WouldBlock:
                                        pass  # Channel full, exit will be handled
                            else:
                                # First Ctrl+C - show message and wait
                                self._ctrl_c_pending = now
                                if self._renderer:
                                    self._renderer.add_system_message("Press Ctrl+C again to exit")
                            continue

                        # Check for Escape - interrupt current operation
                        if key == "\x1b":
                            # Set flag so get_input returns InputInterrupt
                            self._interrupt_pending = True
                            if self._renderer:
                                self._renderer.add_system_message("Interrupted")
                            # Send interrupt through channel to wake up get_input
                            if self._input_send:
                                try:
                                    self._input_send.send_nowait(InputInterrupt())
                                except trio.WouldBlock:
                                    pass  # Channel full, interrupt flag is set
                            continue

                        # Any other key cancels the pending Ctrl+C (paste events included).
                        if self._ctrl_c_pending is not None:
                            self._ctrl_c_pending = None

                        # Up arrow: when editor is empty and messages are queued, restore them for editing.
                        if (
                            key == "\x1b[A"
                            and self._input_component
                            and self._tui
                            and self._tui._focused_component is self._input_component
                            and not self._input_component.get_text().strip()
                            and self._pending_messages
                            and self._pending_messages.count() > 0
                        ):
                            self._restore_queued_messages_to_input()
                            continue

                        if self._tui:
                            self._tui._handle_input(msg)
                await trio.sleep(0.01)

        nursery.start_soon(input_reading_loop)

        # Also start animation loop
        if self._tui:
            nursery.start_soon(self._tui.run_animation_loop)

    def request_render(self) -> None:
        """Request a TUI render."""
        if self._tui:
            self._tui.request_render()
