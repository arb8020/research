"""TextualFrontend - Rich TUI using the Textual library.

This frontend provides a rich, modern terminal UI using Textual,
similar to tools like Consoul, Claude Code, and OpenCode.

Features:
- Markdown rendering for messages
- Syntax-highlighted code blocks
- Tool call visualization
- Scrollable chat history
- Status bar with token counts
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import trio

if TYPE_CHECKING:
    from rollouts.dtypes import StreamEvent, ToolCall


class TextualFrontend:
    """Textual-based rich terminal UI frontend.

    Uses the Textual library for a modern, feature-rich terminal experience.
    This is the recommended frontend for interactive use.

    Example usage:
        frontend = TextualFrontend(theme="monokai")
        await run_interactive(trajectory, endpoint, frontend=frontend)
    """

    def __init__(
        self,
        theme: str = "dark",
        show_thinking: bool = True,
        show_tool_details: bool = True,
    ) -> None:
        """Initialize TextualFrontend.

        Args:
            theme: Color theme name (dark, monokai, dracula, etc.)
            show_thinking: Whether to show thinking/reasoning content
            show_tool_details: Whether to show detailed tool call info
        """
        self.theme_name = theme
        self.show_thinking = show_thinking
        self.show_tool_details = show_tool_details

        # Textual app (initialized in start())
        # Type is App subclass defined in start() - use Any to avoid forward ref issues
        self._app: Any = None
        self._app_task: asyncio.Task | None = None

        # Input coordination
        self._input_event: trio.Event | None = None
        self._input_value: str = ""

        # Status tracking
        self._model: str = ""
        self._session_id: str | None = None
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._cost: float = 0.0

    async def start(self) -> None:
        """Initialize and start the Textual app."""
        try:
            from textual.app import App, ComposeResult
            from textual.binding import Binding
            from textual.containers import Horizontal, Vertical, VerticalScroll
            from textual.widgets import Footer, Header, Input, Markdown, Static
        except ImportError as e:
            raise ImportError(
                "Textual is required for TextualFrontend. Install with: pip install textual"
            ) from e

        frontend = self  # Capture for inner class

        class RolloutsTextualApp(App):
            """Main Textual application for rollouts."""

            CSS = """
            Screen {
                background: $surface;
            }

            #chat-container {
                height: 1fr;
                border: solid $primary;
            }

            .message {
                margin: 1 2;
                padding: 1 2;
            }

            .message-user {
                background: $primary-darken-3;
                border: solid $primary;
            }

            .message-assistant {
                background: $surface-darken-1;
                border: solid $secondary;
            }

            .message-system {
                background: $warning-darken-3;
                border: dashed $warning;
            }

            .message-tool {
                background: $accent-darken-3;
                border: solid $accent;
                margin-left: 4;
            }

            #input-container {
                height: auto;
                dock: bottom;
                padding: 1;
            }

            #user-input {
                width: 100%;
            }

            #status-bar {
                dock: bottom;
                height: 1;
                background: $surface-darken-2;
                padding: 0 2;
            }

            .thinking {
                color: $text-muted;
                background: $surface-darken-2;
                padding: 0 1;
                margin-bottom: 1;
            }
            """

            BINDINGS = [
                Binding("ctrl+c", "quit", "Quit"),
                Binding("escape", "interrupt", "Interrupt"),
            ]

            def __init__(self) -> None:
                super().__init__()
                self.frontend = frontend
                self._current_message: Markdown | None = None
                self._current_text: str = ""
                self._current_thinking: str = ""

            def compose(self) -> ComposeResult:
                yield Header(show_clock=True)
                with Vertical():
                    with VerticalScroll(id="chat-container"):
                        pass  # Messages will be added dynamically
                    with Horizontal(id="input-container"):
                        yield Input(placeholder="Type your message...", id="user-input")
                yield Static(id="status-bar")
                yield Footer()

            def on_mount(self) -> None:
                self.title = "Rollouts"
                self._update_status()

            def on_input_submitted(self, event: Input.Submitted) -> None:
                """Handle input submission."""
                if event.value.strip():
                    self.frontend._input_value = event.value.strip()
                    if self.frontend._input_event:
                        self.frontend._input_event.set()
                    event.input.clear()

            def action_interrupt(self) -> None:
                """Handle escape key - interrupt current operation."""
                # Signal interrupt to frontend
                pass

            def _update_status(self) -> None:
                """Update status bar."""
                status = self.query_one("#status-bar", Static)
                parts = []
                if self.frontend._model:
                    parts.append(f"Model: {self.frontend._model}")
                if self.frontend._session_id:
                    parts.append(f"Session: {self.frontend._session_id[:8]}...")
                if self.frontend._input_tokens or self.frontend._output_tokens:
                    parts.append(
                        f"Tokens: {self.frontend._input_tokens}/{self.frontend._output_tokens}"
                    )
                if self.frontend._cost > 0:
                    parts.append(f"Cost: ${self.frontend._cost:.4f}")
                status.update(" | ".join(parts) if parts else "Ready")

            async def add_user_message(self, text: str) -> None:
                """Add a user message to the chat."""
                chat = self.query_one("#chat-container", VerticalScroll)
                msg = Static(f"**You:** {text}", classes="message message-user")
                await chat.mount(msg)
                chat.scroll_end()

            async def add_assistant_message(self, text: str) -> None:
                """Add or update assistant message."""
                chat = self.query_one("#chat-container", VerticalScroll)

                if self._current_message is None:
                    self._current_message = Markdown("", classes="message message-assistant")
                    await chat.mount(self._current_message)

                self._current_text = text
                self._current_message.update(f"**Assistant:**\n\n{text}")
                chat.scroll_end()

            def finalize_assistant_message(self) -> None:
                """Mark current assistant message as complete."""
                self._current_message = None
                self._current_text = ""
                self._current_thinking = ""

            async def add_thinking(self, text: str) -> None:
                """Add or update thinking content."""
                if not self.frontend.show_thinking:
                    return

                chat = self.query_one("#chat-container", VerticalScroll)
                self._current_thinking += text

                # Update or create thinking widget
                thinking_id = "current-thinking"
                try:
                    thinking = self.query_one(f"#{thinking_id}", Static)
                    thinking.update(f"*Thinking:* {self._current_thinking}")
                except Exception:
                    thinking = Static(
                        f"*Thinking:* {self._current_thinking}",
                        classes="thinking",
                        id=thinking_id,
                    )
                    await chat.mount(thinking)
                chat.scroll_end()

            async def add_tool_call(
                self, name: str, args: dict, result: str | None = None, is_error: bool = False
            ) -> None:
                """Add a tool call to the chat."""
                chat = self.query_one("#chat-container", VerticalScroll)

                # Format tool call
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                if len(args_str) > 60:
                    args_str = args_str[:57] + "..."

                content = f"**Tool:** `{name}({args_str})`"
                if result:
                    preview = result[:200] + "..." if len(result) > 200 else result
                    prefix = "❌ Error" if is_error else "✓ Result"
                    content += f"\n{prefix}: {preview}"

                msg = Static(content, classes="message message-tool")
                await chat.mount(msg)
                chat.scroll_end()

            async def add_system_message(self, text: str) -> None:
                """Add a system message."""
                chat = self.query_one("#chat-container", VerticalScroll)
                msg = Static(f"**System:** {text}", classes="message message-system")
                await chat.mount(msg)
                chat.scroll_end()

            async def show_loader(self, text: str) -> None:
                """Show loading indicator."""
                status = self.query_one("#status-bar", Static)
                status.update(f"⏳ {text}")

            def hide_loader(self) -> None:
                """Hide loading indicator."""
                self._update_status()

        # Store app class for later instantiation
        self._app_class = RolloutsTextualApp
        self._app = RolloutsTextualApp()

        # Run app in background
        # Note: Textual apps are designed to run synchronously
        # We'll need to use run_async or integrate differently
        # For now, this is a placeholder

    async def stop(self) -> None:
        """Stop the Textual app."""
        if self._app:
            self._app.exit()

    async def handle_event(self, event: StreamEvent) -> None:
        """Handle streaming event.

        Args:
            event: StreamEvent from agent loop
        """
        if not self._app:
            return

        from rollouts.dtypes import (
            LLMCallStart,
            StreamDone,
            StreamError,
            StreamStart,
            TextDelta,
            ThinkingDelta,
            ToolCallEnd,
            ToolResultReceived,
        )

        match event:
            case LLMCallStart():
                await self._app.show_loader("Calling LLM...")

            case StreamStart():
                await self._app.show_loader("Streaming...")

            case TextDelta(delta=text):
                self._app._current_text += text
                await self._app.add_assistant_message(self._app._current_text)

            case ThinkingDelta(delta=text):
                await self._app.add_thinking(text)

            case ToolCallEnd(tool_call=tc):
                await self._app.add_tool_call(tc.name, dict(tc.args))

            case ToolResultReceived(tool_call_id=_, content=_, is_error=_):
                # Tool results are shown inline with tool calls
                pass

            case StreamDone():
                self._app.finalize_assistant_message()
                self._app.hide_loader()

            case StreamError(error=error):
                await self._app.add_system_message(f"Error: {error}")
                self._app.hide_loader()

    async def get_input(self, prompt: str = "") -> str:
        """Get user input.

        Args:
            prompt: Ignored (Textual has its own input)

        Returns:
            User's input string
        """
        self._input_event = trio.Event()
        self._input_value = ""

        # Wait for input
        await self._input_event.wait()

        # Add to chat
        if self._app:
            await self._app.add_user_message(self._input_value)

        return self._input_value

    async def confirm_tool(self, tool_call: ToolCall) -> bool:
        """Confirm tool execution.

        Args:
            tool_call: Tool call to confirm

        Returns:
            True if approved
        """
        if self._app:
            args_str = ", ".join(f"{k}={v!r}" for k, v in tool_call.args.items())
            await self._app.add_system_message(
                f"⚠️ Confirm tool: {tool_call.name}({args_str[:50]}...)\n"
                "Type 'y' to approve, 'n' to reject"
            )

        response = await self.get_input()
        return response.strip().lower() in ("y", "yes", "")

    def show_loader(self, text: str) -> None:
        """Show loading indicator.

        Args:
            text: Loading text
        """
        if self._app:
            # Schedule coroutine in the Textual event loop
            asyncio.create_task(self._app.show_loader(text))

    def hide_loader(self) -> None:
        """Hide loading indicator."""
        if self._app:
            self._app.hide_loader()

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
        """Update status bar.

        Args:
            model: Model name
            session_id: Session ID
            input_tokens: Input token count
            output_tokens: Output token count
            cost: Cost in USD
            env_info: Environment info
        """
        if model is not None:
            self._model = model
        if session_id is not None:
            self._session_id = session_id
        if input_tokens is not None:
            self._input_tokens = input_tokens
        if output_tokens is not None:
            self._output_tokens = output_tokens
        if cost is not None:
            self._cost = cost

        if self._app:
            self._app._update_status()


# Note: TextualFrontend is a work in progress.
# The Textual library has its own event loop that doesn't integrate
# cleanly with trio. Full implementation would require either:
# 1. Running Textual in a separate thread
# 2. Using textual's async_run with proper event loop integration
# 3. Using IPC to communicate with a separate Textual process
#
# For now, use TUIFrontend or NoneFrontend for production use.
