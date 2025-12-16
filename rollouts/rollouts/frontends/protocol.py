"""Frontend protocol for pluggable TUI implementations.

This module defines the abstract interface that all frontends must implement,
enabling different UI implementations (Python TUI, Textual, Go/Bubbletea, TS/OpenTUI)
to work with the same agent loop.

The key abstraction is that the agent loop emits StreamEvent objects, and frontends
render them however they choose. Input is gathered via async callbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rollouts.dtypes import StreamEvent, ToolCall


@runtime_checkable
class Frontend(Protocol):
    """Protocol for agent frontends.

    All frontends must implement this interface to work with the agent runner.
    The agent loop calls these methods to:
    - Initialize/cleanup the frontend
    - Send streaming events for rendering
    - Get user input
    - Show/hide loading indicators
    - Confirm tool execution

    Implementations can range from simple stdout printing to full TUI frameworks
    to IPC bridges for external processes.

    Example implementations:
    - NoneFrontend: Simple stdout printing
    - TUIFrontend: Python-native terminal UI
    - TextualFrontend: Textual-based rich terminal UI
    - IPCFrontend: Subprocess bridge for Go/TS TUIs
    """

    async def start(self) -> None:
        """Initialize frontend (enter raw mode, spawn subprocess, etc.).

        Called once before the agent loop begins. Use this to:
        - Enter terminal raw mode
        - Initialize UI components
        - Spawn subprocess for IPC frontends
        - Set up signal handlers
        """
        ...

    async def stop(self) -> None:
        """Cleanup frontend (restore terminal, kill subprocess, etc.).

        Called once after the agent loop completes. Use this to:
        - Restore terminal state
        - Clean up UI resources
        - Kill subprocess for IPC frontends
        - Print final status messages
        """
        ...

    async def handle_event(self, event: StreamEvent) -> None:
        """Handle a streaming event from the agent.

        Events include:
        - LLMCallStart: About to call LLM
        - StreamStart: First token received
        - TextDelta: Text token received
        - ThinkingDelta: Thinking token (for reasoning models)
        - ToolCallStart/End: Tool call lifecycle
        - ToolResultReceived: Tool execution result
        - StreamDone/Error: Stream completion

        Args:
            event: Streaming event from the agent loop
        """
        ...

    async def get_input(self, prompt: str = "") -> str:
        """Get user input. Blocks until user submits.

        This is called when the agent needs user input to continue.
        The frontend should display the prompt and wait for the user
        to type a message and press enter.

        Args:
            prompt: Optional prompt to display (may be ignored by rich UIs)

        Returns:
            User's input string
        """
        ...

    async def confirm_tool(self, tool_call: ToolCall) -> bool:
        """Prompt user to confirm tool execution.

        Called when tool confirmation is enabled and the agent wants
        to execute a tool. The frontend should display the tool name
        and arguments, then ask the user to approve or reject.

        Args:
            tool_call: The tool call to confirm

        Returns:
            True if approved, False if rejected
        """
        ...

    def show_loader(self, text: str) -> None:
        """Show loading indicator with text.

        Called when the agent is performing a long-running operation
        like making an API call. The frontend should show a spinner
        or similar indicator with the given text.

        Args:
            text: Loading status text (e.g., "Thinking...", "Calling tool...")
        """
        ...

    def hide_loader(self) -> None:
        """Hide loading indicator.

        Called when the long-running operation completes.
        """
        ...


@runtime_checkable
class FrontendWithStatus(Frontend, Protocol):
    """Extended frontend protocol with status line support.

    Optional extension for frontends that support rich status displays
    like token counts, model info, session ID, etc.
    """

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
            model: Model name (e.g., "anthropic/claude-3-5-sonnet")
            session_id: Session ID for resumption
            input_tokens: Cumulative input tokens
            output_tokens: Cumulative output tokens
            cost: Cumulative cost in USD
            env_info: Environment-specific info (e.g., cwd, branch)
        """
        ...
