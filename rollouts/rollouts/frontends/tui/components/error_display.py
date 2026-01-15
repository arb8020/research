"""
ErrorDisplay component - displays errors as distinct blocks in chat flow.

Similar to QuestionSelectorComponent in visual style but for error messages.
Shows retry errors, stream errors, and other important error information
in a prominent, non-truncated format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..tui import Container
from .spacer import Spacer
from .text import Text

if TYPE_CHECKING:
    from ..theme import Theme


class ErrorDisplay(Container):
    """Component that displays an error as a distinct block in the chat.

    Unlike inline error messages in the loader, this component:
    - Shows the full error message without truncation
    - Appears as a persistent element in chat history
    - Uses styling to draw attention (warning/error colors)
    """

    def __init__(
        self,
        title: str,
        message: str,
        details: str | None = None,
        theme: Theme | None = None,
        error_type: str = "error",  # "error", "warning", "retry"
    ) -> None:
        """Initialize error display component.

        Args:
            title: Short title (e.g., "API Error", "Retry Failed")
            message: Main error message
            details: Optional additional details (e.g., traceback, full response)
            theme: Theme for styling
            error_type: Type of error for styling ("error", "warning", "retry")
        """
        super().__init__()
        self._title = title
        self._message = message
        self._details = details
        self._theme = theme
        self._error_type = error_type

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI components."""
        self.clear()

        # Get styling based on error type
        if self._theme:
            if self._error_type == "warning" or self._error_type == "retry":
                icon = "⚠"
                title_style = self._theme.warning_fg
                bg_fn = self._theme.warning_subtle_bg
            else:
                icon = "✖"
                title_style = self._theme.error_fg
                bg_fn = self._theme.error_subtle_bg
        else:
            icon = "⚠" if self._error_type in ("warning", "retry") else "✖"
            title_style = None
            bg_fn = None

        # Title line
        title_text = f"{icon} {self._title}"
        if self._theme:
            title_styled = title_style(title_text)
        else:
            title_styled = title_text

        self.add_child(Text(title_styled, padding_x=2, padding_y=0, theme=self._theme))

        # Message (can be multi-line)
        if self._message:
            # Indent message lines
            message_lines = self._message.strip().split("\n")
            for line in message_lines:
                if self._theme:
                    line_styled = self._theme.muted_fg(f"  {line}")
                else:
                    line_styled = f"  {line}"
                self.add_child(Text(line_styled, padding_x=2, padding_y=0, theme=self._theme))

        # Details (if provided, show all of it)
        if self._details:
            self.add_child(Spacer(1))
            details_lines = self._details.strip().split("\n")
            for line in details_lines:
                if self._theme:
                    line_styled = self._theme.muted_fg(f"  {line}")
                else:
                    line_styled = f"  {line}"
                self.add_child(
                    Text(line_styled, padding_x=2, padding_y=0, theme=self._theme, dim=True)
                )

        self.add_child(Spacer(1))

    def render(self, width: int) -> list[str]:
        """Render all children."""
        lines: list[str] = []
        for child in self.children:
            lines.extend(child.render(width))
        return lines


class RetryErrorDisplay(ErrorDisplay):
    """Specialized error display for retry failures.

    Shows attempt count and timing information along with the error.
    """

    def __init__(
        self,
        error_message: str,
        attempt: int,
        max_attempts: int,
        delay_seconds: float | None = None,
        theme: Theme | None = None,
        is_final: bool = False,
    ) -> None:
        """Initialize retry error display.

        Args:
            error_message: The error that caused the retry
            attempt: Current attempt number
            max_attempts: Maximum attempts allowed
            delay_seconds: Seconds until next retry (None if final failure)
            theme: Theme for styling
            is_final: Whether this is the final failure (all retries exhausted)
        """
        if is_final:
            title = f"Retry Failed (after {attempt} attempts)"
            error_type = "error"
        else:
            delay_str = f" - retrying in {int(delay_seconds)}s" if delay_seconds else ""
            title = f"API Error (attempt {attempt}/{max_attempts}){delay_str}"
            error_type = "retry"

        super().__init__(
            title=title,
            message=error_message,
            details=None,
            theme=theme,
            error_type=error_type,
        )
