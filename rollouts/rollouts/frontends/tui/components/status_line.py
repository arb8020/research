"""
Status line component - displays session info, model, tokens below input.
"""

from __future__ import annotations

from typing import Callable, Optional, List, TYPE_CHECKING

from ..tui import Component
from ..utils import visible_width

if TYPE_CHECKING:
    from ..theme import Theme


class StatusLine(Component):
    """Single-line status bar showing session info, model, and token counts."""

    def __init__(
        self,
        theme: Optional["Theme"] = None,
    ) -> None:
        """Initialize status line.

        Args:
            theme: Theme for styling
        """
        from ..theme import DARK_THEME
        self._theme = theme or DARK_THEME

        # Status fields
        self._session_id: str | None = None
        self._model: str | None = None
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    def set_session_id(self, session_id: str | None) -> None:
        """Set the session ID to display."""
        self._session_id = session_id

    def set_model(self, model: str | None) -> None:
        """Set the model name to display."""
        self._model = model

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Set token counts."""
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Add to token counts."""
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

    def render(self, width: int) -> List[str]:
        """Render the status line."""
        # Build status parts
        parts: list[str] = []

        if self._session_id:
            # Truncate session ID if needed (show last 6 chars)
            short_id = self._session_id[-15:] if len(self._session_id) > 15 else self._session_id
            parts.append(f"session:{short_id}")

        if self._model:
            # Truncate model name if needed
            short_model = self._model[-20:] if len(self._model) > 20 else self._model
            parts.append(f"model:{short_model}")

        if self._input_tokens > 0 or self._output_tokens > 0:
            parts.append(f"tokens:{self._input_tokens}↓/{self._output_tokens}↑")

        # Join with separators
        content = "  │  ".join(parts) if parts else ""

        # Apply muted color
        gray = "\x1b[38;5;245m"
        reset = "\x1b[0m"

        # Pad to width
        visible_len = visible_width(content)
        padding = " " * max(0, width - visible_len - 4)  # 4 for left margin

        return [f"  {gray}{content}{padding}{reset}"]
