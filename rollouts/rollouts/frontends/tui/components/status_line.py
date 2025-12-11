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
        self._cost: float = 0.0

    def set_session_id(self, session_id: str | None) -> None:
        """Set the session ID to display."""
        self._session_id = session_id

    def set_model(self, model: str | None) -> None:
        """Set the model name to display."""
        self._model = model

    def set_tokens(self, input_tokens: int, output_tokens: int, cost: float = 0.0) -> None:
        """Set token counts and cost."""
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cost = cost

    def add_tokens(self, input_tokens: int, output_tokens: int, cost: float = 0.0) -> None:
        """Add to token counts and cost."""
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._cost += cost

    def render(self, width: int) -> List[str]:
        """Render the status line."""
        # Build status parts
        parts: list[str] = []


        if self._model:
            parts.append(f"model:{self._model}")

        if self._input_tokens > 0 or self._output_tokens > 0:
            parts.append(f"tokens:{self._input_tokens}↓/{self._output_tokens}↑")

        if self._cost > 0:
            parts.append(f"cost:${self._cost:.4f}")

        # Join with separators
        content = "  │  ".join(parts) if parts else ""

        # Apply muted color
        gray = "\x1b[38;5;245m"
        reset = "\x1b[0m"

        # Available width for content (2 for left margin "  ")
        available_width = width - 2
        visible_len = visible_width(content)

        # Truncate content if it exceeds available width
        if visible_len > available_width:
            # Truncate and add ellipsis
            truncated = content[:available_width - 1] + "…"
            return [f"  {gray}{truncated}{reset}"]

        # Pad to width
        padding = " " * max(0, available_width - visible_len)

        return [f"  {gray}{content}{padding}{reset}"]
