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
        """Render the status line as two lines: model on first, tokens/cost on second."""
        gray = "\x1b[38;5;245m"
        reset = "\x1b[0m"
        available_width = width - 2  # 2 for left margin "  "

        lines: list[str] = []

        # Line 1: model
        if self._model:
            model_content = f"model:{self._model}"
            model_len = visible_width(model_content)
            if model_len > available_width:
                model_content = model_content[:available_width - 1] + "…"
            padding = " " * max(0, available_width - visible_width(model_content))
            lines.append(f"  {gray}{model_content}{padding}{reset}")

        # Line 2: tokens and cost
        usage_parts: list[str] = []
        if self._input_tokens > 0 or self._output_tokens > 0:
            usage_parts.append(f"tokens:{self._input_tokens}↓/{self._output_tokens}↑")
        if self._cost > 0:
            usage_parts.append(f"cost:${self._cost:.4f}")

        if usage_parts:
            usage_content = "  │  ".join(usage_parts)
            usage_len = visible_width(usage_content)
            if usage_len > available_width:
                usage_content = usage_content[:available_width - 1] + "…"
            padding = " " * max(0, available_width - visible_width(usage_content))
            lines.append(f"  {gray}{usage_content}{padding}{reset}")

        # If nothing to show, return empty line
        if not lines:
            return [f"  {' ' * available_width}"]

        return lines
