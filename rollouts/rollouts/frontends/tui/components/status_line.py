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
        self._env_info: dict[str, str] | None = None

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

    def set_env_info(self, env_info: dict[str, str] | None) -> None:
        """Set environment info to display."""
        self._env_info = env_info

    def _wrap_parts(self, parts: list[str], available_width: int, separator: str = "  │  ") -> list[str]:
        """Wrap parts across multiple lines at part boundaries.

        Args:
            parts: List of content parts to join
            available_width: Max width per line
            separator: String to join parts with

        Returns:
            List of wrapped lines (without styling/padding)
        """
        if not parts:
            return []

        lines: list[str] = []
        current_line = ""
        sep_width = visible_width(separator)

        for part in parts:
            part_width = visible_width(part)

            if not current_line:
                # First part on this line
                if part_width <= available_width:
                    current_line = part
                else:
                    # Single part too wide - truncate it
                    current_line = part[:available_width - 1] + "…"
            else:
                # Check if adding this part fits
                new_width = visible_width(current_line) + sep_width + part_width
                if new_width <= available_width:
                    current_line = current_line + separator + part
                else:
                    # Wrap to new line
                    lines.append(current_line)
                    if part_width <= available_width:
                        current_line = part
                    else:
                        current_line = part[:available_width - 1] + "…"

        if current_line:
            lines.append(current_line)

        return lines

    def render(self, width: int) -> List[str]:
        """Render the status line, wrapping at part boundaries if needed."""
        gray = "\x1b[38;5;245m"
        reset = "\x1b[0m"
        available_width = width - 2  # 2 for left margin "  "

        lines: list[str] = []

        # Line 1: model and env info
        line1_parts: list[str] = []
        if self._model:
            line1_parts.append(f"model:{self._model}")
        if self._env_info:
            for key, value in self._env_info.items():
                line1_parts.append(f"{key}:{value}")

        for line_content in self._wrap_parts(line1_parts, available_width):
            padding = " " * max(0, available_width - visible_width(line_content))
            lines.append(f"  {gray}{line_content}{padding}{reset}")

        # Line 2: tokens and cost
        usage_parts: list[str] = []
        if self._input_tokens > 0 or self._output_tokens > 0:
            usage_parts.append(f"tokens:{self._input_tokens}↓/{self._output_tokens}↑")
        if self._cost > 0:
            usage_parts.append(f"cost:${self._cost:.4f}")

        for line_content in self._wrap_parts(usage_parts, available_width):
            padding = " " * max(0, available_width - visible_width(line_content))
            lines.append(f"  {gray}{line_content}{padding}{reset}")

        # If nothing to show, return empty line
        if not lines:
            return [f"  {' ' * available_width}"]

        return lines
