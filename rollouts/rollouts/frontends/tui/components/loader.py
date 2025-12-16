"""
Loader component - displays a spinning animation with text.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..tui import Component
from ..utils import visible_width


class Loader(Component):
    """Component that displays a spinning animation with text."""

    _spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(
        self,
        text: str = "Working...",
        spinner_color_fn: Callable[[str], str] | None = None,
        text_color_fn: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize loader component.

        Args:
            text: Text to display after spinner
            spinner_color_fn: Function to colorize spinner (text -> styled text)
            text_color_fn: Function to colorize text (text -> styled text)
        """
        self._text = text
        self._spinner_color_fn = spinner_color_fn or (lambda x: x)
        self._text_color_fn = text_color_fn or (lambda x: x)
        self._start_time = time.time()
        self._running = True

    def stop(self) -> None:
        """Stop the animation."""
        self._running = False

    def invalidate(self) -> None:
        """No cached state to invalidate."""
        pass

    def render(self, width: int) -> list[str]:
        """Render loader with spinning animation."""
        if not self._running:
            return []

        # Calculate frame index based on elapsed time
        elapsed = time.time() - self._start_time
        frame_index = int(elapsed * 10) % len(self._spinner_frames)
        spinner = self._spinner_frames[frame_index]

        # Build line: spinner + space + text
        line = self._spinner_color_fn(spinner) + " " + self._text_color_fn(self._text)

        # Pad to width
        visible_len = visible_width(line)
        padding_needed = max(0, width - visible_len)
        padded_line = line + " " * padding_needed

        return [padded_line]
