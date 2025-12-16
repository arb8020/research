"""
LoaderContainer component - holds loader in a fixed position without pushing content.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..tui import Component
from ..utils import visible_width


class LoaderContainer(Component):
    """Component that renders a loader in a fixed space without pushing other content."""

    _spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(
        self,
        spinner_color_fn: Callable[[str], str] | None = None,
        text_color_fn: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize loader container.

        Args:
            spinner_color_fn: Function to colorize spinner (text -> styled text)
            text_color_fn: Function to colorize text (text -> styled text)
        """
        self._spinner_color_fn = spinner_color_fn or (lambda x: x)
        self._text_color_fn = text_color_fn or (lambda x: x)
        self._loader_text: str | None = None
        self._loader_start_time: float = 0.0

    def set_loader(self, text: str) -> None:
        """Show the loader with the given text."""
        self._loader_text = text
        self._loader_start_time = time.time()

    def clear_loader(self) -> None:
        """Hide the loader."""
        self._loader_text = None

    def is_active(self) -> bool:
        """Check if loader is currently showing."""
        return self._loader_text is not None

    def invalidate(self) -> None:
        """No cached state to invalidate."""
        pass

    def render(self, width: int) -> list[str]:
        """Render loader with its preceding spacer when active, otherwise render nothing."""
        if not self._loader_text:
            # When no loader, render nothing (no spacer, no loader line)
            return []

        # Calculate frame index based on elapsed time
        elapsed = time.time() - self._loader_start_time
        frame_index = int(elapsed * 10) % len(self._spinner_frames)
        spinner = self._spinner_frames[frame_index]

        # Build line: spinner + space + text
        line = self._spinner_color_fn(spinner) + " " + self._text_color_fn(self._loader_text)

        # Pad to width
        visible_len = visible_width(line)
        padding_needed = max(0, width - visible_len)
        padded_line = line + " " * padding_needed

        # Return spacer before loader + loader line (loader brings its "before" with it)
        return ["", padded_line]
