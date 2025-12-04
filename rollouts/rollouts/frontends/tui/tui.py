"""
Minimal TUI implementation with differential rendering.

Ported from pi-mono/packages/tui - same architecture, same visual output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol, Optional, List
import os
import time

from .terminal import Terminal
from .theme import Theme, DARK_THEME
from .utils import visible_width


# Spinner frames for loader animation
_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def render_loader_line(
    text: str,
    elapsed_time: float,
    width: int,
    spinner_color_fn: Callable[[str], str] = lambda x: x,
    text_color_fn: Callable[[str], str] = lambda x: x,
) -> str:
    """Render a single loader line. Pure function.

    Args:
        text: Text to display after spinner
        elapsed_time: Seconds since loader started (for frame calculation)
        width: Terminal width for padding
        spinner_color_fn: Function to colorize spinner
        text_color_fn: Function to colorize text

    Returns:
        Rendered line string
    """
    frame_index = int(elapsed_time * 10) % len(_SPINNER_FRAMES)
    spinner = _SPINNER_FRAMES[frame_index]

    line = spinner_color_fn(spinner) + " " + text_color_fn(text)
    padding = max(0, width - visible_width(line))
    return line + " " * padding


class Component(ABC):
    """Base class for all TUI components."""

    @abstractmethod
    def render(self, width: int) -> List[str]:
        """Render the component to lines for the given viewport width.

        Args:
            width: Current viewport width

        Returns:
            Array of strings, each representing a line
        """
        ...

    def handle_input(self, data: str) -> None:
        """Optional handler for keyboard input when component has focus."""
        pass

    def invalidate(self) -> None:
        """Invalidate any cached rendering state.

        Called when theme changes or when component needs to re-render from scratch.
        """
        pass


class Container(Component):
    """A component that contains other components."""

    def __init__(self) -> None:
        self.children: List[Component] = []

    def add_child(self, component: Component) -> None:
        """Add a child component."""
        self.children.append(component)

    def remove_child(self, component: Component) -> None:
        """Remove a child component."""
        if component in self.children:
            self.children.remove(component)

    def clear(self) -> None:
        """Remove all children."""
        self.children.clear()

    def invalidate(self) -> None:
        """Invalidate all children."""
        for child in self.children:
            child.invalidate()

    def render(self, width: int) -> List[str]:
        """Render all children, concatenating their lines."""
        lines: List[str] = []
        for child in self.children:
            lines.extend(child.render(width))
        return lines


class TUI(Container):
    """Main class for managing terminal UI with differential rendering."""

    def __init__(self, terminal: Terminal, theme: Optional[Theme] = None) -> None:
        super().__init__()
        self._terminal = terminal
        self.theme = theme or DARK_THEME
        self._previous_lines: List[str] = []
        self._previous_width: int = 0
        self._focused_component: Optional[Component] = None
        self._render_requested: bool = False
        self._cursor_row: int = 0  # Track where cursor is (0-indexed, relative to our first line)
        self._running: bool = False

        # Loader state - centralized here instead of separate Loader class
        # Why: Loader needs periodic re-renders during blocking operations (e.g. API calls).
        # TUI owns the render loop, so it's natural for TUI to own animation timing.
        self._loader_text: Optional[str] = None  # None = no loader showing
        self._loader_start_time: float = 0.0
        self._loader_spinner_color_fn: Callable[[str], str] = lambda x: x
        self._loader_text_color_fn: Callable[[str], str] = lambda x: x
        self._animation_task_running: bool = False

    def set_focus(self, component: Optional[Component]) -> None:
        """Set the focused component for input handling."""
        self._focused_component = component

    def start(self) -> None:
        """Start the TUI, enabling raw mode and input handling."""
        self._terminal.start(
            on_input=self._handle_input,
            on_resize=self.request_render,
        )
        self._terminal.hide_cursor()
        self._running = True
        self.request_render()

    def stop(self) -> None:
        """Stop the TUI and restore terminal state."""
        self._running = False
        self._loader_text = None  # Clear loader on stop
        self._terminal.show_cursor()
        self._terminal.stop()

    def show_loader(
        self,
        text: str,
        spinner_color_fn: Callable[[str], str] = lambda x: x,
        text_color_fn: Callable[[str], str] = lambda x: x,
    ) -> None:
        """Show a loader with spinning animation.

        Args:
            text: Text to display after spinner (e.g. "Calling LLM...")
            spinner_color_fn: Function to colorize spinner
            text_color_fn: Function to colorize text
        """
        self._loader_text = text
        self._loader_start_time = time.time()
        self._loader_spinner_color_fn = spinner_color_fn
        self._loader_text_color_fn = text_color_fn
        self.request_render()

    def hide_loader(self) -> None:
        """Hide the loader."""
        self._loader_text = None
        self.request_render()

    def is_loader_active(self) -> bool:
        """Check if loader is currently showing."""
        return self._loader_text is not None

    async def run_animation_loop(self) -> None:
        """Run the animation timer loop.

        Call this as a background task. The loop runs continuously and triggers
        re-renders every 80ms when the loader is active.
        Why 80ms: matches pi-mono's animation interval, gives smooth 12.5fps animation.

        Usage with Trio:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(tui.run_animation_loop)
                # ... do other work ...
        """
        import trio

        self._animation_task_running = True
        try:
            while self._animation_task_running and self._running:
                await trio.sleep(0.08)  # 80ms
                # Only trigger re-render if loader is active
                if self._loader_text is not None:
                    self.request_render()
        finally:
            self._animation_task_running = False

    def stop_animation_loop(self) -> None:
        """Signal the animation loop to stop."""
        self._animation_task_running = False

    def render(self, width: int) -> List[str]:
        """Render all children plus loader if active."""
        lines = super().render(width)

        # Append loader line if active
        if self._loader_text is not None:
            elapsed = time.time() - self._loader_start_time
            loader_line = render_loader_line(
                self._loader_text,
                elapsed,
                width,
                self._loader_spinner_color_fn,
                self._loader_text_color_fn,
            )
            lines.append(loader_line)

        return lines

    def request_render(self) -> None:
        """Request a render on the next tick.

        Multiple requests are coalesced into a single render.
        """
        if self._render_requested:
            return
        self._render_requested = True
        # In Python we do immediate render since we don't have process.nextTick
        # For async usage, caller should await trio.sleep(0) or similar
        self._do_render()
        self._render_requested = False

    def _handle_input(self, data: str) -> None:
        """Handle keyboard input, passing to focused component."""
        # Ctrl+C (ASCII 3) should be handled by the application layer
        # We don't consume it here, just ignore it if we see it
        if len(data) > 0 and ord(data[0]) == 3:
            return
        if self._focused_component is not None:
            self._focused_component.handle_input(data)
            self.request_render()

    def _do_render(self) -> None:
        """Perform the actual render with differential updates."""
        width = self._terminal.columns
        height = self._terminal.rows

        # Render all components to get new lines
        new_lines = self.render(width)

        # Width changed - need full re-render
        width_changed = self._previous_width != 0 and self._previous_width != width

        # First render - just output everything without clearing
        if len(self._previous_lines) == 0:
            buffer = "\x1b[?2026h"  # Begin synchronized output
            for i, line in enumerate(new_lines):
                if i > 0:
                    buffer += "\r\n"
                buffer += line
            buffer += "\x1b[?2026l"  # End synchronized output
            self._terminal.write(buffer)
            # After rendering N lines, cursor is at end of last line (line N-1)
            self._cursor_row = len(new_lines) - 1
            self._previous_lines = new_lines
            self._previous_width = width
            return

        # Width changed - full re-render
        if width_changed:
            buffer = "\x1b[?2026h"  # Begin synchronized output
            buffer += "\x1b[3J\x1b[2J\x1b[H"  # Clear scrollback, screen, and home
            for i, line in enumerate(new_lines):
                if i > 0:
                    buffer += "\r\n"
                buffer += line
            buffer += "\x1b[?2026l"  # End synchronized output
            self._terminal.write(buffer)
            self._cursor_row = len(new_lines) - 1
            self._previous_lines = new_lines
            self._previous_width = width
            return

        # Find first changed line
        first_changed = -1
        max_lines = max(len(new_lines), len(self._previous_lines))
        for i in range(max_lines):
            old_line = self._previous_lines[i] if i < len(self._previous_lines) else ""
            new_line = new_lines[i] if i < len(new_lines) else ""

            if old_line != new_line:
                if first_changed == -1:
                    first_changed = i

        # No changes
        if first_changed == -1:
            return

        # Check if first_changed is outside the viewport
        # cursor_row is the line where cursor is (0-indexed)
        # Viewport shows lines from (cursor_row - height + 1) to cursor_row
        # If first_changed < viewport_top, we need full re-render
        viewport_top = self._cursor_row - height + 1
        if first_changed < viewport_top:
            # First change is above viewport - need full re-render
            buffer = "\x1b[?2026h"  # Begin synchronized output
            buffer += "\x1b[3J\x1b[2J\x1b[H"  # Clear scrollback, screen, and home
            for i, line in enumerate(new_lines):
                if i > 0:
                    buffer += "\r\n"
                buffer += line
            buffer += "\x1b[?2026l"  # End synchronized output
            self._terminal.write(buffer)
            self._cursor_row = len(new_lines) - 1
            self._previous_lines = new_lines
            self._previous_width = width
            return

        # Render from first changed line to end
        buffer = "\x1b[?2026h"  # Begin synchronized output

        # Move cursor to first changed line
        line_diff = first_changed - self._cursor_row
        if line_diff > 0:
            buffer += f"\x1b[{line_diff}B"  # Move down
        elif line_diff < 0:
            buffer += f"\x1b[{-line_diff}A"  # Move up

        buffer += "\r"  # Move to column 0

        # Render from first changed line to end, clearing each line before writing
        for i in range(first_changed, len(new_lines)):
            if i > first_changed:
                buffer += "\r\n"
            buffer += "\x1b[2K"  # Clear current line

            line = new_lines[i]
            if visible_width(line) > width:
                # Line too wide - write debug log and raise
                crash_log_path = Path.home() / ".rollouts" / "crash.log"
                crash_log_path.parent.mkdir(parents=True, exist_ok=True)
                crash_data = [
                    f"Crash at {datetime.now().isoformat()}",
                    f"Terminal width: {width}",
                    f"Line {i} visible width: {visible_width(line)}",
                    "",
                    "=== All rendered lines ===",
                    *[f"[{idx}] (w={visible_width(ln)}) {ln}" for idx, ln in enumerate(new_lines)],
                    "",
                ]
                crash_log_path.write_text("\n".join(crash_data))
                raise ValueError(
                    f"Rendered line {i} exceeds terminal width. Debug log: {crash_log_path}"
                )
            buffer += line

        # If we had more lines before, clear them
        if len(self._previous_lines) > len(new_lines):
            extra_lines = len(self._previous_lines) - len(new_lines)
            # Clear each extra line. We're currently at first_changed.
            # The lines to clear start at len(new_lines).
            for i in range(extra_lines):
                if i == 0 and first_changed == len(new_lines):
                    # Already at first stale line, just clear it
                    buffer += "\x1b[2K"
                else:
                    buffer += "\r\n\x1b[2K"
            # Move cursor back up to end of actual content
            # We need to end at len(new_lines) - 1
            lines_to_move_up = first_changed + extra_lines - (len(new_lines) - 1)
            if first_changed == len(new_lines):
                lines_to_move_up = extra_lines
            if lines_to_move_up > 0:
                buffer += f"\x1b[{lines_to_move_up}A"

        buffer += "\x1b[?2026l"  # End synchronized output

        # Write entire buffer at once
        self._terminal.write(buffer)

        # Cursor is now at end of last line
        self._cursor_row = len(new_lines) - 1

        self._previous_lines = new_lines
        self._previous_width = width
