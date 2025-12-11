"""
Terminal abstraction for TUI.

Provides a clean interface for terminal I/O, cursor control, and raw mode handling.
"""

from __future__ import annotations

import atexit
import sys
import tty
import termios
import signal
import os
from typing import Callable, Protocol, Optional, List


# Global reference for atexit cleanup
_active_terminal: Optional["ProcessTerminal"] = None


def _cleanup_terminal():
    """Atexit handler to restore terminal state."""
    global _active_terminal
    if _active_terminal is not None:
        _active_terminal.stop()


class Terminal(Protocol):
    """Protocol for terminal implementations."""

    def start(self, on_input: Callable[[str], None], on_resize: Callable[[], None]) -> None:
        """Start the terminal with input and resize handlers."""
        ...

    def stop(self) -> None:
        """Stop the terminal and restore state."""
        ...

    def write(self, data: str) -> None:
        """Write output to terminal."""
        ...

    @property
    def columns(self) -> int:
        """Get terminal width."""
        ...

    @property
    def rows(self) -> int:
        """Get terminal height."""
        ...

    def hide_cursor(self) -> None:
        """Hide the cursor."""
        ...

    def show_cursor(self) -> None:
        """Show the cursor."""
        ...

    def clear_line(self) -> None:
        """Clear current line."""
        ...

    def clear_from_cursor(self) -> None:
        """Clear from cursor to end of screen."""
        ...

    def clear_screen(self) -> None:
        """Clear entire screen and move cursor to (0,0)."""
        ...


class ProcessTerminal:
    """Real terminal using sys.stdin/stdout with raw mode support."""

    def __init__(self) -> None:
        self._old_settings: Optional[List] = None
        self._input_handler: Optional[Callable[[str], None]] = None
        self._resize_handler: Optional[Callable[[], None]] = None
        self._old_sigwinch = None
        self._running = False

    def start(self, on_input: Callable[[str], None], on_resize: Callable[[], None]) -> None:
        """Start terminal in raw mode with input/resize handlers."""
        global _active_terminal

        self._input_handler = on_input
        self._resize_handler = on_resize

        # Register atexit handler for cleanup on unexpected exit
        _active_terminal = self
        atexit.register(_cleanup_terminal)

        # Save current terminal settings
        if sys.stdin.isatty():
            self._old_settings = termios.tcgetattr(sys.stdin.fileno())
            # Enable raw mode
            tty.setraw(sys.stdin.fileno())

        # Enable bracketed paste mode
        sys.stdout.write("\x1b[?2004h")
        sys.stdout.flush()

        # Set up SIGWINCH handler for resize events
        self._old_sigwinch = signal.signal(signal.SIGWINCH, self._handle_sigwinch)

        self._running = True

    def stop(self) -> None:
        """Stop terminal and restore previous settings."""
        global _active_terminal

        if not self._running and self._old_settings is None:
            return  # Already stopped

        self._running = False

        # Restore terminal to clean state
        # Show cursor (in case we hid it)
        sys.stdout.write("\x1b[?25h")
        # Disable bracketed paste mode
        sys.stdout.write("\x1b[?2004l")
        # End synchronized output (in case we're in the middle of it)
        sys.stdout.write("\x1b[?2026l")
        # Reset all attributes (colors, bold, etc.)
        sys.stdout.write("\x1b[0m")
        # Ensure cursor is at start of a new line
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Restore terminal settings
        if self._old_settings is not None and sys.stdin.isatty():
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

        # Restore SIGWINCH handler
        if self._old_sigwinch is not None:
            signal.signal(signal.SIGWINCH, self._old_sigwinch)
            self._old_sigwinch = None

        self._input_handler = None
        self._resize_handler = None

        # Clear global reference and unregister atexit
        _active_terminal = None
        try:
            atexit.unregister(_cleanup_terminal)
        except Exception:
            pass  # May fail if already unregistered

    def write(self, data: str) -> None:
        """Write data to stdout."""
        sys.stdout.write(data)
        sys.stdout.flush()

    @property
    def columns(self) -> int:
        """Get terminal width."""
        size = os.get_terminal_size()
        return size.columns

    @property
    def rows(self) -> int:
        """Get terminal height."""
        size = os.get_terminal_size()
        return size.lines

    def hide_cursor(self) -> None:
        """Hide the cursor."""
        self.write("\x1b[?25l")

    def show_cursor(self) -> None:
        """Show the cursor."""
        self.write("\x1b[?25h")

    def clear_line(self) -> None:
        """Clear from cursor to end of line."""
        self.write("\x1b[K")

    def clear_from_cursor(self) -> None:
        """Clear from cursor to end of screen."""
        self.write("\x1b[J")

    def clear_screen(self) -> None:
        """Clear entire screen and move cursor to home (1,1)."""
        self.write("\x1b[2J\x1b[H")

    def _handle_sigwinch(self, signum: int, frame) -> None:
        """Handle terminal resize signal."""
        if self._resize_handler:
            self._resize_handler()

    def read_input(self) -> Optional[str]:
        """Read available input (non-blocking style for use with async).

        Returns None if no input available. Reads all available bytes
        to keep escape sequences together.
        """
        import select
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None

        # Read first byte
        result = sys.stdin.read(1)

        # If it's an escape, try to read the rest of the sequence
        if result == '\x1b':
            # Give a tiny bit of time for the rest of the sequence to arrive
            import time
            time.sleep(0.001)  # 1ms

            # Read all available bytes
            while select.select([sys.stdin], [], [], 0)[0]:
                result += sys.stdin.read(1)

        return result

    def run_external_editor(self, initial_content: str = "") -> Optional[str]:
        """Temporarily exit raw mode, run $EDITOR, and return edited content.

        Args:
            initial_content: Initial text to populate the editor with

        Returns:
            Edited content, or None if editor failed or user quit without saving
        """
        import tempfile
        import subprocess

        # Get editor from environment, default to vim
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vim"))

        # Create temp file with initial content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(initial_content)
            temp_path = f.name

        try:
            # Temporarily restore terminal to cooked mode
            if self._old_settings is not None and sys.stdin.isatty():
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)

            # Disable bracketed paste mode
            sys.stdout.write("\x1b[?2004l")
            # Show cursor
            sys.stdout.write("\x1b[?25h")
            # Clear screen for editor
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.flush()

            # Run editor
            result = subprocess.run([editor, temp_path])

            # Read edited content
            if result.returncode == 0:
                with open(temp_path, "r") as f:
                    content = f.read()
                return content.strip() if content.strip() else None
            return None

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

            # Restore raw mode
            if sys.stdin.isatty():
                tty.setraw(sys.stdin.fileno())

            # Re-enable bracketed paste mode
            sys.stdout.write("\x1b[?2004h")
            # Clear screen (TUI will redraw)
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.flush()

            # Trigger resize handler to force full redraw
            if self._resize_handler:
                self._resize_handler()
