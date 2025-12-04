"""
Terminal abstraction for TUI.

Provides a clean interface for terminal I/O, cursor control, and raw mode handling.
"""

from __future__ import annotations

import sys
import tty
import termios
import signal
import os
from typing import Callable, Protocol, Optional, List


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
        self._input_handler = on_input
        self._resize_handler = on_resize

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
        self._running = False

        # Disable bracketed paste mode
        sys.stdout.write("\x1b[?2004l")
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

        Returns None if no input available.
        """
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None
