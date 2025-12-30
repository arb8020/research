"""
Terminal abstraction for TUI.

Provides a clean interface for terminal I/O, cursor control, and raw mode handling.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import termios
import tty
from collections.abc import Callable
from types import FrameType
from typing import Protocol

# Global reference for atexit cleanup
_active_terminal: ProcessTerminal | None = None
_active_session_id: str | None = None


def set_active_session_id(session_id: str | None) -> None:
    """Set the active session ID for crash reporting."""
    global _active_session_id
    _active_session_id = session_id


def _cleanup_terminal() -> None:
    """Atexit handler to restore terminal state and print session ID."""
    global _active_terminal, _active_session_id
    if _active_terminal is not None:
        _active_terminal.stop()

    # Print session ID on crash/exit so user can resume
    if _active_session_id:
        print(f"\nSession: {_active_session_id}")
        print(f"Resume with: --session {_active_session_id}")


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
        self._old_settings: list | None = None
        self._input_handler: Callable[[str], None] | None = None
        self._resize_handler: Callable[[], None] | None = None
        self._old_sigwinch = None
        self._running = False
        self._tty_fd: int | None = None  # File descriptor for /dev/tty

    def start(self, on_input: Callable[[str], None], on_resize: Callable[[], None]) -> None:
        """Start terminal in raw mode with input/resize handlers."""
        global _active_terminal

        self._input_handler = on_input
        self._resize_handler = on_resize

        # Register atexit handler for cleanup on unexpected exit
        _active_terminal = self
        atexit.register(_cleanup_terminal)

        # Always try to open /dev/tty for terminal control
        # This allows keyboard input even when stdin is piped
        try:
            self._tty_fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            # Save terminal settings from /dev/tty
            self._old_settings = termios.tcgetattr(self._tty_fd)
            # Enable raw mode on /dev/tty
            tty.setraw(self._tty_fd)
        except (OSError, termios.error):
            # Fall back to sys.stdin if /dev/tty unavailable (e.g., no controlling terminal)
            self._tty_fd = None
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
        if self._old_settings is not None:
            if self._tty_fd is not None:
                # Restore /dev/tty settings
                termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, self._old_settings)
                os.close(self._tty_fd)
                self._tty_fd = None
            elif sys.stdin.isatty():
                # Restore stdin settings
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

    def _handle_sigwinch(self, signum: int, frame: FrameType | None) -> None:
        """Handle terminal resize signal."""
        if self._resize_handler:
            self._resize_handler()

    def read_input(self) -> str | None:
        """Read available input (non-blocking style for use with async).

        Returns None if no input available. Reads all available bytes
        to keep escape sequences together.
        """
        import select

        # Read from /dev/tty if available, otherwise fall back to stdin
        if self._tty_fd is not None:
            # Check if /dev/tty has data
            if not select.select([self._tty_fd], [], [], 0)[0]:
                return None

            # Read first byte
            result = os.read(self._tty_fd, 1).decode("utf-8", errors="replace")

            # If it's an escape, try to read the rest of the sequence
            if result == "\x1b":
                import time

                time.sleep(0.001)  # 1ms

                # Read all available bytes
                while select.select([self._tty_fd], [], [], 0)[0]:
                    result += os.read(self._tty_fd, 1).decode("utf-8", errors="replace")

            return result
        else:
            # Fall back to stdin
            if not select.select([sys.stdin], [], [], 0)[0]:
                return None

            # Read first byte
            result = sys.stdin.read(1)

            # If it's an escape, try to read the rest of the sequence
            if result == "\x1b":
                # Give a tiny bit of time for the rest of the sequence to arrive
                import time

                time.sleep(0.001)  # 1ms

                # Read all available bytes
                while select.select([sys.stdin], [], [], 0)[0]:
                    result += sys.stdin.read(1)

            return result

    def run_external_editor(self, initial_content: str = "") -> str | None:
        """Temporarily exit raw mode, run $EDITOR, and return edited content.

        Args:
            initial_content: Initial text to populate the editor with

        Returns:
            Edited content, or None if editor failed or user quit without saving
        """
        import subprocess
        import tempfile

        # Get editor from environment, default to vim
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vim"))

        # Create temp file with initial content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(initial_content)
            temp_path = f.name

        try:
            # Temporarily restore terminal to cooked mode
            if self._old_settings is not None:
                if self._tty_fd is not None:
                    # Restore /dev/tty settings for editor
                    termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, self._old_settings)
                elif sys.stdin.isatty():
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)

            # Disable bracketed paste mode
            sys.stdout.write("\x1b[?2004l")
            # Show cursor
            sys.stdout.write("\x1b[?25h")
            # Clear screen for editor
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.flush()

            # Run editor - use /dev/tty for stdin/stdout to ensure editor works
            # even when rollouts stdin is piped
            with open("/dev/tty") as tty_in, open("/dev/tty", "w") as tty_out:
                result = subprocess.run(
                    [editor, temp_path],
                    stdin=tty_in,
                    stdout=tty_out,
                    stderr=tty_out,
                )

            # Read edited content
            if result.returncode == 0:
                with open(temp_path) as f:
                    content = f.read()
                return content.strip() if content.strip() else None
            return None

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

            # Restore raw mode on the correct fd
            if self._tty_fd is not None:
                tty.setraw(self._tty_fd)
            elif sys.stdin.isatty():
                tty.setraw(sys.stdin.fileno())

            # Re-enable bracketed paste mode
            sys.stdout.write("\x1b[?2004h")
            # Hide cursor during redraw
            sys.stdout.write("\x1b[?25l")
            # Clear screen (TUI will redraw)
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.flush()

            # Trigger resize handler to force full redraw
            # Note: This will cause the TUI to re-render all components
            if self._resize_handler:
                self._resize_handler()
