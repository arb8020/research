"""Terminal abstraction - raw mode, cursor control, input handling.

Borrowed from pyvimdiff with minor modifications.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import termios
import tty
from collections.abc import Callable

_active_terminal: Terminal | None = None


def _cleanup_terminal() -> None:
    global _active_terminal
    if _active_terminal is not None:
        _active_terminal.stop()


class Terminal:
    """Terminal with raw mode and alternate screen buffer support."""

    def __init__(self, use_alternate_screen: bool = True) -> None:
        self._old_settings: list | None = None
        self._input_handler: Callable[[str], None] | None = None
        self._resize_handler: Callable[[], None] | None = None
        self._old_sigwinch: signal.Handlers | None = None
        self._running = False
        self._tty_fd: int | None = None
        self._use_alternate_screen = use_alternate_screen

    def start(
        self,
        on_input: Callable[[str], None],
        on_resize: Callable[[], None],
    ) -> None:
        global _active_terminal

        self._input_handler = on_input
        self._resize_handler = on_resize

        _active_terminal = self
        atexit.register(_cleanup_terminal)

        # Open /dev/tty for keyboard input
        try:
            self._tty_fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            self._old_settings = termios.tcgetattr(self._tty_fd)
            tty.setraw(self._tty_fd)
        except (OSError, termios.error):
            self._tty_fd = None
            if sys.stdin.isatty():
                self._old_settings = termios.tcgetattr(sys.stdin.fileno())
                tty.setraw(sys.stdin.fileno())

        # Enter alternate screen buffer (like vim/less)
        if self._use_alternate_screen:
            sys.stdout.write("\x1b[?1049h")  # Enter alternate screen
        sys.stdout.write("\x1b[?25l")  # Hide cursor
        sys.stdout.flush()

        self._old_sigwinch = signal.signal(signal.SIGWINCH, self._handle_sigwinch)
        self._running = True

    def stop(self) -> None:
        global _active_terminal

        if not self._running and self._old_settings is None:
            return

        self._running = False

        # Restore terminal
        sys.stdout.write("\x1b[?25h")  # Show cursor
        if self._use_alternate_screen:
            sys.stdout.write("\x1b[?1049l")  # Exit alternate screen
        sys.stdout.write("\x1b[0m")  # Reset attributes
        sys.stdout.flush()

        if self._old_settings is not None:
            if self._tty_fd is not None:
                termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, self._old_settings)
                os.close(self._tty_fd)
                self._tty_fd = None
            elif sys.stdin.isatty():
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

        if self._old_sigwinch is not None:
            signal.signal(signal.SIGWINCH, self._old_sigwinch)
            self._old_sigwinch = None

        _active_terminal = None
        try:
            atexit.unregister(_cleanup_terminal)
        except Exception:
            pass

    def write(self, data: str) -> None:
        sys.stdout.write(data)
        sys.stdout.flush()

    @property
    def columns(self) -> int:
        return os.get_terminal_size().columns

    @property
    def rows(self) -> int:
        return os.get_terminal_size().lines

    def clear_screen(self) -> None:
        self.write("\x1b[2J\x1b[H")

    def move_cursor(self, row: int, col: int) -> None:
        """Move cursor to position (1-indexed)."""
        self.write(f"\x1b[{row};{col}H")

    def _handle_sigwinch(self, signum: int, frame: object) -> None:
        if self._resize_handler:
            self._resize_handler()

    def read_input(self) -> str | None:
        """Read available input (non-blocking)."""
        import select

        if self._tty_fd is not None:
            if not select.select([self._tty_fd], [], [], 0)[0]:
                return None
            result = os.read(self._tty_fd, 1).decode("utf-8", errors="replace")
            if result == "\x1b":
                import time

                time.sleep(0.001)
                while select.select([self._tty_fd], [], [], 0)[0]:
                    result += os.read(self._tty_fd, 1).decode("utf-8", errors="replace")
            return result
        else:
            if not select.select([sys.stdin], [], [], 0)[0]:
                return None
            result = sys.stdin.read(1)
            if result == "\x1b":
                import time

                time.sleep(0.001)
                while select.select([sys.stdin], [], [], 0)[0]:
                    result += sys.stdin.read(1)
            return result
