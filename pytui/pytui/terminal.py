"""Terminal abstraction - raw mode, cursor control, input handling.

Unified terminal layer for all TUI apps. Handles:
- Raw mode (termios) with /dev/tty fallback
- Non-blocking keyboard input with escape sequence buffering
- SIGWINCH resize handling
- Alternate screen buffer (optional)
- Bracketed paste mode (optional)
- Atexit cleanup for crash safety
- External editor support (temporarily restore cooked mode)
"""

from __future__ import annotations

import atexit
import os
import select
import signal
import sys
import termios
import time
import tty
from collections.abc import Callable
from types import FrameType
from typing import Any, Protocol

# Global reference for atexit cleanup
_active_terminal: Terminal | None = None
_cleanup_done: bool = False


def _cleanup_terminal() -> None:
    """Atexit handler to restore terminal state."""
    global _active_terminal, _cleanup_done

    if _cleanup_done:
        return
    _cleanup_done = True

    if _active_terminal is not None:
        _active_terminal.stop()

    # Fallback: run stty sane to ensure terminal is usable
    import subprocess

    try:
        subprocess.run(["stty", "sane"], stdin=open("/dev/tty"), check=False)
    except Exception:
        pass


class TerminalProtocol(Protocol):
    """Protocol for terminal implementations."""

    def start(self, on_input: Callable[[str], None], on_resize: Callable[[], None]) -> None: ...
    def stop(self) -> None: ...
    def write(self, data: str) -> None: ...

    @property
    def columns(self) -> int: ...

    @property
    def rows(self) -> int: ...

    def hide_cursor(self) -> None: ...
    def show_cursor(self) -> None: ...
    def clear_line(self) -> None: ...
    def clear_from_cursor(self) -> None: ...
    def clear_screen(self) -> None: ...


class Terminal:
    """Real terminal using /dev/tty with raw mode support.

    Features:
    - /dev/tty for input (works even when stdin is piped)
    - Non-blocking reads with escape sequence buffering
    - Optional alternate screen buffer
    - Optional bracketed paste mode
    - Atexit cleanup for crash recovery
    """

    def __init__(
        self,
        *,
        alternate_screen: bool = False,
        use_alternate_screen: bool | None = None,  # Compat alias
        bracketed_paste: bool = False,
    ) -> None:
        self._old_settings: list | None = None
        self._input_handler: Callable[[str], None] | None = None
        self._resize_handler: Callable[[], None] | None = None
        self._old_sigwinch: Any = None
        self._running = False
        self._tty_fd: int | None = None
        # Accept both alternate_screen and use_alternate_screen (compat)
        if use_alternate_screen is not None:
            self._alternate_screen = use_alternate_screen
        else:
            self._alternate_screen = alternate_screen
        self._bracketed_paste = bracketed_paste

    def start(self, on_input: Callable[[str], None], on_resize: Callable[[], None]) -> None:
        """Start terminal in raw mode with input/resize handlers."""
        global _active_terminal, _cleanup_done

        self._input_handler = on_input
        self._resize_handler = on_resize

        _active_terminal = self
        _cleanup_done = False
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

        # Enter alternate screen buffer
        if self._alternate_screen:
            sys.stdout.write("\x1b[?1049h")

        # Enable bracketed paste mode
        if self._bracketed_paste:
            sys.stdout.write("\x1b[?2004h")

        sys.stdout.flush()

        # SIGWINCH for resize events
        self._old_sigwinch = signal.signal(signal.SIGWINCH, self._handle_sigwinch)
        self._running = True

    def stop(self) -> None:
        """Stop terminal and restore previous settings."""
        global _active_terminal, _cleanup_done

        if not self._running and self._old_settings is None:
            return

        self._running = False
        _cleanup_done = True

        # Restore terminal to clean state
        sys.stdout.write("\x1b[?25h")  # Show cursor

        if self._bracketed_paste:
            sys.stdout.write("\x1b[?2004l")

        # End synchronized output (in case we're mid-render)
        sys.stdout.write("\x1b[?2026l")
        sys.stdout.write("\x1b[0m")  # Reset attributes

        if self._alternate_screen:
            sys.stdout.write("\x1b[?1049l")
        else:
            sys.stdout.write("\n")

        sys.stdout.flush()

        # Restore terminal settings
        if self._old_settings is not None:
            if self._tty_fd is not None:
                termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, self._old_settings)
                os.close(self._tty_fd)
                self._tty_fd = None
            elif sys.stdin.isatty():
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

        # Restore SIGWINCH handler
        if self._old_sigwinch is not None:
            signal.signal(signal.SIGWINCH, self._old_sigwinch)
            self._old_sigwinch = None

        self._input_handler = None
        self._resize_handler = None

        _active_terminal = None
        try:
            atexit.unregister(_cleanup_terminal)
        except Exception:
            pass

    def write(self, data: str) -> None:
        """Write data to stdout."""
        sys.stdout.write(data)
        sys.stdout.flush()

    @property
    def columns(self) -> int:
        """Terminal width in columns."""
        return os.get_terminal_size().columns

    @property
    def rows(self) -> int:
        """Terminal height in rows."""
        return os.get_terminal_size().lines

    def hide_cursor(self) -> None:
        self.write("\x1b[?25l")

    def show_cursor(self) -> None:
        self.write("\x1b[?25h")

    def clear_line(self) -> None:
        self.write("\x1b[K")

    def clear_from_cursor(self) -> None:
        self.write("\x1b[J")

    def clear_screen(self) -> None:
        self.write("\x1b[2J\x1b[H")

    def move_cursor(self, row: int, col: int) -> None:
        """Move cursor to position (1-indexed)."""
        self.write(f"\x1b[{row};{col}H")

    def read_input(self) -> str | None:
        """Read available input (non-blocking).

        Returns None if no input available. Reads all available bytes
        to keep escape sequences together.
        """
        if self._tty_fd is not None:
            if not select.select([self._tty_fd], [], [], 0)[0]:
                return None
            result = os.read(self._tty_fd, 1).decode("utf-8", errors="replace")
            if result == "\x1b":
                time.sleep(0.001)  # 1ms for escape sequence bytes
                while select.select([self._tty_fd], [], [], 0)[0]:
                    result += os.read(self._tty_fd, 1).decode("utf-8", errors="replace")
            return result
        else:
            if not select.select([sys.stdin], [], [], 0)[0]:
                return None
            result = sys.stdin.read(1)
            if result == "\x1b":
                time.sleep(0.001)
                while select.select([sys.stdin], [], [], 0)[0]:
                    result += sys.stdin.read(1)
            return result

    def run_external_editor(self, initial_content: str = "") -> str | None:
        """Temporarily exit raw mode, run $EDITOR, return edited content."""
        import subprocess
        import tempfile

        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vim"))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(initial_content)
            temp_path = f.name

        try:
            # Restore cooked mode
            if self._old_settings is not None:
                if self._tty_fd is not None:
                    termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, self._old_settings)
                elif sys.stdin.isatty():
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)

            if self._bracketed_paste:
                sys.stdout.write("\x1b[?2004l")
            sys.stdout.write("\x1b[?25h")  # Show cursor
            sys.stdout.write("\x1b[2J\x1b[H")  # Clear screen
            sys.stdout.flush()

            with open("/dev/tty") as tty_in, open("/dev/tty", "w") as tty_out:
                result = subprocess.run(
                    [editor, temp_path],
                    stdin=tty_in,
                    stdout=tty_out,
                    stderr=tty_out,
                )

            if result.returncode == 0:
                with open(temp_path) as f:
                    content = f.read()
                return content.strip() if content.strip() else None
            return None

        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

            # Restore raw mode
            if self._tty_fd is not None:
                tty.setraw(self._tty_fd)
            elif sys.stdin.isatty():
                tty.setraw(sys.stdin.fileno())

            if self._bracketed_paste:
                sys.stdout.write("\x1b[?2004h")
            sys.stdout.write("\x1b[?25l")
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.flush()

            if self._resize_handler:
                self._resize_handler()

    def _handle_sigwinch(self, signum: int, frame: FrameType | None) -> None:
        if self._resize_handler:
            self._resize_handler()
