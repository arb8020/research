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

from .input import InputParser

# ANSI escape sequences — named for readability and grep-ability
ALT_SCREEN_ON = "\x1b[?1049h"
ALT_SCREEN_OFF = "\x1b[?1049l"
BRACKETED_PASTE_ON = "\x1b[?2004h"
BRACKETED_PASTE_OFF = "\x1b[?2004l"
SYNC_OUTPUT_ON = "\x1b[?2026h"
SYNC_OUTPUT_OFF = "\x1b[?2026l"
CURSOR_SHOW = "\x1b[?25h"
CURSOR_HIDE = "\x1b[?25l"
CLEAR_SCREEN_HOME = "\x1b[2J\x1b[H"
CLEAR_LINE = "\x1b[K"
CLEAR_LINE_FULL = "\x1b[2K"
CLEAR_TO_END = "\x1b[J"
RESET_ATTRS = "\x1b[0m"

# Mouse tracking (SGR extended mode - handles coordinates > 223)
MOUSE_ON = "\x1b[?1000h\x1b[?1006h"  # Enable button events + SGR encoding
MOUSE_OFF = "\x1b[?1000l\x1b[?1006l"

# Focus reporting
FOCUS_ON = "\x1b[?1004h"
FOCUS_OFF = "\x1b[?1004l"

# Known escape sequences - ported from bubbletea/key.go
# Maps sequence -> key name for reliable detection
KNOWN_SEQUENCES: dict[str, str] = {
    # Arrow keys
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1b[C": "right",
    "\x1b[D": "left",
    "\x1b[1;2A": "shift+up",
    "\x1b[1;2B": "shift+down",
    "\x1b[1;2C": "shift+right",
    "\x1b[1;2D": "shift+left",
    "\x1b[OA": "shift+up",
    "\x1b[OB": "shift+down",  # DECCKM
    "\x1b[OC": "shift+right",
    "\x1b[OD": "shift+left",
    "\x1b[a": "shift+up",
    "\x1b[b": "shift+down",  # urxvt
    "\x1b[c": "shift+right",
    "\x1b[d": "shift+left",
    "\x1b[1;3A": "alt+up",
    "\x1b[1;3B": "alt+down",
    "\x1b[1;3C": "alt+right",
    "\x1b[1;3D": "alt+left",
    "\x1b[1;4A": "alt+shift+up",
    "\x1b[1;4B": "alt+shift+down",
    "\x1b[1;4C": "alt+shift+right",
    "\x1b[1;4D": "alt+shift+left",
    "\x1b[1;5A": "ctrl+up",
    "\x1b[1;5B": "ctrl+down",
    "\x1b[1;5C": "ctrl+right",
    "\x1b[1;5D": "ctrl+left",
    "\x1b[Oa": "alt+ctrl+up",
    "\x1b[Ob": "alt+ctrl+down",  # urxvt
    "\x1b[Oc": "alt+ctrl+right",
    "\x1b[Od": "alt+ctrl+left",
    "\x1b[1;6A": "ctrl+shift+up",
    "\x1b[1;6B": "ctrl+shift+down",
    "\x1b[1;6C": "ctrl+shift+right",
    "\x1b[1;6D": "ctrl+shift+left",
    "\x1b[1;7A": "alt+ctrl+up",
    "\x1b[1;7B": "alt+ctrl+down",
    "\x1b[1;7C": "alt+ctrl+right",
    "\x1b[1;7D": "alt+ctrl+left",
    "\x1b[1;8A": "alt+ctrl+shift+up",
    "\x1b[1;8B": "alt+ctrl+shift+down",
    "\x1b[1;8C": "alt+ctrl+shift+right",
    "\x1b[1;8D": "alt+ctrl+shift+left",
    # Application mode arrows (powershell, etc)
    "\x1bOA": "up",
    "\x1bOB": "down",
    "\x1bOC": "right",
    "\x1bOD": "left",
    # Misc keys
    "\x1b[Z": "shift+tab",
    "\x1b[2~": "insert",
    "\x1b[3;2~": "alt+insert",
    "\x1b[3~": "delete",
    "\x1b[3;3~": "alt+delete",
    # Page Up/Down
    "\x1b[5~": "pageup",
    "\x1b[5;3~": "alt+pageup",
    "\x1b[5;5~": "ctrl+pageup",
    "\x1b[5^": "ctrl+pageup",  # urxvt
    "\x1b[5;7~": "alt+ctrl+pageup",
    "\x1b[6~": "pagedown",
    "\x1b[6;3~": "alt+pagedown",
    "\x1b[6;5~": "ctrl+pagedown",
    "\x1b[6^": "ctrl+pagedown",  # urxvt
    "\x1b[6;7~": "alt+ctrl+pagedown",
    # Home/End - xterm style
    "\x1b[1~": "home",
    "\x1b[H": "home",
    "\x1b[1;3H": "alt+home",
    "\x1b[1;5H": "ctrl+home",
    "\x1b[1;7H": "alt+ctrl+home",
    "\x1b[1;2H": "shift+home",
    "\x1b[1;4H": "alt+shift+home",
    "\x1b[1;6H": "ctrl+shift+home",
    "\x1b[1;8H": "alt+ctrl+shift+home",
    "\x1b[4~": "end",
    "\x1b[F": "end",
    "\x1b[1;3F": "alt+end",
    "\x1b[1;5F": "ctrl+end",
    "\x1b[1;7F": "alt+ctrl+end",
    "\x1b[1;2F": "shift+end",
    "\x1b[1;4F": "alt+shift+end",
    "\x1b[1;6F": "ctrl+shift+end",
    "\x1b[1;8F": "alt+ctrl+shift+end",
    # Home/End - urxvt style
    "\x1b[7~": "home",
    "\x1b[7^": "ctrl+home",
    "\x1b[7$": "shift+home",
    "\x1b[7@": "ctrl+shift+home",
    "\x1b[8~": "end",
    "\x1b[8^": "ctrl+end",
    "\x1b[8$": "shift+end",
    "\x1b[8@": "ctrl+shift+end",
    # Function keys - linux console
    "\x1b[[A": "f1",
    "\x1b[[B": "f2",
    "\x1b[[C": "f3",
    "\x1b[[D": "f4",
    "\x1b[[E": "f5",
    # Function keys - vt100/xterm
    "\x1bOP": "f1",
    "\x1bOQ": "f2",
    "\x1bOR": "f3",
    "\x1bOS": "f4",
    "\x1b[1;3P": "alt+f1",
    "\x1b[1;3Q": "alt+f2",
    "\x1b[1;3R": "alt+f3",
    "\x1b[1;3S": "alt+f4",
    # Function keys - urxvt style
    "\x1b[11~": "f1",
    "\x1b[12~": "f2",
    "\x1b[13~": "f3",
    "\x1b[14~": "f4",
    "\x1b[15~": "f5",
    "\x1b[15;3~": "alt+f5",
    "\x1b[17~": "f6",
    "\x1b[18~": "f7",
    "\x1b[19~": "f8",
    "\x1b[20~": "f9",
    "\x1b[21~": "f10",
    "\x1b[17;3~": "alt+f6",
    "\x1b[18;3~": "alt+f7",
    "\x1b[19;3~": "alt+f8",
    "\x1b[20;3~": "alt+f9",
    "\x1b[21;3~": "alt+f10",
    "\x1b[23~": "f11",
    "\x1b[24~": "f12",
    "\x1b[23;3~": "alt+f11",
    "\x1b[24;3~": "alt+f12",
    # F13-F20
    "\x1b[1;2P": "f13",
    "\x1b[1;2Q": "f14",
    "\x1b[25~": "f13",
    "\x1b[26~": "f14",
    "\x1b[25;3~": "alt+f13",
    "\x1b[26;3~": "alt+f14",
    "\x1b[1;2R": "f15",
    "\x1b[1;2S": "f16",
    "\x1b[28~": "f15",
    "\x1b[29~": "f16",
    "\x1b[28;3~": "alt+f15",
    "\x1b[29;3~": "alt+f16",
    "\x1b[15;2~": "f17",
    "\x1b[17;2~": "f18",
    "\x1b[18;2~": "f19",
    "\x1b[19;2~": "f20",
    "\x1b[31~": "f17",
    "\x1b[32~": "f18",
    "\x1b[33~": "f19",
    "\x1b[34~": "f20",
    # Bracketed paste markers
    "\x1b[200~": "paste_start",
    "\x1b[201~": "paste_end",
    # Focus events
    "\x1b[I": "focus",
    "\x1b[O": "blur",
}

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
        mouse: bool = False,
    ) -> None:
        self._old_settings: list | None = None
        self._input_handler: Callable[[str], None] | None = None
        self._resize_handler: Callable[[], None] | None = None
        self._old_sigwinch: Any = None
        self._running = False
        self._tty_fd: int | None = None
        self._input_buffer: str = ""  # Buffer for multi-byte reads
        self._message_parser = InputParser()
        self._pending_esc_deadline: float | None = None
        self._esc_timeout_s: float = 0.030
        # Accept both alternate_screen and use_alternate_screen (compat)
        if use_alternate_screen is not None:
            self._alternate_screen = use_alternate_screen
        else:
            self._alternate_screen = alternate_screen
        self._bracketed_paste = bracketed_paste
        self._mouse = mouse

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
            sys.stdout.write(ALT_SCREEN_ON)

        # Enable bracketed paste mode
        if self._bracketed_paste:
            sys.stdout.write(BRACKETED_PASTE_ON)

        # Enable mouse tracking
        if self._mouse:
            sys.stdout.write(MOUSE_ON)

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
        sys.stdout.write(CURSOR_SHOW)

        if self._bracketed_paste:
            sys.stdout.write(BRACKETED_PASTE_OFF)

        if self._mouse:
            sys.stdout.write(MOUSE_OFF)

        # End synchronized output (in case we're mid-render)
        sys.stdout.write(SYNC_OUTPUT_OFF)
        sys.stdout.write(RESET_ATTRS)

        if self._alternate_screen:
            sys.stdout.write(ALT_SCREEN_OFF)
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
        self.write(CURSOR_HIDE)

    def show_cursor(self) -> None:
        self.write(CURSOR_SHOW)

    def clear_line(self) -> None:
        self.write(CLEAR_LINE)

    def clear_from_cursor(self) -> None:
        self.write(CLEAR_TO_END)

    def clear_screen(self) -> None:
        self.write(CLEAR_SCREEN_HOME)

    def move_cursor(self, row: int, col: int) -> None:
        """Move cursor to position (1-indexed)."""
        self.write(f"\x1b[{row};{col}H")

    def read_input(self) -> str | None:
        """Read available input (non-blocking).

        Returns None if no input available. Uses buffered reads like bubbletea
        to reduce syscall overhead (read up to 256 bytes at once).

        Escape sequences are detected by looking for terminators (letters, ~).
        Returns one logical input at a time (single char or complete escape sequence).
        """
        fd = self._tty_fd if self._tty_fd is not None else sys.stdin.fileno()

        # Disambiguate bare Escape vs. escape sequences without blocking.
        # If we saw a lone ESC previously, wait briefly for more bytes to arrive.
        if self._pending_esc_deadline is not None:
            if len(self._input_buffer) > 1:
                self._pending_esc_deadline = None
            else:
                if select.select([fd], [], [], 0)[0]:
                    more = os.read(fd, 256).decode("utf-8", errors="replace")
                    if more:
                        self._input_buffer += more
                    self._pending_esc_deadline = None
                else:
                    if time.monotonic() >= self._pending_esc_deadline:
                        self._pending_esc_deadline = None
                        if self._input_buffer.startswith("\x1b"):
                            self._input_buffer = self._input_buffer[1:]
                        return "\x1b"
                    return None

        # Check buffer first
        if not self._input_buffer:
            if not select.select([fd], [], [], 0)[0]:
                return None
            # Read up to 256 bytes at once (like bubbletea) to reduce syscalls
            data = os.read(fd, 256).decode("utf-8", errors="replace")
            if not data:
                return None
            self._input_buffer = data

        # Extract one logical input from buffer
        buf = self._input_buffer

        if buf[0] != "\x1b":
            # Regular character
            self._input_buffer = buf[1:]
            return buf[0]

        # Escape sequence - find the end
        if len(buf) == 1:
            if self._pending_esc_deadline is None:
                self._pending_esc_deadline = time.monotonic() + self._esc_timeout_s
            return None

        # Check against known sequences first (O(1) lookup, longest match first)
        for length in range(min(len(buf), 10), 1, -1):  # Check longest first
            candidate = buf[:length]
            if candidate in KNOWN_SEQUENCES:
                self._input_buffer = buf[length:]
                return candidate

        # Fall back to heuristic: look for sequence terminator
        for i in range(1, len(buf)):
            c = buf[i]
            # CSI sequences end with letter, function keys with ~
            if c.isalpha() or c == "~":
                seq = buf[: i + 1]
                self._input_buffer = buf[i + 1 :]
                return seq

        # No terminator found - if buffer is small, wait for more
        if len(buf) < 12:
            deadline = time.time() + 0.050  # 50ms max
            while time.time() < deadline:
                if select.select([fd], [], [], 0.001)[0]:
                    more = os.read(fd, 256).decode("utf-8", errors="replace")
                    if more:
                        buf = buf + more
                        self._input_buffer = buf
                        # Check known sequences
                        for length in range(min(len(buf), 10), 1, -1):
                            candidate = buf[:length]
                            if candidate in KNOWN_SEQUENCES:
                                self._input_buffer = buf[length:]
                                return candidate
                        # Check for terminator
                        for i in range(1, len(buf)):
                            c = buf[i]
                            if c.isalpha() or c == "~":
                                seq = buf[: i + 1]
                                self._input_buffer = buf[i + 1 :]
                                return seq
                else:
                    break

        # Still no terminator - return bare ESC, keep rest in buffer
        self._input_buffer = buf[1:]
        return buf[0]

    def read_message(self) -> object | None:
        """Read a higher-level input message (KeyPress, PasteEvent, etc.).

        Returns None if no input is available. Bracketed paste is grouped into
        a single message so UI components don't need to parse paste markers.
        """
        while True:
            key = self.read_input()
            if key is None:
                return None
            msg = self._message_parser.parse(key)
            if msg is None:
                continue
            return msg

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
                sys.stdout.write(BRACKETED_PASTE_OFF)
            sys.stdout.write(CURSOR_SHOW)
            sys.stdout.write(CLEAR_SCREEN_HOME)
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
                sys.stdout.write(BRACKETED_PASTE_ON)
            sys.stdout.write(CURSOR_HIDE)
            sys.stdout.write(CLEAR_SCREEN_HOME)
            sys.stdout.flush()

            if self._resize_handler:
                self._resize_handler()

    def _handle_sigwinch(self, signum: int, frame: FrameType | None) -> None:
        if self._resize_handler:
            self._resize_handler()
