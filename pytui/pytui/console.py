"""Console: unified output manager coordinating spinners with logging.

The problem: Spinner animations and log messages both write to stderr.
When they interleave, output gets corrupted. Console solves this by:
1. Owning the spinner lifecycle
2. Providing a logging handler that pauses the spinner before emitting

Usage:
    import logging
    from pytui import Console

    logger = logging.getLogger(__name__)
    console = Console()
    console.install_logging_handler(logger)

    with console.spinner("Provisioning..."):
        # Any logger.info/warning/error calls here will:
        # 1. Pause the spinner (clear its line)
        # 2. Print the log message on its own line
        # 3. Resume the spinner below
        result = do_stuff_that_logs()
    # Shows: ✓ Provisioning...

Three levels of granularity (Casey Muratori's rule: no API holes):
    - Low-level: pause_spinner(), resume_spinner(), write()
    - Mid-level: status() for one-off messages
    - High-level: spinner() context manager
"""

from __future__ import annotations

import itertools
import logging
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO

BRAILLE_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


class Console:
    """Unified output manager coordinating spinners with logging.

    Args:
        output: Stream to write to. Defaults to stderr (matches logging convention).
        spinner_interval: Seconds between spinner frame updates.
    """

    def __init__(
        self,
        output: TextIO = sys.stderr,
        spinner_interval: float = 0.08,
    ) -> None:
        self._output = output
        self._spinner_interval = spinner_interval
        self._lock = threading.Lock()

        # Active spinner state (immediate mode: only one at a time)
        self._spinner_message: str | None = None
        self._spinner_frame_iter: Iterator[str] | None = None
        self._spinner_stop: threading.Event | None = None
        self._spinner_thread: threading.Thread | None = None
        self._spinner_paused: bool = False
        self._spinner_start_time: float | None = None

        # Installed logging handlers (for cleanup)
        self._handlers: list[tuple[logging.Logger, logging.Handler]] = []

    # ─── Low-level API ─────────────────────────────────────────────────────

    def write(self, text: str) -> None:
        """Write text to output stream. Thread-safe."""
        with self._lock:
            self._output.write(text)
            self._output.flush()

    def write_line(self, text: str) -> None:
        """Write a complete line to output. Coordinates with active spinner."""
        with self._lock:
            if self._spinner_message is not None and not self._spinner_paused:
                # Clear spinner line, write message, redraw spinner
                self._output.write(f"\r\033[K{text}\n")
                frame = next(self._spinner_frame_iter) if self._spinner_frame_iter else "⠋"
                self._output.write(f"{frame} {self._spinner_message}")
                self._output.flush()
            else:
                self._output.write(f"{text}\n")
                self._output.flush()

    def pause_spinner(self) -> None:
        """Pause the spinner animation and clear its line.

        Used by ConsoleHandler before emitting log records.
        """
        with self._lock:
            if self._spinner_message is not None and not self._spinner_paused:
                self._spinner_paused = True
                self._output.write("\r\033[K")
                self._output.flush()

    def resume_spinner(self) -> None:
        """Resume the spinner animation.

        Used by ConsoleHandler after emitting log records.
        """
        with self._lock:
            if self._spinner_message is not None and self._spinner_paused:
                self._spinner_paused = False
                frame = next(self._spinner_frame_iter) if self._spinner_frame_iter else "⠋"
                self._output.write(f"{frame} {self._spinner_message}")
                self._output.flush()

    # ─── Mid-level API ─────────────────────────────────────────────────────

    def status(self, text: str, success: bool = True) -> None:
        """Print a status line with ✓ or ✗ prefix."""
        mark = "✓" if success else "✗"
        self.write_line(f"{mark} {text}")

    # ─── High-level API ────────────────────────────────────────────────────

    @contextmanager
    def spinner(self, message: str) -> Iterator[SpinnerHandle]:
        """Context manager for spinner animation.

        Any log messages emitted inside the block will be coordinated:
        the spinner pauses, the message prints, the spinner resumes.

        Args:
            message: Text to display next to the spinner.

        Yields:
            SpinnerHandle with update() method to change the message.

        Example:
            with console.spinner("Loading...") as s:
                s.update("Loading... 50%")
                do_work()
            # Shows: ✓ Loading... 50%
        """
        assert self._spinner_message is None, "Cannot nest spinners"

        handle = SpinnerHandle(self)
        self._spinner_message = message
        self._spinner_frame_iter = itertools.cycle(BRAILLE_FRAMES)
        self._spinner_stop = threading.Event()
        self._spinner_paused = False
        self._spinner_start_time = time.time()

        self._spinner_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spinner_thread.start()

        exc_info: BaseException | None = None
        try:
            yield handle
        except BaseException as e:
            exc_info = e
            raise
        finally:
            # Stop spinner thread
            assert self._spinner_stop is not None
            self._spinner_stop.set()
            if self._spinner_thread is not None:
                self._spinner_thread.join()
                self._spinner_thread = None

            # Print final status with elapsed time
            final_message = self._spinner_message
            success = exc_info is None
            mark = "✓" if success else "✗"
            elapsed = self._format_elapsed(time.time() - self._spinner_start_time)

            with self._lock:
                self._output.write(f"\r\033[K{mark} {final_message} ({elapsed})\n")
                self._output.flush()

            # Clear spinner state
            self._spinner_message = None
            self._spinner_frame_iter = None
            self._spinner_stop = None
            self._spinner_paused = False
            self._spinner_start_time = None

    def _spin_loop(self) -> None:
        """Background thread that animates the spinner."""
        assert self._spinner_stop is not None
        assert self._spinner_frame_iter is not None
        assert self._spinner_start_time is not None

        while not self._spinner_stop.is_set():
            with self._lock:
                if not self._spinner_paused and self._spinner_message is not None:
                    frame = next(self._spinner_frame_iter)
                    elapsed = self._format_elapsed(time.time() - self._spinner_start_time)
                    self._output.write(f"\r\033[K{frame} {self._spinner_message} ({elapsed})")
                    self._output.flush()
            time.sleep(self._spinner_interval)

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Format elapsed time as human-readable string (e.g., '1m7s', '45s')."""
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        secs = seconds % 60
        if minutes < 60:
            return f"{minutes}m{secs}s" if secs else f"{minutes}m"
        hours = minutes // 60
        mins = minutes % 60
        if secs:
            return f"{hours}h{mins}m{secs}s"
        elif mins:
            return f"{hours}h{mins}m"
        else:
            return f"{hours}h"

    # ─── Logging Integration ───────────────────────────────────────────────

    def install_logging_handler(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.DEBUG,
        fmt: str = "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt: str = "%H:%M:%S",
    ) -> logging.Handler:
        """Install a logging handler that coordinates with the spinner.

        Args:
            logger: Logger to attach to. Defaults to root logger.
            level: Minimum log level to handle.
            fmt: Log message format string.
            datefmt: Date format for %(asctime)s.

        Returns:
            The installed handler (for advanced use cases).
        """
        if logger is None:
            logger = logging.getLogger()

        handler = ConsoleHandler(self)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(handler)

        self._handlers.append((logger, handler))
        return handler

    def remove_logging_handlers(self) -> None:
        """Remove all logging handlers installed by this Console."""
        for logger, handler in self._handlers:
            logger.removeHandler(handler)
        self._handlers.clear()


class SpinnerHandle:
    """Handle returned by Console.spinner() for updating the message."""

    def __init__(self, console: Console) -> None:
        self._console = console

    def update(self, message: str) -> None:
        """Update the spinner message."""
        with self._console._lock:
            self._console._spinner_message = message


class ConsoleHandler(logging.Handler):
    """Logging handler that coordinates with Console's spinner.

    When a log record is emitted:
    1. Pause the spinner (clear its line)
    2. Print the formatted log message
    3. Resume the spinner on a new line
    """

    def __init__(self, console: Console) -> None:
        super().__init__()
        self._console = console

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._console.pause_spinner()
            self._console.write_line(msg)
            self._console.resume_spinner()
        except Exception:
            self.handleError(record)
