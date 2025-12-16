"""Intercept print() calls and redirect to logging.

Tiger Style: Explicit, bounded, fail-fast.
Sean Goedecke: Use stdlib patterns (sys.stdout override).

Usage:
    from shared.print_interceptor import intercept_prints
    import logging

    logger = logging.getLogger(__name__)

    # Redirect all print() calls to logger.info()
    with intercept_prints(logger):
        print("This becomes logger.info()")
        third_party_lib.do_something()  # Their prints get logged too

Why:
    - Third-party libraries often use print() instead of logging
    - print() output goes to stdout, not log files
    - In async code, print() can mess up terminal output
    - We want all output in structured logs for analysis

Tiger Style:
    - Context manager ensures cleanup (fail-safe)
    - Configurable flush behavior
    - Can capture or pass-through stderr
    - Bounded: No infinite buffering
"""

import logging
import sys
from contextlib import contextmanager
from typing import TextIO


class PrintToLogger:
    """File-like object that redirects writes to logger.

    Tiger Style: Bounded writes, explicit flush behavior.
    """

    def __init__(
        self,
        logger: logging.Logger,
        level: int = logging.INFO,
        original_stream: TextIO | None = None,
        also_print: bool = False,
    ):
        """Initialize print-to-logger adapter.

        Args:
            logger: Logger instance to write to
            level: Log level for messages (default: INFO)
            original_stream: Original stream (for restoration), optional
            also_print: If True, also write to original stream (tee behavior)
        """
        self.logger = logger
        self.level = level
        self.original_stream = original_stream
        self.also_print = also_print
        self._buffer = ""  # Tiger: Bounded! Flushed on newline

    def write(self, message: str) -> int:
        """Write message to logger (implements file-like interface).

        Tiger Style: Flush on newline to prevent unbounded buffering.

        Args:
            message: String to write

        Returns:
            Number of characters written (for file-like compatibility)
        """
        if message == "\n":
            # Bare newline - flush buffer if non-empty
            if self._buffer:
                self.logger.log(self.level, self._buffer)
                self._buffer = ""
        elif "\n" in message:
            # Contains newlines - split and flush each line
            lines = (self._buffer + message).split("\n")
            # Log all complete lines (all but last)
            for line in lines[:-1]:
                if line:  # Skip empty lines
                    self.logger.log(self.level, line)
            # Keep incomplete last line in buffer
            self._buffer = lines[-1]
        else:
            # No newlines - add to buffer
            self._buffer += message

        # Tee to original stream if requested
        if self.also_print and self.original_stream:
            self.original_stream.write(message)

        return len(message)

    def flush(self) -> None:
        """Flush buffered content (implements file-like interface).

        Tiger Style: Explicit flush clears buffer.
        """
        if self._buffer:
            self.logger.log(self.level, self._buffer)
            self._buffer = ""

        if self.also_print and self.original_stream:
            self.original_stream.flush()

    def isatty(self) -> bool:
        """Return False (not a TTY)."""
        return False

    def __enter__(self):
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Flush on exit (Tiger: cleanup!)."""
        self.flush()
        return False


@contextmanager
def intercept_prints(
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    intercept_stdout: bool = True,
    intercept_stderr: bool = False,
    also_print: bool = False,
):
    """Context manager to redirect print() and sys.stdout to logger.

    Tiger Style: Explicit scope, guaranteed cleanup.
    Sean Goedecke: Use stdlib sys.stdout override pattern.

    Args:
        logger: Logger to send output to (default: root logger)
        level: Log level for stdout messages (default: INFO)
               stderr always uses ERROR level
        intercept_stdout: Redirect stdout (print() calls)
        intercept_stderr: Redirect stderr (useful for noisy libraries)
        also_print: If True, also print to original stdout/stderr (tee mode)

    Example:
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>>
        >>> with intercept_prints(logger):
        ...     print("This goes to logger.info()")
        ...     import backend_bench  # Their prints get logged

    Example (with tee):
        >>> with intercept_prints(logger, also_print=True):
        ...     print("Goes to both logger AND stdout")
    """
    if logger is None:
        logger = logging.getLogger()

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        if intercept_stdout:
            sys.stdout = PrintToLogger(
                logger,
                level=level,
                original_stream=old_stdout,
                also_print=also_print,
            )

        if intercept_stderr:
            sys.stderr = PrintToLogger(
                logger,
                level=logging.ERROR,  # stderr -> ERROR level
                original_stream=old_stderr,
                also_print=also_print,
            )

        yield

    finally:
        # Tiger Style: Guaranteed cleanup!
        if intercept_stdout and isinstance(sys.stdout, PrintToLogger):
            sys.stdout.flush()
            sys.stdout = old_stdout

        if intercept_stderr and isinstance(sys.stderr, PrintToLogger):
            sys.stderr.flush()
            sys.stderr = old_stderr
