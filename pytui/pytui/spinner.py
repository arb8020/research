"""Standalone CLI spinner for long-running operations.

No TUI app needed — just stderr + ANSI escape codes.
Uses the same braille frames as the rollouts Loader component.

Usage:
    with Spinner("Provisioning GPU...") as s:
        provision()
        s.update("Deploying code...")
        deploy()
    # prints: ✓ Deploying code...

    # Or without context manager:
    s = Spinner("Working...")
    s.start()
    s.update("Still working...")
    s.stop()            # prints: ✓ Still working...
    s.stop("Failed!")   # prints: ✗ Failed!
"""

from __future__ import annotations

import itertools
import sys
import threading
import time

BRAILLE_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


class Spinner:
    """Braille spinner that overwrites a single stderr line."""

    def __init__(self, message: str) -> None:
        self._message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def update(self, message: str) -> None:
        """Change the spinner text (thread-safe)."""
        self._message = message

    def start(self) -> None:
        """Start the spinner background thread."""
        assert self._thread is None, "Spinner already started"
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, final_message: str | None = None, success: bool = True) -> None:
        """Stop the spinner and print final status."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        msg = final_message or self._message
        mark = "✓" if success else "✗"
        sys.stderr.write(f"\r\033[K{mark} {msg}\n")
        sys.stderr.flush()

    def _spin(self) -> None:
        for frame in itertools.cycle(BRAILLE_FRAMES):
            if self._stop.is_set():
                break
            sys.stderr.write(f"\r\033[K{frame} {self._message}")
            sys.stderr.flush()
            time.sleep(0.08)

    def __enter__(self) -> Spinner:
        self.start()
        return self

    def __exit__(self, exc_type: type | None, *_: object) -> None:
        self.stop(success=exc_type is None)
