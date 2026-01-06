"""Unified event emission for TUI consumption.

All long-running processes (eval, GEPA, RL training) emit structured events
to a JSONL file. The TUI tails this file and renders progress.

This decouples the process from the UI - the process doesn't know or care
if anyone is watching. Events are always written for later analysis too.

Event types:
    sample_start/end - Evaluation sample lifecycle
    turn - Agent turn within a sample
    modal_progress - GPU eval phases (compiling, checking, benchmarking)
    gepa_iteration - GEPA optimization progress
    rl_step - RL training step metrics
    log - Generic log message (routed to panes by logger name)

Usage:
    from rollouts.events import EventEmitter, get_emitter

    # In evaluate():
    emitter = EventEmitter(output_dir)
    emitter.emit("sample_start", id="001", name="Square_matmul", total=10)
    # ... do work ...
    emitter.emit("sample_end", id="001", score=0.85, time_sec=45.2)

    # In GEPA:
    emitter = get_emitter()  # Gets from context
    emitter.emit("gepa_iteration", iteration=3, evals_used=12, best_score=0.42)

Tiger Style: Pure data out, no UI code, explicit file handle.
"""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)

# Context variable for accessing emitter from anywhere in the call stack
_emitter_ctx: contextvars.ContextVar[EventEmitter | None] = contextvars.ContextVar(
    "event_emitter", default=None
)


def get_emitter() -> EventEmitter | None:
    """Get the current EventEmitter from context.

    Returns None if no emitter is set (events will be silently dropped).
    """
    return _emitter_ctx.get()


def emit_event(event_type: str, **data: Any) -> None:
    """Emit an event if an emitter is configured.

    Convenience function that gets emitter from context.
    Safe to call even if no emitter is set (no-op).

    Args:
        event_type: Event type (sample_start, turn, modal_progress, etc.)
        **data: Event data fields
    """
    emitter = get_emitter()
    if emitter is not None:
        emitter.emit(event_type, **data)


class EventEmitter:
    """Writes structured events to JSONL file/stream.

    Thread-safe via flush-after-write. Events include timestamp automatically.

    Can write to:
    - File (default): {output_dir}/events.jsonl
    - Stdout: For piping to TUI
    - Any TextIO stream

    Usage:
        # File-based (for local runs)
        with EventEmitter(output_dir=Path("./results")) as emitter:
            emitter.emit("sample_start", id="001", name="test")

        # Stdout (for piped runs)
        with EventEmitter(stream=sys.stdout) as emitter:
            emitter.emit("sample_start", id="001", name="test")

        # With context (accessible via get_emitter())
        with EventEmitter(output_dir=path).as_context():
            emit_event("sample_start", id="001")  # Works anywhere in call stack
    """

    def __init__(
        self,
        output_dir: Path | str | None = None,
        stream: TextIO | None = None,
        events_file: str = "events.jsonl",
    ) -> None:
        """Initialize emitter.

        Args:
            output_dir: Directory to write events.jsonl (mutually exclusive with stream)
            stream: Stream to write to (e.g., sys.stdout)
            events_file: Filename within output_dir (default: events.jsonl)
        """
        self._file: TextIO | None = None
        self._owns_file = False
        self._ctx_token: contextvars.Token | None = None

        if stream is not None:
            self._file = stream
            self._owns_file = False
        elif output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self._file = open(output_path / events_file, "a")
            self._owns_file = True
            logger.debug(f"EventEmitter writing to {output_path / events_file}")
        else:
            # No output configured - events will be dropped
            logger.debug("EventEmitter created with no output (events will be dropped)")

    def emit(self, event_type: str, **data: Any) -> None:
        """Emit a structured event.

        Args:
            event_type: Event type (sample_start, sample_end, turn, modal_progress, etc.)
            **data: Event data fields
        """
        if self._file is None:
            return

        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }

        try:
            self._file.write(json.dumps(event) + "\n")
            self._file.flush()
        except Exception as e:
            logger.warning(f"Failed to emit event: {e}")

    def as_context(self) -> EventEmitter:
        """Set this emitter as the context emitter.

        Use as context manager:
            with EventEmitter(output_dir=path).as_context():
                emit_event("sample_start", ...)  # Uses this emitter

        Returns self for chaining.
        """
        self._ctx_token = _emitter_ctx.set(self)
        return self

    def __enter__(self) -> EventEmitter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the emitter and release resources."""
        # Reset context if we set it
        if self._ctx_token is not None:
            _emitter_ctx.reset(self._ctx_token)
            self._ctx_token = None

        # Close file if we own it
        if self._owns_file and self._file is not None:
            self._file.close()
            self._file = None


class LoggingEventEmitter(EventEmitter):
    """EventEmitter that also logs events via Python logging.

    Useful for debugging - events appear in both events.jsonl and logs.
    """

    def __init__(
        self,
        output_dir: Path | str | None = None,
        stream: TextIO | None = None,
        events_file: str = "events.jsonl",
        log_level: int = logging.DEBUG,
    ) -> None:
        super().__init__(output_dir, stream, events_file)
        self._log_level = log_level

    def emit(self, event_type: str, **data: Any) -> None:
        super().emit(event_type, **data)
        logger.log(self._log_level, f"Event: {event_type} {data}")


# ── Event type constants ──────────────────────────────────────────────────────
# Use these for consistency across codebase


class EventTypes:
    """Standard event type constants."""

    # Eval lifecycle
    EVAL_START = "eval_start"
    EVAL_END = "eval_end"
    SAMPLE_START = "sample_start"
    SAMPLE_END = "sample_end"
    TURN = "turn"

    # Modal/GPU progress
    MODAL_PROGRESS = "modal_progress"

    # GEPA
    GEPA_START = "gepa_start"
    GEPA_ITERATION = "gepa_iteration"
    GEPA_ACCEPTED = "gepa_accepted"
    GEPA_REJECTED = "gepa_rejected"
    GEPA_END = "gepa_end"

    # RL Training
    RL_STEP = "rl_step"
    RL_CHECKPOINT = "rl_checkpoint"

    # Generic
    LOG = "log"
    ERROR = "error"
