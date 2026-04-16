from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Literal, Protocol

ARGUS_RUN_EVENT_SENTINEL = "__ARGUS_RUN_EVENT__"
ARGUS_RUN_EVENT_STREAM_ENV = "ARGUS_RUN_EVENT_STREAM"

_LogLevel = Literal["debug", "info", "warning", "error"]


class RunEventSink(Protocol):
    def __call__(self, event: str, **data: Any) -> None: ...


def emit_logger_event(
    logger: logging.Logger,
    event: str,
    *,
    message: str | None = None,
    level: _LogLevel = "info",
    **data: Any,
) -> None:
    """Emit one canonical structured logging event."""
    log = getattr(logger, level)
    log(message or event, extra={"event": event, **data})


@dataclass(frozen=True, slots=True)
class RunEventSinks:
    """Configured sinks for structured run-event emission."""

    emit_event: RunEventSink | None = None
    text_logger: logging.Logger | None = None
    stream_events: bool = False
    stream: IO[str] = field(default_factory=lambda: sys.stdout)


class JsonlEventSink:
    """Append-only JSONL sink for run events."""

    def __init__(self, log_file: str | Path) -> None:
        import threading

        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def __call__(self, event: str, **data: Any) -> None:
        entry = {
            "ts": datetime.now().isoformat(),
            "event": event,
            **data,
        }
        with self._write_lock:
            with self.log_file.open("a") as f:
                f.write(json.dumps(entry) + "\n")


def build_jsonl_run_event_sinks(
    log_file: str | Path,
    *,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> RunEventSinks:
    """Construct the canonical JSONL-backed run event sinks for a journal file."""
    sink = JsonlEventSink(log_file)
    resolved_log_file = Path(log_file)

    def _emit_event(event: str, **data: Any) -> None:
        sink(event, **data)
        if on_event is not None:
            on_event(event, data)

    _emit_event.log_file = resolved_log_file
    return RunEventSinks(emit_event=_emit_event)


def emit_run_event(run_event_sinks: RunEventSinks, event: str, **data: Any) -> None:
    if run_event_sinks.emit_event is not None:
        run_event_sinks.emit_event(event, **data)
    if run_event_sinks.stream_events:
        payload = {"event": event, **data}
        print(
            f"{ARGUS_RUN_EVENT_SENTINEL}{json.dumps(payload)}",
            file=run_event_sinks.stream,
            flush=True,
        )


def emit_run_info(
    run_event_sinks: RunEventSinks,
    message: str,
    *,
    event: str | None = None,
    **data: Any,
) -> None:
    if run_event_sinks.text_logger is not None:
        run_event_sinks.text_logger.info(message)
    if event is not None:
        emit_run_event(run_event_sinks, event, message=message, **data)


def emit_run_warning(
    run_event_sinks: RunEventSinks,
    message: str,
    *,
    event: str | None = None,
    **data: Any,
) -> None:
    if run_event_sinks.text_logger is not None:
        run_event_sinks.text_logger.warning(message)
    if event is not None:
        emit_run_event(run_event_sinks, event, message=message, level="warning", **data)


def emit_run_error(
    run_event_sinks: RunEventSinks,
    message: str,
    *,
    event: str | None = None,
    **data: Any,
) -> None:
    if run_event_sinks.text_logger is not None:
        run_event_sinks.text_logger.error(message)
    if event is not None:
        emit_run_event(run_event_sinks, event, message=message, level="error", **data)


def stream_run_event_sinks(text_logger: logging.Logger | None = None) -> RunEventSinks:
    return RunEventSinks(text_logger=text_logger, stream_events=True)
