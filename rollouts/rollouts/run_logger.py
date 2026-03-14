from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ARGUS_RUN_EVENT_SENTINEL = "__ARGUS_RUN_EVENT__"
ARGUS_RUN_EVENT_STREAM_ENV = "ARGUS_RUN_EVENT_STREAM"


@dataclass(slots=True)
class RunLogger:
    """Structured run-event logger with optional text and stream sinks.

    One canonical event object can be fanned out to multiple representations:
    - durable structured event sink (`emit_event`)
    - human-readable Python logger (`text_logger`)
    - stdout event stream across a process boundary (`stream_events`)
    """

    emit_event: Any | None = None
    text_logger: logging.Logger | None = None
    stream_events: bool = False
    stream = None

    def __post_init__(self) -> None:
        if self.stream is None:
            self.stream = sys.stdout

    def event(self, event: str, **data: Any) -> None:
        if self.emit_event is not None:
            self.emit_event(event, **data)
        if self.stream_events:
            payload = {"event": event, **data}
            print(f"{ARGUS_RUN_EVENT_SENTINEL}{json.dumps(payload)}", file=self.stream, flush=True)

    def info(self, message: str, *, event: str | None = None, **data: Any) -> None:
        if self.text_logger is not None:
            self.text_logger.info(message)
        if event is not None:
            self.event(event, message=message, **data)

    def warning(self, message: str, *, event: str | None = None, **data: Any) -> None:
        if self.text_logger is not None:
            self.text_logger.warning(message)
        if event is not None:
            self.event(event, message=message, level="warning", **data)

    def error(self, message: str, *, event: str | None = None, **data: Any) -> None:
        if self.text_logger is not None:
            self.text_logger.error(message)
        if event is not None:
            self.event(event, message=message, level="error", **data)

    def __call__(self, event: str, **data: Any) -> None:
        self.event(event, **data)


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


def stream_run_logger(text_logger: logging.Logger | None = None) -> RunLogger:
    return RunLogger(text_logger=text_logger, stream_events=True)
