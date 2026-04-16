"""Engine observability — event emission and JSONL sink.

EngineEvent is the single semantic boundary for all engine observability.
JsonlObserver writes events to a JSONL file from a background thread.

Usage:
    observer = JsonlObserver(Path("trace.jsonl"))
    observer.emit(EngineEvent(ts_ns=time.time_ns(), req_id="abc", kind="prefill_end", metadata={...}))
    observer.close()
"""

import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

EngineEventKind = Literal[
    "request_start",
    "prefill_end",
    "decode_step_end",
    "sample",
    "request_end",
    "request_error",
]


@dataclass(frozen=True)
class EngineEvent:
    ts_ns: int
    req_id: str
    kind: EngineEventKind
    metadata: dict[str, object]


@dataclass(frozen=True)
class SampleResult:
    token_id: int
    topk_token_ids: list[int]
    topk_logprobs: list[float]


class JsonlObserver:
    """Writes EngineEvents to a JSONL file from a background thread.

    Non-blocking: emit() enqueues; a daemon thread flushes to disk.
    Call close() at shutdown to drain the queue.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._queue: queue.Queue[EngineEvent | None] = queue.Queue()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def emit(self, event: EngineEvent) -> None:
        self._queue.put(event)

    def close(self) -> None:
        self._queue.put(None)  # sentinel
        self._thread.join()

    def _writer_loop(self) -> None:
        with self._path.open("a") as f:
            while True:
                event = self._queue.get()
                if event is None:
                    break
                f.write(
                    json.dumps({
                        "ts_ns": event.ts_ns,
                        "req_id": event.req_id,
                        "kind": event.kind,
                        **event.metadata,
                    })
                    + "\n"
                )
                f.flush()
