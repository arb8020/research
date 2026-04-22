"""Internal observability primitives: spans + sinks.

This package implements the OTel-shaped span tree that rollouts emits at run
time. It is deliberately thin and hand-rolled (no ``opentelemetry-*`` SDK) so
the trio story stays first-class and the on-disk format is ours to control.

Two invariants hold across all emitters:

* One span = one JSONL row. Emitted once, at close (``__aexit__``).
  Accumulate attributes during the span's lifetime via ``set_attr``.
* Parent linkage via ``trace_id`` (constant across a tree) + ``parent_span_id``
  (points at the enclosing span). A ``contextvars.ContextVar`` threads the
  current span id through async call stacks; trio propagates contextvars into
  child tasks automatically, so start_soon-launched work inherits the parent.

Field names follow OpenTelemetry semantic conventions where they exist
(``gen_ai.request.model``, ``gen_ai.usage.input_tokens``, etc.). Structural
fields (``trace_id``, ``span_id``, ``parent_span_id``, ``name``,
``start_time_unix_nano``, ``end_time_unix_nano``) match the OTel span
envelope so jsonl rows can be piped into an OTLP filelog receiver without
schema rewrites later.

Why not the OpenTelemetry SDK: at our volume (tens of spans/sec peak) the
SDK's batching/exporting machinery buys us little, its trio context
propagation is not as clean as stdlib ``contextvars``, and keeping the
emission path small means every field in the jsonl comes from one place
we own.
"""

from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Per-line cap for spans.jsonl writes. Two processes append to the same file;
# POSIX guarantees atomic append for writes below PIPE_BUF (commonly 4KB) but
# in practice up to at least 16KB on Linux/macOS for local-disk append-only
# files. We pick 16KB as a pragmatic ceiling and truncate oversized attribute
# values at emit time. The canonical-full data for any given span lives in
# sidecar artifacts (samples/<id>.json, etc.); spans are telemetry indexing
# into those, not the source of truth.
_SPAN_LINE_CAP_BYTES = 16 * 1024
# Per-attribute truncation threshold. Attribute values larger than this get
# replaced with an inline marker carrying the original length and a ``
# truncated=True`` flag, and the whole span is re-serialized. If the span is
# still oversize we also drop the largest remaining attributes, preserving
# structural fields.
_ATTR_VALUE_CAP_BYTES = 4 * 1024


@dataclass
class Span:
    """One in-flight span, mutated during its lifetime and emitted on close.

    Mirrors the OTel span shape: trace/span ids, name, nano timestamps,
    attributes bag, status. Status defaults to ``OK`` and flips to ``ERROR``
    if an exception propagates through the context manager. Attributes are a
    flat dict, keyed by OTel semantic-convention names where applicable.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time_unix_nano: int
    attributes: dict[str, Any] = field(default_factory=dict)
    end_time_unix_nano: int | None = None
    status_code: str = "OK"  # "OK" | "ERROR"
    status_message: str | None = None

    def set_attr(self, key: str, value: Any) -> None:
        """Stash one attribute. Last write wins."""
        self.attributes[key] = value

    def record_error(self, exc: BaseException) -> None:
        """Flip status to ERROR and attach exception type + message.

        Called from the ``start_span`` context manager's exception path. We
        keep the type and short message on the span itself; full tracebacks
        should be directed at sidecar artifacts, not inlined, so span rows
        stay under the line cap.
        """
        self.status_code = "ERROR"
        self.status_message = f"{type(exc).__name__}: {exc}"[:512]


# Current-span contextvar. ``None`` means "no parent" (root span). Child spans
# read this at start to populate parent_span_id, then bind themselves as the
# new current for their own children.
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "rollouts_current_span",
    default=None,
)

# The sink contextvar gives nested spans a way to find the writer without
# explicit plumbing. Set once per process (or per-run-dir) at the top of the
# run; unset defaults to a no-op sink.
_current_sink: contextvars.ContextVar[SpanSink | None] = contextvars.ContextVar(
    "rollouts_current_span_sink",
    default=None,
)


def _new_id(length: int) -> str:
    """Generate a trace/span id.

    OTel trace_id is 16 bytes (32 hex chars); span_id is 8 bytes (16 hex
    chars). We use uuid4-derived hex so ids are globally unique without
    coordination.
    """
    return uuid.uuid4().hex[:length]


def new_trace_id() -> str:
    return _new_id(32)


def new_span_id() -> str:
    return _new_id(16)


class SpanSink:
    """Append-only JSONL sink for spans.

    Safe to share across trio tasks in one process. Safe to *point the same
    file* at writers in two sibling processes: POSIX atomic append holds for
    our line sizes (<16KB, well under filesystem block boundaries on local
    disk). We don't take a lock; we rely on kernel atomicity. If the eval
    child and the serving supervisor both open the same ``spans.jsonl`` in
    append mode and write full lines, interleaving at the byte level will
    not happen.

    The sink lives for the lifetime of one run. Closing it flushes and
    closes the file; writing after close asserts.
    """

    def __init__(self, path: Path) -> None:
        # Line-buffered so each complete ``write(line)`` hits the kernel
        # promptly; ``O_APPEND`` is implied by mode ``"a"`` on POSIX which is
        # what gives us the atomic-append guarantee.
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a", buffering=1, encoding="utf-8")
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._f.close()

    def emit(self, span: Span) -> None:
        """Serialize + write one span. Called once, at span close."""
        assert not self._closed, "SpanSink.emit called after close"
        payload = _span_to_row(span)
        line = self._encode_with_cap(payload)
        self._f.write(line)

    def _encode_with_cap(self, payload: dict[str, Any]) -> str:
        """Serialize to JSON + newline, truncating oversized attributes.

        Strategy:
          1. Try the straight serialization.
          2. If over cap, truncate each attribute value whose JSON form
             exceeds ``_ATTR_VALUE_CAP_BYTES``. Retry.
          3. If still over cap, drop attribute values in descending size
             order, replacing them with a marker. Retry.
          4. Structural fields (trace_id, span_id, name, times, status)
             are never dropped.
        """
        line = json.dumps(payload, default=_json_default, ensure_ascii=False) + "\n"
        if len(line.encode("utf-8")) <= _SPAN_LINE_CAP_BYTES:
            return line

        attrs = payload.get("attributes", {})
        if isinstance(attrs, dict) and attrs:
            payload["attributes"] = _truncate_large_values(attrs)
            line = json.dumps(payload, default=_json_default, ensure_ascii=False) + "\n"
            if len(line.encode("utf-8")) <= _SPAN_LINE_CAP_BYTES:
                return line

        # Last-resort: drop the biggest remaining attributes until we fit.
        attrs = payload.get("attributes", {})
        if isinstance(attrs, dict) and attrs:
            payload["attributes"] = _drop_until_fits(payload, attrs)
            line = json.dumps(payload, default=_json_default, ensure_ascii=False) + "\n"
        # If we still don't fit after dropping all attributes, something has
        # gone badly wrong with a structural field; let the downstream reader
        # see the oversized line rather than silently corrupt.
        return line


def _truncate_large_values(attrs: dict[str, Any]) -> dict[str, Any]:
    """Replace each attribute whose serialized form exceeds the per-value cap."""
    out: dict[str, Any] = {}
    for k, v in attrs.items():
        encoded = json.dumps(v, default=_json_default, ensure_ascii=False)
        size = len(encoded.encode("utf-8"))
        if size <= _ATTR_VALUE_CAP_BYTES:
            out[k] = v
            continue
        # Keep a short head of the value plus a marker so readers can see
        # it was truncated and how large the original was.
        head = encoded[: _ATTR_VALUE_CAP_BYTES // 2]
        out[k] = {
            "_truncated": True,
            "_original_bytes": size,
            "head": head,
        }
    return out


def _drop_until_fits(payload: dict[str, Any], attrs: dict[str, Any]) -> dict[str, Any]:
    """Drop attributes in descending size order until the span fits the cap."""
    remaining = dict(attrs)
    by_size = sorted(
        remaining.items(),
        key=lambda kv: len(json.dumps(kv[1], default=_json_default, ensure_ascii=False)),
        reverse=True,
    )
    for k, _ in by_size:
        remaining.pop(k, None)
        remaining["_dropped_attrs"] = remaining.get("_dropped_attrs", []) + [k]
        trial = {**payload, "attributes": remaining}
        line = json.dumps(trial, default=_json_default, ensure_ascii=False) + "\n"
        if len(line.encode("utf-8")) <= _SPAN_LINE_CAP_BYTES:
            return remaining
    return remaining


def _json_default(obj: Any) -> Any:
    """Best-effort JSON encoder hook for non-native types in attributes."""
    if isinstance(obj, Path):
        return str(obj)
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def _span_to_row(span: Span) -> dict[str, Any]:
    """Render a span into the on-disk JSONL shape.

    Top-level keys mirror the OTel span envelope so this row can be fed into
    an OTLP filelog receiver with a near-identity transform.
    """
    return {
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_span_id,
        "name": span.name,
        "start_time_unix_nano": span.start_time_unix_nano,
        "end_time_unix_nano": span.end_time_unix_nano,
        "status": {
            "code": span.status_code,
            "message": span.status_message,
        },
        "attributes": dict(span.attributes),
    }


def set_sink(sink: SpanSink | None) -> contextvars.Token[SpanSink | None]:
    """Bind a sink as the current-process span sink.

    Returns the token for later reset. In most callers this is done once at
    the top of a run; scoped resetting is uncommon.
    """
    return _current_sink.set(sink)


def current_sink() -> SpanSink | None:
    return _current_sink.get()


def current_span() -> Span | None:
    return _current_span.get()


@asynccontextmanager
async def start_span(
    name: str,
    *,
    trace_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> AsyncIterator[Span]:
    """Open a span, bind it as the current span for children, emit on close.

    Used as ``async with start_span("llm_call", attributes={...}) as span:``.
    Callers can mutate ``span`` during the body (``span.set_attr(k, v)``)
    and those mutations land on the emitted row.

    Parent linkage:
      * ``parent_span_id`` is read from ``_current_span`` at entry.
      * ``trace_id`` is inherited from the parent by default; the root of a
        new tree passes an explicit ``trace_id`` (or ``None`` + no parent,
        in which case we mint a new trace id).

    Emission discipline:
      * One row per span, written once, at ``__aexit__``.
      * Exception in the body flips status to ``ERROR`` with type+message,
        then re-raises. The span still emits — telemetry should always
        reflect what happened, including failures.
      * If no sink is bound, the span is a no-op from an output perspective
        but still threads parent context so nested spans continue to link.
    """
    parent = _current_span.get()
    if trace_id is None:
        trace_id = parent.trace_id if parent is not None else new_trace_id()
    span = Span(
        trace_id=trace_id,
        span_id=new_span_id(),
        parent_span_id=parent.span_id if parent is not None else None,
        name=name,
        start_time_unix_nano=time.time_ns(),
        attributes=dict(attributes) if attributes else {},
    )
    token = _current_span.set(span)
    try:
        yield span
    except BaseException as exc:
        span.record_error(exc)
        raise
    finally:
        span.end_time_unix_nano = time.time_ns()
        sink = _current_sink.get()
        if sink is not None:
            try:
                sink.emit(span)
            except Exception:
                # Telemetry must never kill the actual workload. If the sink
                # breaks we swallow the emit failure; the run itself is the
                # source of truth for whether work happened.
                pass
        _current_span.reset(token)


def resolve_run_dir_from_env(default: Path | None = None) -> Path | None:
    """Return the run output dir from ``ROLLOUTS_OUTPUT_DIR`` if set.

    Both the serving supervisor and the eval child already set this env var
    for their subprocesses (see ``rollouts/serving/supervisor.py``). Passing
    the run dir via env keeps the sink-construction site free of cross-cutting
    plumbing.
    """
    env = os.environ.get("ROLLOUTS_OUTPUT_DIR")
    if env:
        return Path(env)
    return default
