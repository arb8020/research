"""Compatibility shim for the Argus-owned run/journal substrate.

The canonical generic run-event helpers now live in `argus.event_log`.
Rollouts still re-exports them during the migration so workload code can move
incrementally without reintroducing Argus-owned semantics into Rollouts.
"""

from argus.event_log import (
    ARGUS_RUN_EVENT_SENTINEL,
    ARGUS_RUN_EVENT_STREAM_ENV,
    JsonlEventSink,
    RunEventSink,
    RunEventSinks,
    build_jsonl_run_event_sinks,
    emit_logger_event,
    emit_run_error,
    emit_run_event,
    emit_run_info,
    emit_run_warning,
    existing_journal_paths,
    journal_path,
    list_jsonl_journal_paths,
    stream_run_event_sinks,
)

__all__ = [
    "ARGUS_RUN_EVENT_SENTINEL",
    "ARGUS_RUN_EVENT_STREAM_ENV",
    "JsonlEventSink",
    "RunEventSink",
    "RunEventSinks",
    "build_jsonl_run_event_sinks",
    "emit_logger_event",
    "emit_run_error",
    "emit_run_event",
    "emit_run_info",
    "emit_run_warning",
    "existing_journal_paths",
    "journal_path",
    "list_jsonl_journal_paths",
    "stream_run_event_sinks",
]
