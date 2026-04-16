"""argus tail — stream events from a run directory to stdout.

Reads run.jsonl and events.jsonl from a local run directory and formats
them for human consumption. No rollouts dependency, no TUI.

Usage:
    argus tail <run_dir>
    argus tail <run_dir> --format json
    argus tail <run_dir> --timestamps
    argus tail results/eval/run_20260414-231604

Output shape:
    [source] event
        line
        line
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from argus.event_log import list_jsonl_journal_paths

# (formatted_text | None, done)
# None = suppress line. done = run has ended, stop polling.
FormatResult = tuple[str | None, bool]

_NOISE = {
    "event",
    "timestamp",
    "ts",
    "logger",
    "level",
    "taskName",
    "provider",
    "ssh_target",
    "engine_log_path",
    "engine_trace_path",
    "engine_cuda_device_ids",
    "service_name",
    "output_dir",
    "stdout_log",
    "stderr_log",
    "command",
    "engine_name",
    "engine_port",
    "model_name",
    "readiness_target",
    "launcher_id",
    "message",
    "log_stream",
    "line",
    "log_tail",
    "log_blob",
}


def _ts(ev: dict) -> str:
    raw = ev.get("timestamp") or ev.get("ts", "")
    return raw[11:19] if len(raw) >= 19 else raw


def _fmt(source: str, text: str, ts: str) -> str:
    return f"{ts}[{source}] {text}"


def _parse_blob(blob: str) -> str:
    """Strip == stdout == / == stderr == headers and indent lines."""
    lines = []
    for line in blob.splitlines():
        if line.strip() in ("== stdout ==", "== stderr =="):
            continue
        lines.append(f"    {line}")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _extra(ev: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in ev.items() if k not in _NOISE and v is not None)


def _event_key(ev: dict) -> str:
    """Canonical event discriminator with backward compatibility."""
    # TODO(event_log_refactor): delete the `message` fallback once producers
    # emit the canonical event envelope everywhere. See
    # docs/design/event_log_refactor.md.
    event = ev.get("event")
    if isinstance(event, str) and event:
        return event
    message = ev.get("message")
    if isinstance(message, str):
        return message
    return ""


def _fmt_run_end(ev: dict, ts: str) -> FormatResult:
    status = ev.get("status", "?")
    return _fmt("argus", f"run ended  {status}  (Ctrl-C to exit)", ts), True


# event key -> formatter
_EVENT_FORMATTERS: dict[str, Callable[[dict, str], FormatResult]] = {
    "run_end": _fmt_run_end,
}


def _format(raw: str, timestamps: bool = False) -> FormatResult:
    """Format one JSONL line. Returns (text | None, done)."""
    try:
        ev = json.loads(raw)
    except json.JSONDecodeError:
        return raw, False

    ts = (_ts(ev) + "  ") if timestamps else ""
    line = ev.get("line")
    if isinstance(line, str):
        return (line if not timestamps else f"{_ts(ev)}  {line}"), False

    blob = ev.get("log_tail") or ev.get("log_blob")
    if isinstance(blob, str) and blob.strip():
        parsed = _parse_blob(blob)
        return (parsed if not timestamps else f"{_ts(ev)}  {parsed}"), False

    event = _event_key(ev)

    if event in _EVENT_FORMATTERS:
        return _EVENT_FORMATTERS[event](ev, ts)

    # Generic fallback
    extra = _extra(ev)
    if event:
        return _fmt("argus", f"{event}  {extra}" if extra else event, ts), False
    message = ev.get("message", "")
    return _fmt("eval", f"{message}  {extra}" if extra else message, ts), False


def tail_run(run_dir: Path, fmt: str = "pretty", timestamps: bool = False) -> int:
    """Stream events from run_dir to stdout. Blocks until Ctrl-C or run_end."""
    print(f"Tailing: {run_dir}  (Ctrl-C to stop)", file=sys.stderr)
    tail_offsets: dict[str, int] = {}

    try:
        while True:
            done = False
            for path in list_jsonl_journal_paths(run_dir):
                prev = tail_offsets.get(path.name, 0)
                cur = path.stat().st_size
                if cur <= prev:
                    continue
                with open(path) as f:
                    f.seek(prev)
                    for raw in f:
                        raw = raw.rstrip()
                        if not raw:
                            continue
                        if fmt == "json":
                            sys.stdout.write(raw + "\n")
                        else:
                            text, is_done = _format(raw, timestamps=timestamps)
                            if text is not None:
                                sys.stdout.write(text + "\n")
                            if is_done:
                                done = True
                    tail_offsets[path.name] = path.stat().st_size
                sys.stdout.flush()
            if done:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)
    return 0


def tail_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="argus tail",
        description="Stream events from a run directory to stdout.",
    )
    parser.add_argument(
        "run_dir", help="Path to run directory (results/eval/<run> or results/rl/<run>)"
    )
    parser.add_argument(
        "--format",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format: pretty (default) or json (raw JSONL)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Prefix each line with HH:MM:SS",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory", file=sys.stderr)
        return 1

    return tail_run(run_dir, fmt=args.format, timestamps=args.timestamps)
