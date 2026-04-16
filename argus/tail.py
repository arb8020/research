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
    event = ev.get("event")
    if isinstance(event, str) and event:
        return event
    message = ev.get("message")
    if isinstance(message, str):
        return message
    return ""


# ---------------------------------------------------------------------------
# Per-event formatters: ev, ts -> FormatResult
# ---------------------------------------------------------------------------


def _suppress(ev: dict, ts: str) -> FormatResult:
    return None, False


def _fmt_service_final_log(ev: dict, ts: str) -> FormatResult:
    blob = (ev.get("log_tail") or ev.get("log_blob") or "").strip()
    if not blob:
        return None, False
    return f"[server]\n{_parse_blob(blob)}", False


def _fmt_eval_start(ev: dict, ts: str) -> FormatResult:
    name = ev.get("eval_name", "?")
    total = ev.get("total", "?")
    return _fmt("eval", f"started  {name}  {total} samples", ts), False


def _fmt_eval_end(ev: dict, ts: str) -> FormatResult:
    status = ev.get("status", "?")
    reward = ev.get("mean_reward")
    suffix = f"  reward {reward:.3f}" if isinstance(reward, (int, float)) else ""
    return _fmt("eval", f"done  {status}{suffix}", ts), False


def _fmt_sample_start(ev: dict, ts: str) -> FormatResult:
    sid = ev.get("sample_id", "?")
    name = ev.get("sample_name", "")
    return _fmt("eval", f"sample {sid}  start" + (f"  {name}" if name else ""), ts), False


def _fmt_sample_end(ev: dict, ts: str) -> FormatResult:
    sid = ev.get("sample_id", "?")
    status = ev.get("status", "?")
    reward = ev.get("reward")
    suffix = f"  reward {reward:.3f}" if isinstance(reward, (int, float)) else ""
    return _fmt("eval", f"sample {sid}  {status}{suffix}", ts), False


def _fmt_turn(ev: dict, ts: str) -> FormatResult:
    return _fmt("eval", f"sample {ev.get('sample_id', '?')}  {ev.get('status', '')}", ts), False


def _fmt_assistant(ev: dict, ts: str) -> FormatResult:
    content = ev.get("content", "")
    preview = content[:80].replace("\n", " ") + ("..." if len(content) > 80 else "")
    return _fmt("eval", f"sample {ev.get('sample_id', '?')}  {preview}", ts), False


def _fmt_llm_call(ev: dict, ts: str) -> FormatResult:
    sid = ev.get("sample_id", "?")
    model = ev.get("model", "?")
    tok_in = ev.get("tokens_in", "?")
    tok_out = ev.get("tokens_out", "?")
    ms = ev.get("duration_ms", "?")
    return _fmt("eval", f"sample {sid}  {model}  {tok_in}→{tok_out} tok  {ms}ms", ts), False


def _fmt_engine_launch(ev: dict, ts: str) -> FormatResult:
    model = ev.get("model_name", "?")
    return _fmt("server", f"starting  {model}", ts), False


def _fmt_startup_failed(ev: dict, ts: str) -> FormatResult:
    kind = ev.get("failure_kind", "?")
    attempts = ev.get("health_attempt", "?")
    return _fmt("server", f"failed  {kind}  after {attempts} attempts", ts), False


def _fmt_run_end(ev: dict, ts: str) -> FormatResult:
    status = ev.get("status", "?")
    return _fmt("argus", f"run ended  {status}  (Ctrl-C to exit)", ts), True


# message key -> formatter
_MSG_FORMATTERS: dict[str, Callable[[dict, str], FormatResult]] = {
    "inference_service_final_log": _fmt_service_final_log,
    "eval_start": _fmt_eval_start,
    "eval_end": _fmt_eval_end,
    "sample_start": _fmt_sample_start,
    "sample_end": _fmt_sample_end,
    "turn": _fmt_turn,
    "assistant_message": _fmt_assistant,
    "llm_call": _fmt_llm_call,
}

# event key -> formatter
_EVENT_FORMATTERS: dict[str, Callable[[dict, str], FormatResult]] = {
    "run_start": _suppress,
    "submit_done": _suppress,
    "eval_inference_service_log": _suppress,
    "inference_service_final_log": _fmt_service_final_log,
    "inference_engine_launch": _fmt_engine_launch,
    "inference_healthcheck_start": _suppress,
    "inference_health_state": _suppress,
    "inference_startup_failed": _fmt_startup_failed,
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

    event = _event_key(ev)

    if event in _EVENT_FORMATTERS:
        return _EVENT_FORMATTERS[event](ev, ts)
    if event in _MSG_FORMATTERS:
        return _MSG_FORMATTERS[event](ev, ts)

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
            for path in sorted(run_dir.glob("*.jsonl")):
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
