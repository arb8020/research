"""argus tail — stream events from a run directory to stdout.

Reads run.jsonl and events.jsonl from a local run directory and formats
them for human consumption. No rollouts dependency, no TUI.

Usage:
    argus tail <run_dir>
    argus tail <run_dir> --format json
    argus tail <run_dir> --timestamps
    argus tail results/eval/run_20260414-231604

Output shape:
    [source] event  key=value ...
        line
        line
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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


def _fmt_line(source: str, event: str, ts: str) -> str:
    return f"{ts}[{source}] {event}"


def _format(raw: str, timestamps: bool = False) -> str | None:
    """Format one JSONL line. Returns None to suppress."""
    try:
        ev = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    ts = (_ts(ev) + "  ") if timestamps else ""
    msg = ev.get("message", "")
    event = ev.get("event", "")

    # Suppress individual service log lines — handled by caller as buffered block
    if msg == "eval_inference_service_log":
        return None
    # inference_service_final_log contains the full log blob — use it as fallback
    if msg == "inference_service_final_log" or event == "inference_service_final_log":
        blob = (ev.get("log_tail") or ev.get("log_blob") or "").strip()
        if not blob:
            return None
        lines = "\n".join(f"    {l}" for l in blob.splitlines())
        return f"[server]\n{lines}"
    if event in ("run_start", "submit_done"):
        return None

    # eval events.jsonl
    if msg == "eval_start":
        return _fmt_line(
            "eval", f"started  name={ev.get('eval_name', '?')}  total={ev.get('total', '?')}", ts
        )
    if msg == "eval_end":
        return _fmt_line(
            "eval", f"done  status={ev.get('status', '?')}  reward={ev.get('mean_reward', '?')}", ts
        )
    if msg == "sample_start":
        return _fmt_line(
            "eval", f"sample {ev.get('sample_id', '?')} start  name={ev.get('sample_name', '')}", ts
        )
    if msg == "sample_end":
        return _fmt_line(
            "eval",
            f"sample {ev.get('sample_id', '?')} end  status={ev.get('status', '?')}  reward={ev.get('reward', '?')}",
            ts,
        )
    if msg == "turn":
        return _fmt_line("eval", f"sample {ev.get('sample_id', '?')}  {ev.get('status', '')}", ts)
    if msg == "assistant_message":
        content = ev.get("content", "")
        preview = content[:80].replace("\n", " ") + ("..." if len(content) > 80 else "")
        return _fmt_line("eval", f"sample {ev.get('sample_id', '?')}  assistant: {preview}", ts)
    if msg == "llm_call":
        return _fmt_line(
            "eval",
            f"sample {ev.get('sample_id', '?')}  llm  "
            f"model={ev.get('model', '?')}  "
            f"in={ev.get('tokens_in', '?')}  out={ev.get('tokens_out', '?')}  "
            f"ms={ev.get('duration_ms', '?')}",
            ts,
        )

    # argus run.jsonl lifecycle events
    if event:
        if event == "inference_engine_launch":
            return _fmt_line(
                "argus",
                f"server starting  model={ev.get('model_name', '?')}  port={ev.get('engine_port', '?')}",
                ts,
            )
        if event == "inference_healthcheck_start":
            return _fmt_line(
                "argus", f"waiting for server  timeout={ev.get('startup_timeout', '?')}s", ts
            )
        if event == "inference_health_state":
            state = ev.get("health_state", "?")
            detail = ev.get("health_detail", "")
            return _fmt_line("argus", f"health  {state}" + (f"  {detail}" if detail else ""), ts)
        if event == "inference_startup_failed":
            return _fmt_line(
                "argus",
                f"server failed  reason={ev.get('failure_kind', '?')}  attempts={ev.get('health_attempt', '?')}",
                ts,
            )
        extra = "  ".join(f"{k}={v}" for k, v in ev.items() if k not in _NOISE and v is not None)
        return _fmt_line("argus", f"{event}  {extra}" if extra else event, ts)

    # Generic
    extra = "  ".join(f"{k}={v}" for k, v in ev.items() if k not in _NOISE and v is not None)
    return _fmt_line("eval", f"{msg}  {extra}" if extra else msg, ts)


def tail_run(run_dir: Path, fmt: str = "pretty", timestamps: bool = False) -> int:
    """Stream events from run_dir to stdout. Blocks until Ctrl-C."""
    print(f"Tailing: {run_dir}  (Ctrl-C to stop)", file=sys.stderr)
    tail_offsets: dict[str, int] = {}
    svc_buf: list[str] = []
    svc_flushed: list[bool] = [False]  # tracks if we already printed the server block

    def flush_svc() -> None:
        if svc_buf:
            sys.stdout.write("[server]\n")
            for line in svc_buf:
                sys.stdout.write(f"    {line}\n")
            svc_buf.clear()
            svc_flushed[0] = True

    try:
        while True:
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
                            continue
                        try:
                            ev = json.loads(raw)
                            if ev.get("message") == "eval_inference_service_log":
                                svc_buf.append((ev.get("line") or "").rstrip())
                                continue
                        except json.JSONDecodeError:
                            pass
                        flush_svc()
                        formatted = _format(raw, timestamps=timestamps)
                        if formatted is not None:
                            # Skip final log blob if we already printed individual lines
                            is_final = "inference_service_final_log" in raw
                            if is_final and svc_flushed[0]:
                                continue
                            sys.stdout.write(formatted + "\n")
                    tail_offsets[path.name] = path.stat().st_size
                sys.stdout.flush()
            flush_svc()
            sys.stdout.flush()
            time.sleep(0.5)
    except KeyboardInterrupt:
        flush_svc()
        sys.stdout.flush()
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
