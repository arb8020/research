"""argus tail — stream events from a run directory to stdout.

Reads run.jsonl and events.jsonl from a local run directory and formats
them for human consumption. No rollouts dependency, no TUI.

Usage:
    argus tail <run_dir>
    argus tail <run_dir> --format json
    argus tail <run_dir> --timestamps
    argus tail results/eval/run_20260414-231604
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Fields that are infrastructure noise — never shown in pretty output
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


def _format(raw: str, timestamps: bool = False) -> str | None:
    """Format one JSONL line. Returns None to suppress."""
    try:
        ev = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    ts = (_ts(ev) + "  ") if timestamps else ""
    msg = ev.get("message", "")
    event = ev.get("event", "")

    # Suppress — handled by caller as buffered block
    if msg in ("eval_inference_service_log", "inference_service_final_log"):
        return None
    if event == "inference_service_final_log":
        return None

    # eval events.jsonl
    if msg == "eval_start":
        return f"{ts}eval started  name={ev.get('eval_name', '?')}  total={ev.get('total', '?')}"
    if msg == "eval_end":
        return f"{ts}eval done  status={ev.get('status', '?')}  reward={ev.get('mean_reward', '?')}"
    if msg == "sample_start":
        return f"{ts}sample {ev.get('sample_id', '?')} start  name={ev.get('sample_name', '')}"
    if msg == "sample_end":
        return f"{ts}sample {ev.get('sample_id', '?')} end  status={ev.get('status', '?')}  reward={ev.get('reward', '?')}"
    if msg == "turn":
        return f"{ts}sample {ev.get('sample_id', '?')}  turn={ev.get('turn', '?')}  {ev.get('status', '')}"
    if msg == "assistant_message":
        content = ev.get("content", "")
        preview = content[:80].replace("\n", " ") + ("..." if len(content) > 80 else "")
        return f"{ts}sample {ev.get('sample_id', '?')}  assistant: {preview}"
    if msg == "llm_call":
        return (
            f"{ts}sample {ev.get('sample_id', '?')}  llm  "
            f"model={ev.get('model', '?')}  "
            f"in={ev.get('tokens_in', '?')}  out={ev.get('tokens_out', '?')}  "
            f"ms={ev.get('duration_ms', '?')}"
        )

    # argus run.jsonl lifecycle events
    if event:
        if event == "inference_engine_launch":
            return f"{ts}server starting  model={ev.get('model_name', '?')}  port={ev.get('engine_port', '?')}"
        if event == "inference_healthcheck_start":
            return f"{ts}waiting for server  timeout={ev.get('startup_timeout', '?')}s"
        if event == "inference_health_state":
            state = ev.get("health_state", "?")
            detail = ev.get("health_detail", "")
            return f"{ts}health  {state}" + (f"  {detail}" if detail else "")
        if event == "inference_startup_failed":
            return f"{ts}server failed  reason={ev.get('failure_kind', '?')}  attempts={ev.get('health_attempt', '?')}"
        if event in ("run_start", "submit_done"):
            return None  # noise for interactive use
        extra = "  ".join(f"{k}={v}" for k, v in ev.items() if k not in _NOISE and v is not None)
        return f"{ts}{event}  {extra}" if extra else f"{ts}{event}"

    # Generic
    extra = "  ".join(f"{k}={v}" for k, v in ev.items() if k not in _NOISE and v is not None)
    return f"{ts}{msg}  {extra}" if extra else f"{ts}{msg}"


def tail_run(run_dir: Path, fmt: str = "pretty", timestamps: bool = False) -> int:
    """Stream events from run_dir to stdout. Blocks until Ctrl-C."""
    print(f"Tailing: {run_dir}  (Ctrl-C to stop)", file=sys.stderr)
    tail_offsets: dict[str, int] = {}
    svc_buf: list[str] = []

    def flush_svc() -> None:
        if svc_buf:
            for line in svc_buf:
                sys.stdout.write(f"    {line}\n")
            svc_buf.clear()

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
                            sys.stdout.write(formatted + "\n")
                    tail_offsets[path.name] = path.stat().st_size
                sys.stdout.flush()
            # Flush any buffered service logs at end of each poll cycle
            # in case they arrived without a subsequent non-service-log event
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
