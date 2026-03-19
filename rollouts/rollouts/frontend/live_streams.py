from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .live_runs import (
    append_output_line,
    get_run,
    has_run,
    load_external_messages,
    mark_run_complete,
    parse_external_run_id,
    release_run_slot,
)

logger = logging.getLogger(__name__)


def _set_sse_headers(handler: Any) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()


def _write_sse(handler: Any, payload: dict[str, Any]) -> None:
    handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
    handler.wfile.flush()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                thinking = block.get("thinking")
                if isinstance(text, str) and text:
                    parts.append(text)
                elif isinstance(thinking, str) and thinking:
                    parts.append(thinking)
            else:
                text = getattr(block, "text", None)
                thinking = getattr(block, "thinking", None)
                if isinstance(text, str) and text:
                    parts.append(text)
                elif isinstance(thinking, str) and thinking:
                    parts.append(thinking)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _normalize_watch_event(raw: dict[str, Any]) -> dict[str, Any]:
    event_type = raw.get("message", "")
    out: dict[str, Any] = {"type": event_type, "timestamp": raw.get("timestamp", "")}

    if event_type == "eval_start":
        out["name"] = raw.get("eval_name", "")
        out["total"] = raw.get("total", 0)
    elif event_type == "sample_start":
        out["id"] = raw.get("sample_id", "")
        out["name"] = raw.get("sample_name", "")
    elif event_type == "turn":
        out["id"] = raw.get("sample_id", "")
        out["turn"] = raw.get("turn", 0)
        out["status"] = raw.get("status", "")
    elif event_type == "sample_end":
        out["id"] = raw.get("sample_id", "")
        out["score"] = raw.get("score", 0)
    elif event_type == "eval_end":
        out["name"] = raw.get("eval_name", "")
        out["total"] = raw.get("total", 0)
    else:
        out.update({k: v for k, v in raw.items() if k not in ("message", "timestamp")})

    return out


def stream_registered_run(handler: Any, run_id: str) -> None:
    if not has_run(run_id):
        raise KeyError(run_id)

    run_data = get_run(run_id)
    assert run_data is not None
    process = run_data["process"]

    _set_sse_headers(handler)

    try:
        events_file: Path | None = None
        events_file_handle = None
        result_dir_found = False
        stdout_buffer = ""

        while True:
            char = process.stdout.read(1)
            if not char:
                break

            stdout_buffer += char
            if char == "\n":
                line = stdout_buffer.rstrip()
                if not result_dir_found:
                    match = re.search(r"📂 Results directory: (.+)", line)
                    if match:
                        result_dir = Path(match.group(1))
                        events_file = result_dir / "events.jsonl"
                        result_dir_found = True

                append_output_line(run_id, line)
                _write_sse(handler, {"line": line, "type": "stdout"})
                stdout_buffer = ""

            if result_dir_found and events_file and events_file.exists():
                if events_file_handle is None:
                    events_file_handle = open(events_file)
                    events_file_handle.seek(0, 2)

                event_line = events_file_handle.readline()
                while event_line:
                    line = event_line.strip()
                    if not line:
                        event_line = events_file_handle.readline()
                        continue

                    event_obj = json.loads(line)
                    assert isinstance(event_obj, dict), (
                        f"Event must be dict, got {type(event_obj)}: {event_obj}"
                    )
                    assert event_obj.get("message") is not None, (
                        f"Event missing 'message' field: {event_obj}"
                    )
                    assert "timestamp" in event_obj, (
                        f"Event missing 'timestamp' field: {event_obj}"
                    )
                    _write_sse(handler, event_obj)
                    event_line = events_file_handle.readline()

        if stdout_buffer:
            _write_sse(handler, {"line": stdout_buffer, "type": "stdout"})

        if events_file_handle:
            events_file_handle.close()

        exit_code = process.wait()
        release_run_slot()
        _write_sse(
            handler,
            {
                "type": "complete",
                "exit_code": exit_code,
                "status": "success" if exit_code == 0 else "failed",
            },
        )
        mark_run_complete(
            run_id,
            status="completed" if exit_code == 0 else "failed",
            exit_code=exit_code,
        )
    except Exception:
        release_run_slot()
        mark_run_complete(run_id, status="failed", exit_code=run_data.get("exit_code"))
        raise


def stream_watch_run(
    handler: Any,
    run_id: str,
    *,
    results_dir: Path,
    known_results_dirs: list[Path],
) -> None:
    parsed_external = parse_external_run_id(run_id)
    if parsed_external is not None:
        runtime, session_id = parsed_external
        _stream_external_session(handler, runtime, session_id)
        return

    events_file: Path | None = None
    for search_dir in [results_dir, *known_results_dirs]:
        candidate = search_dir / run_id / "events.jsonl"
        if candidate.exists():
            events_file = candidate
            break

    if events_file is None:
        raise FileNotFoundError(run_id)

    _set_sse_headers(handler)

    poll_interval = 0.25
    max_idle = 300
    idle_since: float | None = None
    done = False

    with open(events_file) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            _write_sse(handler, _normalize_watch_event(raw))
            if raw.get("message") == "eval_end":
                done = True
                break

        while not done:
            line = fh.readline()
            if line:
                line = line.strip()
                if line:
                    raw = json.loads(line)
                    _write_sse(handler, _normalize_watch_event(raw))
                    if raw.get("message") == "eval_end":
                        done = True
                idle_since = None
            else:
                if idle_since is None:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since > max_idle:
                    break
                time.sleep(poll_interval)

    _write_sse(handler, {"type": "complete", "exit_code": 0, "status": "success"})


def _stream_external_session(handler: Any, runtime: str, session_id: str) -> None:
    _set_sse_headers(handler)

    sample_id = session_id[:8]
    poll_interval = 1.0
    max_idle = 300.0
    idle_since: float | None = None
    last_assistant_count = 0
    sample_started = False
    run_name = f"{runtime} interactive"

    _write_sse(
        handler,
        {
            "type": "eval_start",
            "name": run_name,
            "total": 1,
            "timestamp": datetime.now().isoformat(),
        },
    )

    while True:
        loaded = load_external_messages(runtime, session_id)
        if loaded is None:
            if idle_since is None:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since > max_idle:
                break
            time.sleep(poll_interval)
            continue

        messages, session_path = loaded
        try:
            mtime = session_path.stat().st_mtime
        except OSError:
            mtime = time.time()

        if not sample_started:
            _write_sse(
                handler,
                {
                    "type": "sample_start",
                    "id": sample_id,
                    "name": f"{runtime}:{sample_id}",
                    "timestamp": datetime.fromtimestamp(mtime).isoformat(),
                },
            )
            sample_started = True

        assistant_messages = [msg for msg in messages if getattr(msg, "role", None) == "assistant"]
        if len(assistant_messages) > last_assistant_count:
            for turn_index, msg in enumerate(
                assistant_messages[last_assistant_count:],
                start=last_assistant_count + 1,
            ):
                text = _message_text(getattr(msg, "content", None))
                if not text.strip():
                    continue
                timestamp = getattr(msg, "timestamp", None) or datetime.fromtimestamp(
                    mtime
                ).isoformat()
                _write_sse(
                    handler,
                    {
                        "type": "turn",
                        "id": sample_id,
                        "turn": turn_index,
                        "status": "running",
                        "timestamp": timestamp,
                    },
                )
                _write_sse(
                    handler,
                    {
                        "type": "assistant_message",
                        "sample_id": sample_id,
                        "turn": turn_index,
                        "content": text,
                        "timestamp": timestamp,
                    },
                )
            last_assistant_count = len(assistant_messages)
            idle_since = None
        else:
            if idle_since is None:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since > max_idle:
                break

        time.sleep(poll_interval)

    if sample_started:
        _write_sse(
            handler,
            {
                "type": "sample_end",
                "id": sample_id,
                "score": 0.0,
                "timestamp": datetime.now().isoformat(),
            },
        )
    _write_sse(
        handler,
        {
            "type": "eval_end",
            "name": run_name,
            "total": 1,
            "timestamp": datetime.now().isoformat(),
        },
    )
    _write_sse(handler, {"type": "complete", "exit_code": 0, "status": "success"})
