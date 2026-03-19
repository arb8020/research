from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXTERNAL_WATCH_PREFIX = "external_session_"

_active_runs: dict[str, dict[str, Any]] = {}
_run_counter = 0
_run_lock = threading.Lock()
_max_concurrent_runs = 2
_run_semaphore = threading.Semaphore(_max_concurrent_runs)


def external_run_id(runtime: str, session_id: str) -> str:
    return f"{EXTERNAL_WATCH_PREFIX}{runtime}_{session_id}"


def parse_external_run_id(run_id: str) -> tuple[str, str] | None:
    if not run_id.startswith(EXTERNAL_WATCH_PREFIX):
        return None
    rest = run_id[len(EXTERNAL_WATCH_PREFIX) :]
    runtime, sep, session_id = rest.partition("_")
    if not sep or not runtime or not session_id:
        return None
    return runtime, session_id


def recent_codex_sessions(limit: int = 20, max_age_seconds: float = 1800.0) -> list[dict[str, Any]]:
    sessions_dir = Path.home() / ".codex" / "sessions"
    if not sessions_dir.exists():
        return []

    now = time.time()
    sessions: list[dict[str, Any]] = []
    for session_file in sessions_dir.rglob("*.jsonl"):
        try:
            stat = session_file.stat()
        except OSError:
            continue
        if now - stat.st_mtime > max_age_seconds:
            continue

        try:
            with session_file.open() as f:
                first_line = f.readline().strip()
            if not first_line:
                continue
            first = json.loads(first_line)
        except (OSError, json.JSONDecodeError):
            continue

        if first.get("type") != "session_meta":
            continue
        payload = first.get("payload", {})
        session_id = payload.get("id")
        cwd = payload.get("cwd")
        if not isinstance(session_id, str) or not session_id:
            continue

        sessions.append({
            "runtime": "codex",
            "session_id": session_id,
            "path": session_file,
            "cwd": cwd if isinstance(cwd, str) else None,
            "modified": stat.st_mtime,
            "model": payload.get("model_slug") or payload.get("model"),
        })

    sessions.sort(key=lambda item: item["modified"], reverse=True)
    return sessions[:limit]


def recent_claude_sessions(
    limit: int = 20, max_age_seconds: float = 1800.0
) -> list[dict[str, Any]]:
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return []

    now = time.time()
    sessions: list[dict[str, Any]] = []
    for session_file in projects_dir.rglob("*.jsonl"):
        try:
            stat = session_file.stat()
        except OSError:
            continue
        if now - stat.st_mtime > max_age_seconds:
            continue

        try:
            with session_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    first = json.loads(line)
                    break
                else:
                    continue
        except (OSError, json.JSONDecodeError):
            continue

        session_id = first.get("sessionId")
        cwd = first.get("cwd")
        if not isinstance(session_id, str) or not session_id:
            continue

        sessions.append({
            "runtime": "claude_code",
            "session_id": session_id,
            "path": session_file,
            "cwd": cwd if isinstance(cwd, str) else None,
            "modified": stat.st_mtime,
            "model": None,
        })

    sessions.sort(key=lambda item: item["modified"], reverse=True)
    return sessions[:limit]


def discover_external_session_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in [*recent_codex_sessions(), *recent_claude_sessions()]:
        run_id = external_run_id(item["runtime"], item["session_id"])
        if run_id in seen:
            continue
        seen.add(run_id)
        cwd = item.get("cwd")
        cwd_name = Path(cwd).name if isinstance(cwd, str) and cwd else item["session_id"][:8]
        model_suffix = (
            f" · {item['model']}" if isinstance(item.get("model"), str) and item["model"] else ""
        )
        runs.append({
            "run_id": run_id,
            "config_name": f"{item['runtime']} · {cwd_name}{model_suffix}",
            "start_time": item["modified"],
            "status": "watching",
            "exit_code": None,
            "output_length": 1,
        })

    runs.sort(key=lambda run: run["start_time"], reverse=True)
    return runs


def load_external_messages(runtime: str, session_id: str) -> tuple[list[Any], Path] | None:
    try:
        if runtime == "codex":
            from ..drivers.session_adapter import codex_session_to_messages, find_codex_session

            session_path = find_codex_session(session_id)
            if session_path is None or not session_path.exists():
                return None
            return codex_session_to_messages(session_path), session_path
        if runtime == "claude_code":
            from ..drivers.session_adapter import claude_session_to_messages, find_claude_session

            session_path = find_claude_session(session_id)
            if session_path is None or not session_path.exists():
                return None
            return claude_session_to_messages(session_path), session_path
    except Exception:
        logger.exception("Failed to load external session %s/%s", runtime, session_id)
        return None
    return None


def max_concurrent_runs() -> int:
    return _max_concurrent_runs


def acquire_run_slot() -> bool:
    return _run_semaphore.acquire(blocking=False)


def release_run_slot() -> None:
    _run_semaphore.release()


def next_run_id() -> str:
    global _run_counter
    with _run_lock:
        _run_counter += 1
        return f"run_{_run_counter}_{int(time.time())}"


def register_run(run_id: str, data: dict[str, Any]) -> None:
    with _run_lock:
        _active_runs[run_id] = data


def has_run(run_id: str) -> bool:
    return run_id in _active_runs


def get_run(run_id: str) -> dict[str, Any] | None:
    return _active_runs.get(run_id)


def active_run_ids() -> list[str]:
    return list(_active_runs.keys())


def append_output_line(run_id: str, line: str) -> None:
    with _run_lock:
        run_data = _active_runs.get(run_id)
        if run_data is not None:
            run_data.setdefault("output_lines", []).append(line)


def mark_run_complete(run_id: str, *, status: str, exit_code: int | None) -> None:
    with _run_lock:
        run_data = _active_runs.get(run_id)
        if run_data is not None:
            run_data["status"] = status
            run_data["exit_code"] = exit_code


def list_registered_runs() -> list[dict[str, Any]]:
    runs = []
    with _run_lock:
        for run_id, data in _active_runs.items():
            runs.append({
                "run_id": run_id,
                "config_name": data["config_name"],
                "start_time": data["start_time"],
                "status": data["status"],
                "exit_code": data.get("exit_code"),
                "output_length": len(data.get("output_lines", [])),
            })
    return runs


def discover_watching_runs(results_dir: Path, known_results_dirs: list[Path]) -> list[dict[str, Any]]:
    watching = []
    seen_ids: set[str] = set(_active_runs.keys())
    all_dirs = [results_dir, *known_results_dirs]

    for search_dir in all_dirs:
        if not search_dir.exists():
            continue
        for subdir in search_dir.iterdir():
            if not subdir.is_dir():
                continue
            run_id = subdir.name
            if run_id in seen_ids:
                continue
            events_file = subdir / "events.jsonl"
            report_file = subdir / "report.json"
            if not events_file.exists() or report_file.exists():
                continue
            try:
                with open(events_file, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 4096))
                    tail = f.read().decode("utf-8", errors="replace")
                last_line = next((l for l in reversed(tail.splitlines()) if l.strip()), "")
                if '"eval_end"' in last_line or '"message": "eval_end"' in last_line:
                    continue
            except OSError:
                pass
            age_seconds = time.time() - events_file.stat().st_mtime
            if age_seconds > 1800:
                continue
            seen_ids.add(run_id)
            watching.append({
                "run_id": run_id,
                "config_name": run_id,
                "start_time": events_file.stat().st_mtime,
                "status": "watching",
                "exit_code": None,
                "output_length": 0,
            })

    for external_run in discover_external_session_runs():
        run_id = external_run["run_id"]
        if run_id in seen_ids:
            continue
        seen_ids.add(run_id)
        watching.append(external_run)

    return watching


def kill_run(run_id: str) -> tuple[bool, str]:
    run_data = _active_runs.get(run_id)
    if run_data is None:
        return False, f"Run not found: {run_id}"

    process = run_data["process"]
    if run_data["status"] != "running":
        return False, f"Run is not running (status: {run_data['status']})"

    try:
        if process.poll() is None:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(0.5)
            if process.poll() is None:
                os.killpg(pgid, signal.SIGKILL)

        release_run_slot()
        mark_run_complete(run_id, status="killed", exit_code=-1)
        return True, f"Killed run {run_id}"
    except Exception as e:
        logger.error("Failed to kill run %s: %s", run_id, e, exc_info=True)
        return False, f"Failed to kill run: {e}"


def delete_run(run_id: str) -> tuple[bool, str]:
    run_data = _active_runs.get(run_id)
    if run_data is None:
        return False, f"Run not found: {run_id}"
    if run_data["status"] == "running":
        return False, "Cannot delete running process. Kill it first."
    with _run_lock:
        del _active_runs[run_id]
    return True, f"Deleted run {run_id}"
