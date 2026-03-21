from __future__ import annotations

import io
import json
from pathlib import Path

from pytest import MonkeyPatch

from rollouts.frontend import live_streams


class _FakeStdout:
    def __init__(self, text: str = "") -> None:
        self._text = text
        self._index = 0

    def read(self, size: int = 1) -> str:
        if self._index >= len(self._text):
            return ""
        chunk = self._text[self._index : self._index + size]
        self._index += len(chunk)
        return chunk


class _FakeProcess:
    def __init__(self, text: str = "", exit_code: int = 0) -> None:
        self.stdout = _FakeStdout(text)
        self._exit_code = exit_code

    def wait(self) -> int:
        return self._exit_code


class _FakeHandler:
    def __init__(self) -> None:
        self.wfile = io.BytesIO()

    def send_response(self, code: int) -> None:
        return

    def send_header(self, key: str, value: str) -> None:
        return

    def end_headers(self) -> None:
        return


def _read_sse_payloads(handler: _FakeHandler) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for chunk in handler.wfile.getvalue().decode().split("\n\n"):
        if not chunk.startswith("data: "):
            continue
        payloads.append(json.loads(chunk[len("data: ") :]))
    return payloads


def test_normalize_watch_event_preserves_sample_start_messages() -> None:
    event = live_streams._normalize_watch_event({
        "message": "sample_start",
        "timestamp": "2026-03-20T12:00:00",
        "sample_id": "sample_0000",
        "sample_name": "sample_0000",
        "sample_data": {"prompt": "solve it"},
        "messages": [{"role": "user", "content": "hello"}],
    })

    assert event == {
        "type": "sample_start",
        "timestamp": "2026-03-20T12:00:00",
        "id": "sample_0000",
        "name": "sample_0000",
        "sample_data": {"prompt": "solve it"},
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_stream_registered_run_replays_normalized_events_from_existing_results_dir(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_123"
    run_dir.mkdir()
    events_path = run_dir / "events.jsonl"
    events_path.write_text(
        "\n".join([
            json.dumps({
                "message": "sample_start",
                "timestamp": "2026-03-20T12:00:00",
                "sample_id": "sample_0000",
                "sample_name": "sample_0000",
                "sample_data": {"prompt": "solve it"},
                "messages": [{"role": "user", "content": "hello"}],
            }),
            json.dumps({
                "message": "assistant_message",
                "timestamp": "2026-03-20T12:00:01",
                "sample_id": "sample_0000",
                "turn": 0,
                "content": "working",
            }),
        ])
    )

    run_data = {
        "process": _FakeProcess(""),
        "config_name": "test",
        "start_time": 0.0,
        "status": "running",
        "output_lines": [f"📂 Results directory: {run_dir}"],
        "exit_code": None,
    }

    monkeypatch.setattr(live_streams, "has_run", lambda run_id: True)
    monkeypatch.setattr(live_streams, "get_run", lambda run_id: run_data)
    monkeypatch.setattr(live_streams, "release_run_slot", lambda: None)
    monkeypatch.setattr(
        live_streams,
        "mark_run_complete",
        lambda run_id, *, status, exit_code: None,
    )

    handler = _FakeHandler()
    live_streams.stream_registered_run(handler, "run_123")
    payloads = _read_sse_payloads(handler)

    assert payloads[0] == {
        "type": "stdout",
        "line": f"📂 Results directory: {run_dir}",
    }
    assert payloads[1] == {
        "type": "sample_start",
        "timestamp": "2026-03-20T12:00:00",
        "id": "sample_0000",
        "name": "sample_0000",
        "sample_data": {"prompt": "solve it"},
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert payloads[2] == {
        "type": "assistant_message",
        "timestamp": "2026-03-20T12:00:01",
        "sample_id": "sample_0000",
        "turn": 0,
        "content": "working",
    }
    assert payloads[-1] == {
        "type": "complete",
        "exit_code": 0,
        "status": "success",
    }
