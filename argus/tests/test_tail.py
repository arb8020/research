from __future__ import annotations

import json

from argus import tail


def test_format_prefers_line_for_event_discriminated_service_logs() -> None:
    raw = json.dumps({
        "ts": "2026-04-16T02:27:21.564848",
        "event": "eval_inference_service_log",
        "line": "INFO:rollouts.inference.student.server:Loading model Qwen/Qwen3-0.6B ...",
    })

    text, done = tail._format(raw)

    assert done is False
    assert text == "INFO:rollouts.inference.student.server:Loading model Qwen/Qwen3-0.6B ..."


def test_format_uses_event_as_canonical_discriminator_for_eval_events() -> None:
    raw = json.dumps({
        "timestamp": "2026-04-16T02:27:28.942179+00:00",
        "event": "eval_start",
        "message": "stale_text_do_not_key_off_me",
        "eval_name": "student_server_eval_b200_ssh",
        "total": 4,
    })

    text, done = tail._format(raw)

    assert done is False
    assert text == "[eval] started  student_server_eval_b200_ssh  4 samples"
