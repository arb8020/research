from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rollouts.event_log import build_jsonl_run_event_sinks, emit_logger_event, emit_run_event


def test_emit_logger_event_sets_canonical_event_on_log_record(caplog: Any) -> None:
    logger = logging.getLogger("rollouts.tests.event_log")

    with caplog.at_level(logging.INFO, logger=logger.name):
        emit_logger_event(logger, "sample_start", sample_id="sample_0001")

    record = caplog.records[-1]
    assert record.getMessage() == "sample_start"
    assert record.event == "sample_start"
    assert record.sample_id == "sample_0001"


def test_emit_logger_event_preserves_human_message_without_changing_event(caplog: Any) -> None:
    logger = logging.getLogger("rollouts.tests.event_log.message")

    with caplog.at_level(logging.INFO, logger=logger.name):
        emit_logger_event(
            logger,
            "inference_service_final_log",
            message="inference service final log\nhello",
        )

    record = caplog.records[-1]
    assert record.getMessage() == "inference service final log\nhello"
    assert record.event == "inference_service_final_log"


def test_build_jsonl_run_event_sinks_writes_event_and_calls_callback(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    callbacks: list[tuple[str, dict[str, object]]] = []

    run_event_sinks = build_jsonl_run_event_sinks(
        log_path,
        on_event=lambda event, data: callbacks.append((event, data)),
    )
    emit_run_event(run_event_sinks, "eval_start", eval_name="demo", total=2)

    written = json.loads(log_path.read_text().strip())
    assert written["event"] == "eval_start"
    assert written["eval_name"] == "demo"
    assert written["total"] == 2
    assert "ts" in written
    assert callbacks == [("eval_start", {"eval_name": "demo", "total": 2})]
