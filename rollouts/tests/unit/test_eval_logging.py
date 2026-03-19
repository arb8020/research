from __future__ import annotations

import json
import logging
from pathlib import Path

from rollouts._logging import setup_eval_logging


def test_setup_eval_logging_writes_only_events_stream(tmp_path: Path) -> None:
    ctx = setup_eval_logging(tmp_path)
    logger = logging.getLogger("rollouts.eval.events")

    try:
        logger.info("sample_start", extra={"sample_id": "sample_0000"})
        logger.info(
            "raw_driver_line",
            extra={
                "sample_id": "sample_0000",
                "driver": "claude",
                "raw_line": '{"type":"assistant"}',
            },
        )
        logger.info("sample_end", extra={"sample_id": "sample_0000", "score": 1.0})
    finally:
        ctx.teardown()

    events_path = tmp_path / "events.jsonl"
    assert events_path.exists()

    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    assert [event["message"] for event in events] == [
        "sample_start",
        "raw_driver_line",
        "sample_end",
    ]

    assert not (tmp_path / "samples" / "sample_0000.jsonl").exists()
