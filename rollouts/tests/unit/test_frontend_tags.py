from __future__ import annotations

import json
from pathlib import Path

from rollouts.frontend.tags import build_run_tags, derive_run_tags, update_user_tags


def test_update_user_tags_writes_and_deletes_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    updated = update_user_tags(run_dir, {"keeper": "true", "bucket": "kernelbench"})
    assert updated == {"keeper": "true", "bucket": "kernelbench"}

    updated = update_user_tags(run_dir, {"keeper": None})
    assert updated == {"bucket": "kernelbench"}

    assert json.loads((run_dir / "tags.json").read_text()) == {"bucket": "kernelbench"}


def test_derive_run_tags_marks_successful_completed_runs() -> None:
    report = {
        "eval_name": "kernelbench_v3",
        "total_samples": 2,
        "config": {
            "endpoint": {"provider": "anthropic", "model": "claude-sonnet"},
            "interrupted": False,
        },
        "summary_metrics": {
            "completion_rate": 1.0,
            "success_rate": 1.0,
            "provider_errors": 0,
            "failed_samples": 0,
            "aborted_samples": 0,
        },
    }

    derived = derive_run_tags(report)
    assert derived["status"] == "completed"
    assert derived["completed"] == "true"
    assert derived["successful"] == "true"
    assert derived["provider"] == "anthropic"
    assert derived["model"] == "claude-sonnet"


def test_build_run_tags_combines_user_and_derived(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tags.json").write_text('{"owner":"chiraag"}\n')

    report = {
        "config": {"interrupted": True},
        "summary_metrics": {"aborted_samples": 1, "total_samples": 1},
        "total_samples": 1,
    }

    tags = build_run_tags(run_dir, report)
    assert tags["user"] == {"owner": "chiraag"}
    assert tags["derived"]["status"] == "aborted"
    assert tags["derived"]["interrupted"] == "true"
