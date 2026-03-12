from __future__ import annotations

import json
from pathlib import Path

import pytest

from rollouts.core import Message, Metric, Score, Trajectory
from rollouts.eval.native import EvalReport
from rollouts.training.types import AttemptRow, ProblemRow


@pytest.mark.trio
async def test_eval_report_saves_compact_sample_and_separate_trajectory(tmp_path: Path) -> None:
    sample = AttemptRow(
        attempt_id="sample_0000",
        problem=ProblemRow(
            problem_id="sample_0000",
            payload={"messages": [{"role": "user", "content": "hi"}]},
        ),
        trajectory=Trajectory(
            messages=[
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello"),
            ],
            metadata={"sample_data": {"messages": [{"role": "user", "content": "hi"}]}},
        ),
        environment_state={"backend": "cuda"},
        metadata={
            "sample_data": {"messages": [{"role": "user", "content": "hi"}]},
            "turn_history": [{"turn": 1, "has_code": False}],
        },
        reward=1.0,
        score=Score(metrics=(Metric("reward", 1.0, weight=1.0),)),
    )

    report = EvalReport(
        eval_name="test_eval",
        dataset_path="test_eval",
        total_samples=1,
        summary_metrics={"mean_reward": 1.0},
        sample_results=[sample],
        config={},
    )

    await report.save(tmp_path)

    sample_json = json.loads((tmp_path / "samples" / "sample_0000.json").read_text())
    assert "trajectory" not in sample_json
    assert sample_json["trajectory_path"] == "trajectories/sample_0000.jsonl"
    assert "sample_data" not in sample_json["metadata"]

    trajectory_file = tmp_path / "trajectories" / "sample_0000.jsonl"
    assert trajectory_file.exists()
    trajectory_lines = trajectory_file.read_text().strip().splitlines()
    assert len(trajectory_lines) == 1
