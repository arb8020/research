from __future__ import annotations

import json
from pathlib import Path

import pytest

from rollouts.core import Message, Metric, Score, Trajectory
from rollouts.eval.native import EvalReport
from rollouts.training.types import AttemptEvaluation, AttemptResult, ProblemRow


@pytest.mark.trio
async def test_eval_report_saves_full_canonical_sample_artifact(tmp_path: Path) -> None:
    sample = AttemptResult(
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
        evaluation=AttemptEvaluation(
            reward=1.0,
            score=Score(metrics=(Metric("reward", 1.0, weight=1.0),)),
        ),
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
    report_html = (tmp_path / "report.html").read_text()
    sample_html = (tmp_path / "samples" / "sample_0000.html").read_text()
    assert sample_json["trajectory"]["messages"][1]["content"] == "hello"
    assert "sample_data" not in sample_json["metadata"]
    assert sample_json["evaluation"]["reward"] == 1.0
    assert sample_json["evaluation"]["score"]["metrics"][0]["name"] == "reward"
    assert not (tmp_path / "trajectories").exists()
    assert "sample_0000" in report_html
    assert 'href="samples/sample_0000.html"' in report_html
    assert "hello" in sample_html
