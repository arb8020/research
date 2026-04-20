from __future__ import annotations

from pathlib import Path

import pytest

from rollouts.core import Message, Metric, Score, Trajectory
from rollouts.eval import AgentRunSpec, EvalOutputConfig, EvalRunConfig, EvalTaskSpec
from rollouts.eval.configs import ExternalEndpoint
from rollouts.serving.configs import EvalServingWorkload, ServingOutputConfig, ServingScenario
from rollouts.serving.run import _run_scenario
from rollouts.training.scoring import FunctionScorer
from rollouts.training.types import DatasetRow, RowAttempt


def _score_exact_match(sample: RowAttempt, _context: object) -> Score:
    predicted = float(sample.response)
    ground_truth = float(sample.ground_truth)
    return Score(
        metrics=(
            Metric("correct", 1.0 if abs(predicted - ground_truth) < 0.01 else 0.0, weight=1.0),
        )
    )


async def _attempt_executor(
    sample_data: dict[str, object],
    sample_id: str,
    environment: object,
    run_config: object,
) -> RowAttempt:
    del environment, run_config
    return RowAttempt(
        attempt_id=sample_id,
        problem=DatasetRow(
            problem_id=str(sample_data["id"]),
            payload=sample_data,
            ground_truth=sample_data["answer"],
        ),
        trajectory=Trajectory(
            messages=[
                Message(role="user", content=str(sample_data["prompt"])),
                Message(role="assistant", content=str(sample_data["answer"])),
            ],
            metadata={"executor": "test"},
        ),
        metadata={"status": "success", "turns_used": 1},
    )


@pytest.mark.trio
async def test_run_scenario_writes_per_workload_eval_artifacts(tmp_path: Path) -> None:
    eval_task = EvalTaskSpec(
        tasks=[
            {"id": "one", "prompt": "1+1", "answer": 2},
            {"id": "two", "prompt": "2+2", "answer": 4},
        ],
        run_spec=AgentRunSpec(
            endpoint=ExternalEndpoint(
                url="http://localhost:30000/v1",
                model="dummy",
                provider="sglang",
            ),
            attempt_executor=_attempt_executor,
        ),
        scorer=FunctionScorer(_score_exact_match),
        run=EvalRunConfig(
            max_concurrent=1,
            max_samples=2,
            max_turns=1,
            verbose=False,
            show_progress=False,
        ),
        output=EvalOutputConfig(experiment_name="base_eval"),
    )

    scenario = ServingScenario(
        endpoint=ExternalEndpoint(
            url="http://localhost:30000/v1",
            model="dummy",
            provider="sglang",
        ),
        workloads=[
            EvalServingWorkload(name="single_turn", eval_task=eval_task, concurrency=1),
            EvalServingWorkload(name="calculator", eval_task=eval_task, concurrency=1),
        ],
        output=ServingOutputConfig(experiment_name="serving_smoke"),
    )

    output_dir = tmp_path / "serving_run"
    report = await _run_scenario(
        config_path=tmp_path / "serving_config.py",
        scenario=scenario,
        output_dir=output_dir,
    )

    assert report["completed_workloads"] == 2
    assert report["total_samples"] == 4
    assert (output_dir / "scenario_manifest.json").exists()
    assert (output_dir / "scenario_report.json").exists()
    assert (output_dir / "workloads" / "single_turn" / "report.json").exists()
    assert (output_dir / "workloads" / "single_turn" / "events.jsonl").exists()
    assert (output_dir / "workloads" / "calculator" / "report.json").exists()
    assert (output_dir / "workloads" / "calculator" / "events.jsonl").exists()
