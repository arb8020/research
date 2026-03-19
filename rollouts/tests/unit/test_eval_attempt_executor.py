from __future__ import annotations

from typing import NoReturn

import pytest
from rollouts.core import EvalConfig, Message, Metric, Score, Trajectory
from rollouts.eval.native import EvalRuntime, evaluate_sample
from rollouts.training.types import AttemptRow


class _FakeEnvironment:
    async def serialize(self) -> dict:
        return {"runtime": "fake"}

    def get_runtime_metadata(self) -> dict:
        return {"best_speedup": 1.5}


@pytest.mark.trio
async def test_evaluate_sample_accepts_direct_attempt_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fail_if_called(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("run_agent path should not be used when execute_attempt is set")

    monkeypatch.setattr(
        "rollouts.eval.native._run_agent_with_error_handling",
        _fail_if_called,
    )

    async def _execute_attempt(
        sample_data: dict[str, object],
        sample_id: str,
        environment: object,
        run_config: object,
    ) -> AttemptRow:
        del run_config
        assert sample_data["text"] == "hello"
        assert sample_id == "sample_0000"
        assert environment is not None
        return AttemptRow(
            trajectory=Trajectory(
                messages=[
                    Message(role="user", content="Reverse hello"),
                    Message(role="assistant", content="olleh"),
                ],
                metadata={"executor": "direct"},
            ),
            metadata={"status": "success", "turns_used": 3},
        )

    config = EvalConfig(
        endpoint=None,
        prepare_messages=None,
        attempt_executor=_execute_attempt,
        score_fn=lambda trajectory, sample_data: Score(
            metrics=(
                Metric(
                    "exact_match",
                    1.0
                    if trajectory.messages[-1].content == "olleh" and sample_data["text"] == "hello"
                    else 0.0,
                    weight=1.0,
                ),
            )
        ),
        verbose=False,
        show_progress=False,
    )
    runtime = EvalRuntime(config=config)

    result = await evaluate_sample(
        sample_data={"text": "hello", "metadata": {"source": "test"}},
        sample_id="sample_0000",
        runtime=runtime,
        environment=_FakeEnvironment(),
    )

    assert result.reward == 1.0
    assert result.attempt_id == "sample_0000"
    assert result.response == "olleh"
    assert result.metadata["executor"] == "direct"
    assert result.metadata["best_speedup"] == 1.5
    assert result.metadata["turns_used"] == 3
    assert result.environment_state == {"runtime": "fake"}
