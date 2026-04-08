from dataclasses import replace

import pytest

from rollouts.agents import Actor, AgentState
from rollouts.core import EvalConfig, Metric, Score
from rollouts.dtypes import Message, StopReason, Trajectory
from rollouts.eval.native import (
    EvalRuntime,
    _AgentRunResult,
    compute_summary_metrics,
    evaluate_sample,
)
from rollouts.training.types import RowAttempt, ScoringContext, Status


class _FakeEnvironment:
    def __init__(
        self,
        *,
        best_speedup: float,
        has_correct_kernel: bool,
        finalize_best_speedup: float | None = None,
        finalize_has_correct_kernel: bool | None = None,
    ) -> None:
        self.best_speedup = best_speedup
        self.has_correct_kernel = has_correct_kernel
        self.finalize_calls = 0
        self.finalize_best_speedup = finalize_best_speedup
        self.finalize_has_correct_kernel = finalize_has_correct_kernel

    def get_tools(self) -> list[object]:
        return []

    async def serialize(self) -> dict:
        return {
            "best_speedup": self.best_speedup,
            "has_correct_kernel": self.has_correct_kernel,
        }

    def get_runtime_metadata(self) -> dict:
        return {
            "best_speedup": self.best_speedup,
            "has_correct_kernel": self.has_correct_kernel,
            "turn_history": [],
        }

    async def finalize_attempt(self, **_: object) -> None:
        self.finalize_calls += 1
        if self.finalize_best_speedup is not None:
            self.best_speedup = self.finalize_best_speedup
        if self.finalize_has_correct_kernel is not None:
            self.has_correct_kernel = self.finalize_has_correct_kernel


class _ContextualScorer:
    async def score(self, result: RowAttempt, context: ScoringContext) -> Score:
        del result
        metadata = context.environment.get_runtime_metadata()
        correct = 1.0 if metadata["has_correct_kernel"] else 0.0
        speedup = float(metadata["best_speedup"])
        return Score(
            metrics=(
                Metric("reward", speedup, weight=1.0),
                Metric("correct", correct, weight=0.0),
                Metric("speedup", speedup, weight=0.0),
            )
        )


@pytest.mark.trio
async def test_evaluate_sample_scores_against_final_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_env = _FakeEnvironment(best_speedup=0.0, has_correct_kernel=False)
    final_env = _FakeEnvironment(best_speedup=1.25, has_correct_kernel=True)

    stale_trajectory = Trajectory(
        messages=[Message(role="assistant", content="done")],
        metadata={"best_speedup": 0.0, "has_correct_kernel": False},
    )
    final_state = AgentState(
        actor=Actor(trajectory=stale_trajectory, endpoint=None, tools=[]),
        environment=final_env,
    )

    async def _fake_run_agent_with_error_handling(
        initial_state: AgentState,
        run_config: object,
        sample_id: str,
    ) -> _AgentRunResult:
        del run_config, sample_id
        return _AgentRunResult(
            states=[initial_state, replace(final_state, turn_idx=1)],
            final_trajectory=stale_trajectory,
        )

    monkeypatch.setattr(
        "rollouts.eval.native._run_agent_with_error_handling",
        _fake_run_agent_with_error_handling,
    )

    config = EvalConfig(
        endpoint=None,
        prepare_messages=lambda _: [Message(role="user", content="hi")],
        scorer=_ContextualScorer(),
        verbose=False,
        show_progress=False,
    )
    runtime = EvalRuntime(config=config)

    result = await evaluate_sample(
        sample_data={"name": "sample"},
        sample_id="sample_0000",
        runtime=runtime,
        environment=initial_env,
    )

    assert result.score is not None
    assert result.reward == 1.25
    assert result.metadata["has_correct_kernel"] is True
    assert result.metadata["best_speedup"] == 1.25
    assert result.environment_state is not None
    assert result.environment_state["has_correct_kernel"] is True


@pytest.mark.trio
async def test_evaluate_sample_finalizes_environment_before_serialization_and_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_env = _FakeEnvironment(best_speedup=0.0, has_correct_kernel=False)
    final_env = _FakeEnvironment(
        best_speedup=0.0,
        has_correct_kernel=False,
        finalize_best_speedup=2.0,
        finalize_has_correct_kernel=True,
    )

    stale_trajectory = Trajectory(
        messages=[Message(role="assistant", content="done")],
        metadata={},
    )
    final_state = AgentState(
        actor=Actor(trajectory=stale_trajectory, endpoint=None, tools=[]),
        environment=final_env,
    )

    async def _fake_run_agent_with_error_handling(
        initial_state: AgentState,
        run_config: object,
        sample_id: str,
    ) -> _AgentRunResult:
        del run_config, sample_id
        return _AgentRunResult(
            states=[initial_state, replace(final_state, turn_idx=1)],
            final_trajectory=stale_trajectory,
        )

    monkeypatch.setattr(
        "rollouts.eval.native._run_agent_with_error_handling",
        _fake_run_agent_with_error_handling,
    )

    config = EvalConfig(
        endpoint=None,
        prepare_messages=lambda _: [Message(role="user", content="hi")],
        scorer=_ContextualScorer(),
        verbose=False,
        show_progress=False,
    )
    runtime = EvalRuntime(config=config)

    result = await evaluate_sample(
        sample_data={"name": "sample"},
        sample_id="sample_0000",
        runtime=runtime,
        environment=initial_env,
    )

    assert final_env.finalize_calls == 1
    assert result.reward == 2.0
    assert result.metadata["has_correct_kernel"] is True
    assert result.metadata["best_speedup"] == 2.0
    assert result.environment_state == {
        "best_speedup": 2.0,
        "has_correct_kernel": True,
    }


@pytest.mark.trio
async def test_evaluate_sample_marks_aborted_runs_honestly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _FakeEnvironment(best_speedup=0.0, has_correct_kernel=False)
    aborted_trajectory = Trajectory(
        messages=[Message(role="assistant", content="partial")],
        metadata={},
    )
    aborted_state = AgentState(
        actor=Actor(trajectory=aborted_trajectory, endpoint=None, tools=[]),
        environment=environment,
        stop=StopReason.ABORTED,
    )

    async def _fake_run_agent_with_error_handling(
        initial_state: AgentState,
        run_config: object,
        sample_id: str,
    ) -> _AgentRunResult:
        del run_config, sample_id
        return _AgentRunResult(
            states=[initial_state, replace(aborted_state, turn_idx=0)],
            final_trajectory=aborted_trajectory,
        )

    async def _fail_if_scored(result: RowAttempt, context: ScoringContext) -> Score:
        del result, context
        raise AssertionError("aborted samples should not be scored")

    monkeypatch.setattr(
        "rollouts.eval.native._run_agent_with_error_handling",
        _fake_run_agent_with_error_handling,
    )

    config = EvalConfig(
        endpoint=None,
        prepare_messages=lambda _: [Message(role="user", content="hi")],
        scorer=_fail_if_scored,
        verbose=False,
        show_progress=False,
    )
    runtime = EvalRuntime(config=config)

    result = await evaluate_sample(
        sample_data={"name": "sample"},
        sample_id="sample_0000",
        runtime=runtime,
        environment=environment,
    )

    assert result.metadata["status"] == "aborted"
    assert result.status == Status.ABORTED
    assert result.score is None
    assert result.reward == 0.0


def test_compute_summary_metrics_excludes_aborted_from_completion_and_success() -> None:
    aborted = RowAttempt(
        attempt_id="aborted",
        status=Status.ABORTED,
        metadata={"status": "aborted", "turns_used": 0, "total_tokens": 10},
    )
    failed = RowAttempt(
        attempt_id="failed",
        status=Status.COMPLETED,
        metadata={
            "status": "failed",
            "error": "ValueError: boom",
            "turns_used": 1,
            "total_tokens": 20,
        },
    )
    success = RowAttempt(
        attempt_id="success",
        status=Status.COMPLETED,
        metadata={"status": "success", "turns_used": 2, "total_tokens": 30},
    )
    success.score = Score(metrics=(Metric("reward", 1.0, weight=1.0),))

    summary = compute_summary_metrics([aborted, failed, success])

    assert summary["aborted_samples"] == 1
    assert summary["failed_samples"] == 1
    assert summary["successful_samples"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["completion_rate"] == 2 / 3
