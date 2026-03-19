from dataclasses import replace

import pytest

from rollouts.agents import Actor, AgentState
from rollouts.core import EvalConfig, Metric, Score
from rollouts.dtypes import Message, Trajectory
from rollouts.eval.native import EvalRuntime, _AgentRunResult, evaluate_sample
from rollouts.training.types import AttemptResult, ScoringContext


class _FakeEnvironment:
    def __init__(self, *, best_speedup: float, has_correct_kernel: bool) -> None:
        self.best_speedup = best_speedup
        self.has_correct_kernel = has_correct_kernel

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


class _ContextualScorer:
    async def score(self, result: AttemptResult, context: ScoringContext) -> Score:
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
