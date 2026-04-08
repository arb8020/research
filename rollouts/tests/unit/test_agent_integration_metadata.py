from __future__ import annotations

import pytest

from rollouts.agents import Actor, AgentState
from rollouts.core import Message, StopReason, Trajectory
from rollouts.training.agent_integration import agent_rollout_to_sample
from rollouts.training.types import DatasetRow, RowAttempt, TrainingSample


class _FakeEnvironment:
    def get_tools(self) -> list[object]:
        return []


class _FinalizingEnvironment:
    def __init__(self) -> None:
        self.finalize_calls = 0
        self.best_speedup = 0.0
        self.turn_history: list[dict[str, object]] = []

    def get_tools(self) -> list[object]:
        return []

    async def finalize_attempt(self, **_: object) -> None:
        self.finalize_calls += 1
        self.best_speedup = 2.0
        self.turn_history = [{"turn": 1, "compiled": True, "correct": True}]

    def get_runtime_metadata(self) -> dict[str, object]:
        return {
            "best_speedup": self.best_speedup,
            "has_correct_kernel": self.best_speedup > 0,
            "turn_history": list(self.turn_history),
            "evaluator_provenance": {"runtime": "finalized-env"},
        }


@pytest.mark.trio
async def test_agent_rollout_to_sample_preserves_final_trajectory_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_trajectory = Trajectory(
        messages=[
            Message(role="user", content="optimize this"),
            Message(role="assistant", content="```python\nclass ModelNew: pass\n```"),
        ],
        metadata={
            "best_speedup": 1.5,
            "has_correct_kernel": True,
            "turn_history": [{"turn": 1, "compiled": True}],
            "evaluator_provenance": {"torch": {"cuda_available": True}},
        },
    )
    final_state = AgentState(
        actor=Actor(trajectory=final_trajectory, endpoint=None, tools=[]),
        environment=_FakeEnvironment(),
        turn_idx=1,
        stop=StopReason.TASK_COMPLETED,
    )

    async def _fake_run_agent(
        state: AgentState,
        run_config: object,
    ) -> list[AgentState]:
        del run_config
        return [state, final_state]

    captured: dict[str, object] = {}

    def _fake_trajectory_to_sample(
        trajectory: Trajectory,
        tokenizer: object,
        metadata: dict[str, object] | None = None,
        problem_row: DatasetRow | None = None,
    ) -> RowAttempt:
        del trajectory, tokenizer
        captured["metadata"] = dict(metadata or {})
        return RowAttempt(
            problem=problem_row,
            trajectory=final_trajectory,
            training_sample=TrainingSample(tokens=[1], loss_mask=[1.0], response_length=1),
            metadata=dict(metadata or {}),
        )

    monkeypatch.setattr("rollouts.training.agent_integration.run_agent", _fake_run_agent)
    monkeypatch.setattr(
        "rollouts.training.agent_integration.trajectory_to_sample",
        _fake_trajectory_to_sample,
    )

    sample = await agent_rollout_to_sample(
        prompt="optimize this",
        environment_cls=_FakeEnvironment,
        endpoint=object(),
        tokenizer=object(),
        max_turns=2,
        metadata={"ref_code": "class Model(torch.nn.Module): pass"},
        sample_data={"id": "kb-1", "messages": [{"role": "user", "content": "optimize this"}]},
    )

    assert sample.metadata["best_speedup"] == 1.5
    assert sample.metadata["has_correct_kernel"] is True
    assert sample.metadata["turn_history"] == [{"turn": 1, "compiled": True}]
    assert sample.metadata["evaluator_provenance"] == {"torch": {"cuda_available": True}}
    assert sample.metadata["ref_code"] == "class Model(torch.nn.Module): pass"
    assert sample.metadata["turns"] == 1
    assert sample.metadata["stop_reason"] == StopReason.TASK_COMPLETED.value
    assert captured["metadata"] == sample.metadata


@pytest.mark.trio
async def test_agent_rollout_to_sample_merges_final_environment_runtime_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _FinalizingEnvironment()
    final_trajectory = Trajectory(
        messages=[
            Message(role="user", content="optimize this"),
            Message(role="assistant", content="done"),
        ],
        metadata={},
    )
    final_state = AgentState(
        actor=Actor(trajectory=final_trajectory, endpoint=None, tools=[]),
        environment=environment,
        turn_idx=1,
        stop=StopReason.TASK_COMPLETED,
    )

    async def _fake_run_agent(
        state: AgentState,
        run_config: object,
    ) -> list[AgentState]:
        del state, run_config
        return [final_state]

    def _fake_trajectory_to_sample(
        trajectory: Trajectory,
        tokenizer: object,
        metadata: dict[str, object] | None = None,
        problem_row: DatasetRow | None = None,
    ) -> RowAttempt:
        del trajectory, tokenizer
        return RowAttempt(
            problem=problem_row,
            trajectory=final_trajectory,
            training_sample=TrainingSample(tokens=[1], loss_mask=[1.0], response_length=1),
            metadata=dict(metadata or {}),
        )

    monkeypatch.setattr("rollouts.training.agent_integration.run_agent", _fake_run_agent)
    monkeypatch.setattr(
        "rollouts.training.agent_integration.trajectory_to_sample",
        _fake_trajectory_to_sample,
    )

    sample = await agent_rollout_to_sample(
        prompt="optimize this",
        environment_cls=lambda: environment,
        endpoint=object(),
        tokenizer=object(),
        max_turns=2,
        sample_data={"id": "kb-2", "messages": [{"role": "user", "content": "optimize this"}]},
    )

    assert environment.finalize_calls == 1
    assert sample.metadata["best_speedup"] == 2.0
    assert sample.metadata["has_correct_kernel"] is True
    assert sample.metadata["turn_history"] == [{"turn": 1, "compiled": True, "correct": True}]
    assert sample.metadata["evaluator_provenance"] == {"runtime": "finalized-env"}
