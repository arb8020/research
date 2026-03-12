from __future__ import annotations

import pytest

from rollouts.agents import AgentState
from rollouts.agents.types import Actor
from rollouts.core import Endpoint, Message, Trajectory
from rollouts.environments.kernelbench_multi import KernelBenchMultiTurnEnvironment


class FakeKernelEvaluator:
    def __init__(self) -> None:
        self.start_calls = 0
        self.score_calls: list[tuple[str, str, float]] = []
        self.runtime_provenance = {
            "hostname": "gpu-box-1",
            "machine": "x86_64",
            "torch": {
                "version": "2.8.0",
                "cuda_version": "12.4",
                "device_name": "NVIDIA A100",
                "device_capability": [8, 0],
            },
        }

    async def start(self) -> None:
        self.start_calls += 1

    async def score_one(
        self,
        kernel_code: str,
        ref_code: str,
        timeout: float,
    ) -> dict[str, float | str | None]:
        self.score_calls.append((kernel_code, ref_code, timeout))
        return {
            "compiled": 1.0,
            "correct": 1.0,
            "speedup": 1.75,
            "pass_rate": 1.0,
            "error": None,
            "runtime_provenance": self.runtime_provenance,
        }


@pytest.mark.trio
async def test_kernelbench_uses_injected_evaluator() -> None:
    evaluator = FakeKernelEvaluator()
    env = KernelBenchMultiTurnEnvironment(
        ref_code="def ref(): pass",
        evaluator=evaluator,
    )

    await env.on_session_start("session-1")
    result = await env._evaluate_kernel("class ModelNew: pass")

    assert evaluator.start_calls == 1
    assert evaluator.score_calls == [("class ModelNew: pass", "def ref(): pass", 120.0)]
    assert result == {
        "compiled": True,
        "correct": True,
        "speedup": 1.75,
        "pass_rate": 1.0,
        "error": None,
        "runtime_provenance": evaluator.runtime_provenance,
    }


@pytest.mark.trio
async def test_kernelbench_persists_evaluator_provenance_in_trajectory_metadata() -> None:
    evaluator = FakeKernelEvaluator()
    env = KernelBenchMultiTurnEnvironment(
        ref_code="def ref(): pass",
        evaluator=evaluator,
    )
    state = AgentState(
        actor=Actor(
            trajectory=Trajectory(messages=[]),
            endpoint=Endpoint(
                model="anthropic/test-model",
                base_url="https://api.anthropic.com/v1",
                api_format="anthropic-messages",
            ),
            tools=[],
        ),
        environment=env,
    )

    next_state = await env.on_assistant_message(
        Message(
            role="assistant",
            content="```python\nclass ModelNew: pass\n```",
        ),
        state,
    )

    assert env.evaluator_provenance == evaluator.runtime_provenance
    assert (
        next_state.actor.trajectory.metadata["evaluator_provenance"] == evaluator.runtime_provenance
    )
    assert next_state.actor.trajectory.metadata["turn_history"] == [
        {
            "turn": 1,
            "has_code": True,
            "compiled": True,
            "correct": True,
            "speedup": 1.75,
            "error": None,
            "runtime_provenance": evaluator.runtime_provenance,
        }
    ]
