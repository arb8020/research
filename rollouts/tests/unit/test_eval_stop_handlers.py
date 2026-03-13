from __future__ import annotations

import time
from dataclasses import replace

from rollouts.agents import Actor, AgentState, compose_handlers, handle_stop_max_turns
from rollouts.core import Endpoint, Message, Trajectory
from rollouts.dtypes import ChatCompletion, Choice, Cost, StopReason, Usage
from rollouts.eval.configs import (
    CostBudgetStop,
    EvalRunConfig,
    MaxTurnsStop,
    TokenBudgetStop,
    WallClockStop,
)
from rollouts.eval.run import _build_stop_handler


def _state_with_usage(*, turn_idx: int = 0, usage: Usage | None = None) -> AgentState:
    trajectory = Trajectory(messages=[Message(role="user", content="hello")])
    if usage is not None:
        trajectory = replace(
            trajectory,
            completions=[
                ChatCompletion(
                    id="c1",
                    object="chat.completion",
                    created=0,
                    model="anthropic/test-model",
                    usage=usage,
                    choices=[
                        Choice(
                            index=0,
                            message=Message(role="assistant", content="done"),
                            finish_reason="stop",
                        )
                    ],
                )
            ],
        )
    return AgentState(
        actor=Actor(
            trajectory=trajectory,
            endpoint=Endpoint(
                model="anthropic/test-model",
                base_url="https://api.anthropic.com/v1",
                api_format="anthropic-messages",
            ),
            tools=[],
        ),
        environment=None,
        turn_idx=turn_idx,
    )


def test_eval_run_config_defaults_to_max_turns_handler() -> None:
    run_config = EvalRunConfig(max_turns=2)
    handler = _build_stop_handler(run_config)

    result = handler(_state_with_usage(turn_idx=2))

    assert result.stop == StopReason.MAX_TURNS


def test_eval_stop_handlers_support_token_budget() -> None:
    run_config = EvalRunConfig(stop_handler=TokenBudgetStop(5))
    handler = _build_stop_handler(run_config)

    result = handler(
        _state_with_usage(
            usage=Usage(input_tokens=3, output_tokens=4),
        )
    )

    assert result.stop == StopReason.BUDGET_EXCEEDED


def test_eval_stop_handlers_support_cost_budget() -> None:
    run_config = EvalRunConfig(stop_handler=CostBudgetStop(0.5))
    handler = _build_stop_handler(run_config)

    result = handler(
        _state_with_usage(
            usage=Usage(cost=Cost(input=0.3, output=0.4)),
        )
    )

    assert result.stop == StopReason.BUDGET_EXCEEDED


def test_eval_stop_handlers_support_wall_clock_budget() -> None:
    run_config = EvalRunConfig(stop_handler=WallClockStop(0.001))
    handler = _build_stop_handler(run_config)
    time.sleep(0.01)

    result = handler(_state_with_usage())

    assert result.stop == StopReason.BUDGET_EXCEEDED


def test_eval_stop_handlers_accept_precomposed_handler() -> None:
    def custom_stop(state: AgentState) -> AgentState:
        return replace(state, stop=StopReason.TASK_COMPLETED)

    run_config = EvalRunConfig(
        stop_handler=compose_handlers([handle_stop_max_turns(10), custom_stop])
    )
    handler = _build_stop_handler(run_config)

    result = handler(_state_with_usage(turn_idx=0))

    assert result.stop == StopReason.TASK_COMPLETED
