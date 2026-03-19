from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rollouts.config_contracts import validate_eval_config_module
from rollouts.core import Metric, Score
from rollouts.eval import AgentRunSpec, EndpointConfig
from rollouts.training.types import AttemptResult


class _NoopScorer:
    async def score(self, result: AttemptResult, context: object) -> Score:
        del result, context
        return Score(metrics=(Metric("reward", 0.0, weight=1.0),))


def test_validate_eval_config_accepts_run_spec_without_prepare_messages() -> None:
    run_spec = AgentRunSpec(
        endpoint=EndpointConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
        prepare_messages=lambda sample: sample["messages"],
    )
    module = SimpleNamespace(
        tasks=[{"messages": []}],
        run_spec=run_spec,
        scorer=_NoopScorer(),
    )

    validate_eval_config_module(module, Path("configs/trusted/example.py"))


def test_validate_eval_config_rejects_non_agent_run_spec() -> None:
    module = SimpleNamespace(
        tasks=[{"messages": []}],
        run_spec=object(),
        scorer=_NoopScorer(),
    )

    with pytest.raises(ValueError, match="run_spec: AgentRunSpec"):
        validate_eval_config_module(module, Path("configs/trusted/example.py"))


def test_validate_eval_config_accepts_direct_attempt_executor() -> None:
    run_spec = AgentRunSpec(
        endpoint=None,
        attempt_executor=lambda sample, sample_id, environment, run_config: AttemptResult(
            attempt_id=str(sample.get("id", sample_id))
        ),
    )
    module = SimpleNamespace(
        tasks=[{"messages": []}],
        run_spec=run_spec,
        scorer=_NoopScorer(),
    )

    validate_eval_config_module(module, Path("configs/trusted/example.py"))


def test_validate_eval_config_rejects_missing_scoring_path() -> None:
    run_spec = AgentRunSpec(
        endpoint=EndpointConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
        prepare_messages=lambda sample: sample["messages"],
    )
    module = SimpleNamespace(
        tasks=[{"messages": []}],
        run_spec=run_spec,
    )

    with pytest.raises(ValueError, match="must define an explicit scorer"):
        validate_eval_config_module(module, Path("configs/trusted/example.py"))
