from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rollouts.config_contracts import validate_eval_config_module
from rollouts.eval import AgentRunSpec, EndpointConfig


def test_validate_eval_config_accepts_run_spec_without_prepare_messages() -> None:
    run_spec = AgentRunSpec(
        endpoint=EndpointConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
        prepare_messages=lambda sample: sample["messages"],
    )
    module = SimpleNamespace(
        tasks=[{"messages": []}],
        run_spec=run_spec,
        score_fn=lambda attempt: None,
    )

    validate_eval_config_module(module, Path("configs/trusted/example.py"))


def test_validate_eval_config_rejects_non_agent_run_spec() -> None:
    module = SimpleNamespace(
        tasks=[{"messages": []}],
        run_spec=object(),
        score_fn=lambda attempt: None,
    )

    with pytest.raises(ValueError, match="run_spec: AgentRunSpec"):
        validate_eval_config_module(module, Path("configs/trusted/example.py"))


def test_validate_eval_config_accepts_direct_attempt_executor() -> None:
    run_spec = AgentRunSpec(
        endpoint=None,
        execute_attempt=lambda sample, sample_id, environment, run_config: sample,
    )
    module = SimpleNamespace(
        tasks=[{"messages": []}],
        run_spec=run_spec,
        score_fn=lambda attempt: None,
    )

    validate_eval_config_module(module, Path("configs/trusted/example.py"))
