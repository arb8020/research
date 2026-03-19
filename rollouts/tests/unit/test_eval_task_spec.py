from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rollouts.config_contracts import validate_eval_config_module
from rollouts.eval import AgentRunSpec, EndpointConfig, EvalTaskSpec, resolve_eval_task_spec


def test_validate_eval_config_accepts_eval_task() -> None:
    module = SimpleNamespace(
        eval_task=EvalTaskSpec(
            tasks=[{"messages": []}],
            run_spec=AgentRunSpec(
                endpoint=EndpointConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
                prepare_messages=lambda sample: sample["messages"],
            ),
            scorer=object(),
        )
    )

    validate_eval_config_module(module, Path("configs/trusted/example.py"))


def test_resolve_eval_task_spec_normalizes_legacy_exports() -> None:
    module = SimpleNamespace(
        tasks=[{"messages": []}],
        endpoint=EndpointConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
        prepare_messages=lambda sample: sample["messages"],
        scorer=object(),
    )

    eval_task = resolve_eval_task_spec(module)

    assert eval_task.tasks == [{"messages": []}]
    assert eval_task.run_spec.endpoint is not None
    assert eval_task.run_spec.endpoint.provider == "anthropic"
    assert eval_task.run_spec.prepare_messages is not None
    assert eval_task.scorer is not None
