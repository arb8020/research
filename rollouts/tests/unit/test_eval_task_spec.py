from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from rollouts.config_contracts import validate_eval_config_module
from rollouts.eval import AgentRunSpec, EndpointConfig, EvalTaskSpec, resolve_eval_task_spec


def test_validate_eval_config_accepts_eval_task() -> None:
    scorer = cast(Any, object())
    module = SimpleNamespace(
        eval_task=EvalTaskSpec(
            tasks=[{"messages": []}],
            run_spec=AgentRunSpec(
                endpoint=EndpointConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
                prepare_messages=lambda sample: sample["messages"],
            ),
            scorer=scorer,
        )
    )

    validate_eval_config_module(module, Path("configs/trusted/example.py"))


def test_resolve_eval_task_spec_normalizes_legacy_exports() -> None:
    scorer = cast(Any, object())
    module = SimpleNamespace(
        tasks=[{"messages": []}],
        endpoint=EndpointConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
        prepare_messages=lambda sample: sample["messages"],
        scorer=scorer,
    )

    eval_task = resolve_eval_task_spec(module)

    assert eval_task.tasks == [{"messages": []}]
    assert eval_task.run_spec.endpoint is not None
    assert eval_task.run_spec.endpoint.provider == "anthropic"
    assert eval_task.run_spec.prepare_messages is not None
    assert eval_task.scorer is not None


def test_resolve_eval_run_spec_lowers_external_runtime_prompt_builder() -> None:
    scorer = cast(Any, object())
    module = SimpleNamespace(
        run_spec=AgentRunSpec(
            external_runtime="codex",
            prompt_builder=lambda sample: f"prompt:{sample['text']}",
            external_agent_args={
                "cwd": Path("/tmp/workdir"),
                "model": "gpt-5.1-codex-mini",
                "sandbox": "workspace-write",
            },
        )
    )

    run_spec = resolve_eval_task_spec(
        SimpleNamespace(
            tasks=[{"text": "hello"}],
            run_spec=module.run_spec,
            scorer=scorer,
        )
    ).run_spec

    assert run_spec.attempt_executor is not None


def test_resolve_eval_run_spec_lowers_external_runtime_prepare_messages() -> None:
    scorer = cast(Any, object())
    module = SimpleNamespace(
        tasks=[{"text": "hello"}],
        run_spec=AgentRunSpec(
            external_runtime="claude_code",
            prepare_messages=lambda sample: [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": sample["text"]},
            ],
            external_agent_args={
                "cwd": Path("/tmp/workdir"),
                "model": "sonnet",
            },
        ),
        scorer=scorer,
    )

    run_spec = resolve_eval_task_spec(module).run_spec

    assert run_spec.attempt_executor is not None
