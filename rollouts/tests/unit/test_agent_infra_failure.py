from __future__ import annotations

from typing import Any, NoReturn

import pytest

from rollouts.agents import AgentState, RunConfig
from rollouts.agents.runtime import process_pending_tools
from rollouts.agents.types import Actor
from rollouts.core import Endpoint, Message, Trajectory
from rollouts.dtypes import StopReason, ToolCall
from rollouts.infra_errors import WorkspaceInfraError


class _FailingEnv:
    def get_tools(self) -> list[object]:
        return []

    async def serialize(self) -> dict:
        return {}

    @classmethod
    async def deserialize(cls, data: dict) -> _FailingEnv:
        del data
        return cls()

    async def exec_tool(
        self,
        tool_call: Any,
        current_state: Any,
        run_config: Any,
        cancel_scope: Any = None,
    ) -> NoReturn:
        del tool_call, current_state, run_config, cancel_scope
        raise WorkspaceInfraError(
            "workspace command timed out after 120s", kind="workspace_timeout"
        )


@pytest.mark.trio
async def test_process_pending_tools_aborts_on_workspace_infra_error() -> None:
    events: list[object] = []

    async def _on_chunk(event: object) -> None:
        events.append(event)

    state = AgentState(
        actor=Actor(
            trajectory=Trajectory(messages=[Message(role="assistant", content="tool")]),
            endpoint=Endpoint(
                model="anthropic/test-model",
                base_url="https://api.anthropic.com/v1",
                api_format="anthropic-messages",
            ),
            tools=[],
        ),
        environment=_FailingEnv(),
        pending_tool_calls=[ToolCall(id="tc-1", name="write_kernel", args={"kernel_code": "x"})],
    )
    run_config = RunConfig(on_chunk=_on_chunk)

    result = await process_pending_tools(state, run_config)

    assert result.stop == StopReason.ERROR
    assert result.error == "workspace command timed out after 120s"
    assert result.pending_tool_calls == []
    assert any(getattr(event, "type", None) == "infra_failure_terminal" for event in events)
