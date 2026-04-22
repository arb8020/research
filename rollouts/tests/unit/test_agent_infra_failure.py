from __future__ import annotations

from typing import Any, NoReturn

import pytest

from rollouts.agents import AgentState, RunConfig
from rollouts.agents.runtime import process_pending_tools
from rollouts.agents.types import Actor
from rollouts.core import Endpoint, Message, Tool, ToolFunction, ToolFunctionParameter, Trajectory
from rollouts.dtypes import StopReason, ToolCall
from rollouts.infra_errors import WorkspaceInfraError


class _FailingEnv:
    def get_tools(self) -> list[Tool]:
        return [
            Tool(
                function=ToolFunction(
                    name="write_kernel",
                    description="Write a kernel file",
                    parameters=ToolFunctionParameter(
                        properties={"kernel_code": {"type": "string"}}
                    ),
                    required=["kernel_code"],
                )
            )
        ]

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


class _SchemaEnv:
    def __init__(self) -> None:
        self.exec_calls = 0

    def get_tools(self) -> list[Tool]:
        return [
            Tool(
                function=ToolFunction(
                    name="write_file",
                    description="Write a file",
                    parameters=ToolFunctionParameter(
                        properties={
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        }
                    ),
                    required=["path", "content"],
                )
            )
        ]

    async def serialize(self) -> dict:
        return {}

    @classmethod
    async def deserialize(cls, data: dict) -> _SchemaEnv:
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
        self.exec_calls += 1
        raise AssertionError("schema-invalid tool call should not execute")


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
            tools=_FailingEnv().get_tools(),
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


@pytest.mark.trio
async def test_process_pending_tools_rejects_schema_invalid_tool_calls_before_execution() -> None:
    events: list[object] = []

    async def _on_chunk(event: object) -> None:
        events.append(event)

    environment = _SchemaEnv()
    state = AgentState(
        actor=Actor(
            trajectory=Trajectory(messages=[Message(role="assistant", content="tool")]),
            endpoint=Endpoint(
                model="anthropic/test-model",
                base_url="https://api.anthropic.com/v1",
                api_format="anthropic-messages",
            ),
            tools=environment.get_tools(),
        ),
        environment=environment,
        pending_tool_calls=[ToolCall(id="tc-1", name="write_file", args={"content": "hello"})],
    )

    result = await process_pending_tools(state, RunConfig(on_chunk=_on_chunk))

    dispatch_events = [
        event
        for event in events
        if getattr(event, "type", None) == "tool_call_dispatch"
        and getattr(event, "data", {}).get("action") == "schema_error"
    ]

    assert environment.exec_calls == 0
    assert len(dispatch_events) == 1
    assert "required property" in dispatch_events[0].data["error"]
    assert result.pending_tool_calls == []
    assert result.actor.trajectory.messages[-1].role == "tool"
    assert result.actor.trajectory.messages[-1].is_error is True
    assert result.actor.trajectory.messages[-1].error is not None
    assert "required property" in result.actor.trajectory.messages[-1].error
