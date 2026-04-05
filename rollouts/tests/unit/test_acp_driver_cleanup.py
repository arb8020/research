from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

acp = pytest.importorskip("acp")

from acp import InitializeResponse, PromptResponse

from rollouts.drivers.acp import ACPDriver, _ACPEventBridge
from rollouts.dtypes import (
    TextDelta,
    TextEnd,
    TextStart,
    ToolCallEnd,
    ToolCallStart,
    ToolResultReceived,
)


def test_acp_driver_does_not_fail_after_successful_prompt_if_teardown_hangs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _FakeConn:
        async def initialize(self, **_kwargs: Any) -> InitializeResponse:
            return InitializeResponse(protocolVersion=1)

        async def new_session(self, **_kwargs: Any) -> Any:
            return type("_Session", (), {"sessionId": "session-1"})()

        async def prompt(self, *_args: Any, **_kwargs: Any) -> PromptResponse:
            return PromptResponse(stopReason="end_turn")

    class _FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self._terminated = False
            self._killed = False

        def terminate(self) -> None:
            self._terminated = True
            self.returncode = 1

        def kill(self) -> None:
            self._killed = True
            self.returncode = 1

        async def wait(self) -> int:
            if self._killed:
                return 1
            if self._terminated:
                await asyncio.sleep(3600)
            return 0

    @asynccontextmanager
    async def _fake_spawn_agent_process(
        *_args: Any, **_kwargs: Any
    ) -> AsyncIterator[tuple[_FakeConn, _FakeProcess]]:
        yield _FakeConn(), _FakeProcess()

    monkeypatch.setattr(acp, "spawn_agent_process", _fake_spawn_agent_process)
    monkeypatch.setattr("rollouts.drivers.acp._ACP_TEARDOWN_TIMEOUT_SECONDS", 0.01)

    driver = ACPDriver(
        cwd=tmp_path,
        command=("fake-acp-agent",),
        provider="test",
        model="fake",
    )

    bridge, response = asyncio.run(driver._run_once_asyncio("hello"))

    assert bridge.events == []
    assert response.stopReason == "end_turn"


def test_acp_event_bridge_accumulates_text_chunks_until_tool_boundary() -> None:
    bridge = _ACPEventBridge(auto_approve_permissions=True)

    class _TextChunk:
        sessionUpdate = "agent_message_chunk"

        def __init__(self, text: str) -> None:
            self.content = type("_Content", (), {"type": "text", "text": text})()

    class _ToolUpdate:
        sessionUpdate = "tool_call_update"
        toolCallId = "tool-1"
        title = "shell"
        kind = "execute"
        status = "completed"
        rawInput = {"command": "echo hi"}
        rawOutput = {"stdout": "hi"}
        locations = None
        content = None

    async def _exercise() -> None:
        await bridge.session_update("session", _TextChunk("hello"), source="test")
        await bridge.session_update("session", _TextChunk(" world"), source="test")
        await bridge.session_update("session", _ToolUpdate(), source="test")

    asyncio.run(_exercise())

    event_types = [type(event) for event in bridge.events]
    assert event_types == [
        TextStart,
        TextDelta,
        TextDelta,
        TextEnd,
        ToolCallStart,
        ToolCallEnd,
        ToolResultReceived,
    ]

    text_end = next(event for event in bridge.events if isinstance(event, TextEnd))
    assert text_end.content == "hello world"
