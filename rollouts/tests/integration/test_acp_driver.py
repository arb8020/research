from __future__ import annotations

import sys
from pathlib import Path

import pytest

acp = pytest.importorskip("acp")
pytest.importorskip("trio_asyncio")

from rollouts.drivers.acp import ACPDriver


@pytest.mark.integration
@pytest.mark.trio
async def test_acp_driver_emits_ordered_text_events(tmp_path: Path) -> None:
    agent_script = tmp_path / "echo_agent.py"
    agent_script.write_text(
        """
import asyncio
from typing import Any
from uuid import uuid4

from acp import Agent, InitializeResponse, NewSessionResponse, PromptResponse, run_agent, text_block, update_agent_message
from acp.interfaces import Client
from acp.schema import ClientCapabilities, Implementation, TextContentBlock


class EchoAgent(Agent):
    _conn: Client

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(self, protocol_version: int, client_capabilities: ClientCapabilities | None = None, client_info: Implementation | None = None, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(protocolVersion=protocol_version)

    async def new_session(self, cwd: str, mcp_servers: list[Any] | None = None, **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(sessionId=uuid4().hex)

    async def prompt(self, prompt: list[TextContentBlock], session_id: str, **kwargs: Any) -> PromptResponse:
        for block in prompt:
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message(text_block(f"echo:{block.text}")),
                source="echo",
            )
        return PromptResponse(stopReason="end_turn")


asyncio.run(run_agent(EchoAgent()))
""".strip(),
        encoding="utf-8",
    )

    driver = ACPDriver(
        cwd=tmp_path,
        command=(sys.executable, str(agent_script)),
        provider="test",
        model="echo",
    )

    seen: list[tuple[str, str | None]] = []
    async for event in driver.run("hello acp"):
        name = type(event).__name__
        payload = (
            getattr(event, "delta", None)
            or getattr(event, "content", None)
            or getattr(event, "finish_reason", None)
        )
        seen.append((name, payload))

    assert seen == [
        ("StreamStart", None),
        ("LLMCallStart", None),
        ("TextStart", None),
        ("TextDelta", "echo:hello acp"),
        ("TextEnd", "echo:hello acp"),
        ("LLMCallEnd", None),
        ("StreamDone", "end_turn"),
    ]


@pytest.mark.integration
@pytest.mark.trio
async def test_acp_driver_emits_file_edit_delete_tool_events(tmp_path: Path) -> None:
    agent_script = tmp_path / "haiku_agent.py"
    agent_script.write_text(
        """
import asyncio
from typing import Any
from uuid import uuid4

from acp import Agent, InitializeResponse, NewSessionResponse, PromptResponse, run_agent, text_block, update_agent_message
from acp.interfaces import Client
from acp.schema import (
    ClientCapabilities,
    ContentToolCallContent,
    FileEditToolCallContent,
    Implementation,
    TextContentBlock,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
)


class HaikuAgent(Agent):
    _conn: Client

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(self, protocol_version: int, client_capabilities: ClientCapabilities | None = None, client_info: Implementation | None = None, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(protocolVersion=protocol_version)

    async def new_session(self, cwd: str, mcp_servers: list[Any] | None = None, **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(sessionId=uuid4().hex)

    async def _emit_tool(
        self,
        *,
        session_id: str,
        tool_call_id: str,
        title: str,
        kind: str,
        raw_input: dict[str, Any],
        raw_output: dict[str, Any],
        content: list[Any] | None = None,
        locations: list[ToolCallLocation] | None = None,
        status: str = "completed",
    ) -> None:
        await self._conn.session_update(
            session_id=session_id,
            update=ToolCallStart(
                toolCallId=tool_call_id,
                title=title,
                kind=kind,
                status="in_progress",
                rawInput=raw_input,
                locations=locations,
                content=content,
                sessionUpdate="tool_call",
            ),
            source="haiku_agent",
        )
        await self._conn.session_update(
            session_id=session_id,
            update=ToolCallProgress(
                toolCallId=tool_call_id,
                title=title,
                kind=kind,
                status=status,
                rawInput=raw_input,
                rawOutput=raw_output,
                locations=locations,
                content=content,
                sessionUpdate="tool_call_update",
            ),
            source="haiku_agent",
        )

    async def prompt(self, prompt: list[TextContentBlock], session_id: str, **kwargs: Any) -> PromptResponse:
        for block in prompt:
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message(text_block(f"working:{block.text}")),
                source="haiku_agent",
            )

        await self._emit_tool(
            session_id=session_id,
            tool_call_id="mkdir-haikus",
            title="mkdir /haikus",
            kind="execute",
            raw_input={"command": "mkdir -p /haikus"},
            raw_output={"returncode": 0},
            content=[
                ContentToolCallContent(
                    type="content",
                    content=text_block("created /haikus"),
                )
            ],
        )

        file_payloads = [
            (
                "write-haiku-1",
                "/haikus/haiku1.txt",
                "old pond\\nfrog jumps in\\nwater sound\\n",
                None,
                1,
            ),
            (
                "write-haiku-2",
                "/haikus/haiku2.txt",
                "winter moon\\nquiet alley\\ncat watches\\n",
                None,
                1,
            ),
            (
                "write-haiku-3",
                "/haikus/haiku3.txt",
                "morning train\\nwindows full of rain\\nstation sighs\\n",
                None,
                1,
            ),
            (
                "edit-haiku-1",
                "/haikus/haiku1.txt",
                "old silent pond\\nfrog jumps in\\nwater sound\\n",
                "old pond\\nfrog jumps in\\nwater sound\\n",
                1,
            ),
            (
                "edit-haiku-2",
                "/haikus/haiku2.txt",
                "winter moon\\nempty alley\\ncat watches\\n",
                "winter moon\\nquiet alley\\ncat watches\\n",
                2,
            ),
            (
                "edit-haiku-3",
                "/haikus/haiku3.txt",
                "morning train\\nwindows full of rain\\nstation exhales\\n",
                "morning train\\nwindows full of rain\\nstation sighs\\n",
                3,
            ),
        ]

        for tool_call_id, path, new_text, old_text, line in file_payloads:
            await self._emit_tool(
                session_id=session_id,
                tool_call_id=tool_call_id,
                title=f"edit {path}",
                kind="edit",
                raw_input={"path": path},
                raw_output={"ok": True, "path": path},
                content=[
                    FileEditToolCallContent(
                        type="diff",
                        path=path,
                        oldText=old_text,
                        newText=new_text,
                    )
                ],
                locations=[ToolCallLocation(path=path, line=line)],
            )

        for path in ["/haikus/haiku1.txt", "/haikus/haiku2.txt", "/haikus/haiku3.txt", "/haikus"]:
            await self._emit_tool(
                session_id=session_id,
                tool_call_id=f"delete-{path.rsplit('/', 1)[-1] or 'haikus'}",
                title=f"delete {path}",
                kind="delete",
                raw_input={"path": path},
                raw_output={"deleted": path},
                locations=[ToolCallLocation(path=path)],
            )

        await self._conn.session_update(
            session_id=session_id,
            update=update_agent_message(text_block("done: wrote, edited, and deleted the haikus")),
            source="haiku_agent",
        )
        return PromptResponse(stopReason="end_turn")


asyncio.run(run_agent(HaikuAgent()))
""".strip(),
        encoding="utf-8",
    )

    driver = ACPDriver(
        cwd=tmp_path,
        command=(sys.executable, str(agent_script)),
        provider="test",
        model="haiku-agent",
    )

    prompt = (
        "write 3 haikus to a /haikus/ folder, edit the first/second/third line of each one, "
        "then delete all the files, then delete the folder"
    )
    events = [event async for event in driver.run(prompt)]

    tool_starts = [event for event in events if type(event).__name__ == "ToolCallStart"]
    tool_ends = [event for event in events if type(event).__name__ == "ToolCallEnd"]
    tool_results = [event for event in events if type(event).__name__ == "ToolResultReceived"]
    text_deltas = [event.delta for event in events if type(event).__name__ == "TextDelta"]

    assert text_deltas == [
        f"working:{prompt}",
        "done: wrote, edited, and deleted the haikus",
    ]
    assert [event.tool_name for event in tool_starts] == [
        "mkdir /haikus",
        "edit /haikus/haiku1.txt",
        "edit /haikus/haiku2.txt",
        "edit /haikus/haiku3.txt",
        "edit /haikus/haiku1.txt",
        "edit /haikus/haiku2.txt",
        "edit /haikus/haiku3.txt",
        "delete /haikus/haiku1.txt",
        "delete /haikus/haiku2.txt",
        "delete /haikus/haiku3.txt",
        "delete /haikus",
    ]
    assert [event.tool_call.name for event in tool_ends] == [
        event.tool_name for event in tool_starts
    ]
    assert len(tool_results) == 11
    assert tool_results[1].details is not None
    assert tool_results[1].details["kind"] == "edit"
    assert tool_results[1].details["locations"] == [{"path": "/haikus/haiku1.txt", "line": 1}]
    assert tool_results[1].details["content"][0]["path"] == "/haikus/haiku1.txt"
    assert tool_results[-1].details["kind"] == "delete"
    assert tool_results[-1].details["raw_output"] == {"deleted": "/haikus"}
