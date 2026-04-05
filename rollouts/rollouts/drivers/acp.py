from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..dtypes import (
    LLMCallEnd,
    LLMCallStart,
    StreamDone,
    StreamError,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCall,
    ToolCallEnd,
    ToolCallStart,
    ToolResultReceived,
)
from .protocol import ExternalAgentDriver

logger = logging.getLogger(__name__)
_ACP_TEARDOWN_TIMEOUT_SECONDS = 5.0


@dataclass
class _PendingToolCall:
    content_index: int
    name: str
    args: dict[str, Any]
    details: dict[str, Any]
    finalized: bool = False


class _ACPEventBridge:
    def __init__(
        self,
        *,
        auto_approve_permissions: bool,
    ) -> None:
        self._auto_approve_permissions = auto_approve_permissions
        self._content_index = 0
        self._pending_tools: dict[str, _PendingToolCall] = {}
        self._active_text: tuple[int, str] | None = None
        self._active_thinking: tuple[int, str] | None = None
        self.events: list[Any] = []

    def on_connect(self, conn: Any) -> None:
        del conn

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        del session_id, kwargs
        update_type = getattr(update, "sessionUpdate", None)
        if update_type == "agent_message_chunk":
            await self._emit_text_chunk(update)
            return
        if update_type == "agent_thought_chunk":
            await self._emit_thought_chunk(update)
            return
        if update_type in {"tool_call", "tool_call_update"}:
            await self._emit_tool_update(update)
            return

    async def request_permission(
        self, options: list[Any], session_id: str, tool_call: Any, **kwargs: Any
    ) -> Any:
        del session_id, tool_call, kwargs
        from acp import RequestPermissionResponse
        from acp.schema import AllowedOutcome, DeniedOutcome

        if not self._auto_approve_permissions:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

        allowed = next(
            (
                option
                for option in options
                if getattr(option, "kind", None) in {"allow_once", "allow_always"}
            ),
            None,
        )
        if allowed is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(
            outcome=AllowedOutcome(optionId=allowed.optionId, outcome="selected")
        )

    def _next_content_index(self) -> int:
        idx = self._content_index
        self._content_index += 1
        return idx

    async def _emit_text_chunk(self, update: Any) -> None:
        self._flush_thinking()
        text = _content_block_to_text(getattr(update, "content", None))
        if text == "":
            return
        if self._active_text is None:
            idx = self._next_content_index()
            self._active_text = (idx, "")
            self.events.append(TextStart(content_index=idx))
        idx, accumulated = self._active_text
        accumulated += text
        self._active_text = (idx, accumulated)
        self.events.append(TextDelta(content_index=idx, delta=text))

    async def _emit_thought_chunk(self, update: Any) -> None:
        self._flush_text()
        text = _content_block_to_text(getattr(update, "content", None))
        if text == "":
            return
        if self._active_thinking is None:
            idx = self._next_content_index()
            self._active_thinking = (idx, "")
            self.events.append(ThinkingStart(content_index=idx))
        idx, accumulated = self._active_thinking
        accumulated += text
        self._active_thinking = (idx, accumulated)
        self.events.append(ThinkingDelta(content_index=idx, delta=text))

    async def _emit_tool_update(self, update: Any) -> None:
        self._flush_text()
        self._flush_thinking()
        tool_call_id = getattr(update, "toolCallId", None)
        if not tool_call_id:
            return

        pending = self._pending_tools.get(tool_call_id)
        name = _tool_name(update)
        args = _json_safe(getattr(update, "rawInput", None))
        if not isinstance(args, dict):
            args = {"raw_input": args}
        details = {
            "kind": getattr(update, "kind", None),
            "title": getattr(update, "title", None),
            "locations": _json_safe(getattr(update, "locations", None)),
            "raw_input": _json_safe(getattr(update, "rawInput", None)),
            "raw_output": _json_safe(getattr(update, "rawOutput", None)),
            "content": _json_safe(getattr(update, "content", None)),
        }

        if pending is None:
            pending = _PendingToolCall(
                content_index=self._next_content_index(),
                name=name,
                args=args,
                details=details,
            )
            self._pending_tools[tool_call_id] = pending
            self.events.append(
                ToolCallStart(
                    content_index=pending.content_index,
                    tool_call_id=tool_call_id,
                    tool_name=pending.name,
                )
            )
        else:
            pending.name = name or pending.name
            pending.args = args or pending.args
            pending.details.update({k: v for k, v in details.items() if v is not None})

        status = getattr(update, "status", None)
        if status not in {"completed", "failed"} or pending.finalized:
            return

        pending.finalized = True
        tool_call = ToolCall(id=tool_call_id, name=pending.name, args=pending.args)
        self.events.append(ToolCallEnd(content_index=pending.content_index, tool_call=tool_call))
        raw_output = pending.details.get("raw_output")
        result_text = _render_tool_output(raw_output, pending.details.get("content"))
        error = result_text if status == "failed" else None
        self.events.append(
            ToolResultReceived(
                tool_call_id=tool_call_id,
                content=result_text,
                is_error=status == "failed",
                error=error,
                details=pending.details,
            )
        )

    def finalize(self) -> None:
        self._flush_text()
        self._flush_thinking()

    def _flush_text(self) -> None:
        if self._active_text is None:
            return
        idx, content = self._active_text
        self.events.append(TextEnd(content_index=idx, content=content))
        self._active_text = None

    def _flush_thinking(self) -> None:
        if self._active_thinking is None:
            return
        idx, content = self._active_thinking
        self.events.append(ThinkingEnd(content_index=idx, content=content))
        self._active_thinking = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "field_meta" and item is None:
                continue
            normalized[str(key)] = _json_safe(item)
        return normalized
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _json_safe(vars(value))
        except TypeError:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _content_block_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return "\n".join(filter(None, (_content_block_to_text(item) for item in content)))
    block_type = getattr(content, "type", None)
    if block_type == "text":
        return str(getattr(content, "text", ""))
    return json.dumps(_json_safe(content), indent=2)


def _tool_name(update: Any) -> str:
    title = getattr(update, "title", None)
    if title:
        return str(title)
    kind = getattr(update, "kind", None)
    if kind:
        return str(kind)
    return "tool"


def _render_tool_output(raw_output: Any, content: Any) -> str:
    if raw_output is not None:
        return _stringify(raw_output)
    if content is not None:
        return _stringify(content)
    return ""


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2)
    except TypeError:
        return repr(value)


@dataclass
class ACPDriver(ExternalAgentDriver):
    cwd: Path
    command: Sequence[str]
    provider: str
    model: str
    timeout_seconds: float = 600.0
    auto_approve_permissions: bool = True
    env: Mapping[str, str] | None = None
    mcp_servers: list[Any] | None = None
    client_name: str = "rollouts"
    client_version: str = "0.2.0"
    on_raw_line: Callable[[str], Awaitable[None]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd)
        if not self.command:
            raise ValueError("ACPDriver requires a non-empty command")

    async def run(self, prompt: str) -> AsyncIterator[Any]:
        for event in await self._collect_events(prompt):
            yield event

    async def send_input(self, text: str) -> None:
        raise NotImplementedError("ACPDriver currently supports single-turn runs only")

    async def abort(self) -> None:
        return None

    async def _collect_events(self, prompt: str) -> list[Any]:
        import trio_asyncio

        try:
            async with trio_asyncio.open_loop():
                bridge, response = await trio_asyncio.aio_as_trio(self._run_once_asyncio)(
                    prompt,
                )
            events: list[Any] = [
                StreamStart(),
                LLMCallStart(),
                *bridge.events,
                LLMCallEnd(
                    duration_ms=0.0,
                    provider=self.provider,
                    model=self.model,
                    tokens_in=getattr(getattr(response, "usage", None), "inputTokens", None),
                    tokens_out=getattr(getattr(response, "usage", None), "outputTokens", None),
                    status="success",
                ),
                StreamDone(finish_reason=response.stopReason),
            ]
            return events
        except Exception as e:
            logger.exception("ACP driver failed: %s", e)
            return [StreamError(error=str(e))]

    async def _run_once_asyncio(
        self,
        prompt: str,
    ) -> tuple[_ACPEventBridge, Any]:
        from acp import InitializeResponse, PromptResponse, spawn_agent_process
        from acp.schema import ClientCapabilities, Implementation, TextContentBlock

        bridge = _ACPEventBridge(auto_approve_permissions=self.auto_approve_permissions)
        command, *args = self.command
        env = {**os.environ, **dict(self.env or {})}
        async with spawn_agent_process(
            bridge,
            command,
            *args,
            env=env,
            cwd=self.cwd,
        ) as (conn, process):
            init = await conn.initialize(
                protocol_version=1,
                client_capabilities=ClientCapabilities(terminal=False),
                client_info=Implementation(name=self.client_name, version=self.client_version),
            )
            if not isinstance(init, InitializeResponse):
                raise RuntimeError(f"Unexpected ACP initialize response: {init!r}")
            session = await conn.new_session(cwd=str(self.cwd), mcp_servers=self.mcp_servers or [])
            response = await conn.prompt(
                [TextContentBlock(type="text", text=prompt)],
                session_id=session.sessionId,
            )
            if not isinstance(response, PromptResponse):
                raise RuntimeError(f"Unexpected ACP prompt response: {response!r}")
            bridge.finalize()
            cleanup_terminated = False
            if process.returncode is None:
                cleanup_terminated = True
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=_ACP_TEARDOWN_TIMEOUT_SECONDS)
                except TimeoutError:
                    logger.warning(
                        "ACP agent did not exit %.1fs after prompt completion; killing lingering process",
                        _ACP_TEARDOWN_TIMEOUT_SECONDS,
                    )
                    process.kill()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=1.0)
                    except TimeoutError:
                        logger.warning(
                            "ACP agent still did not exit after kill; continuing because prompt already completed"
                        )
            if process.returncode not in (None, 0, -15, -9) and not cleanup_terminated:
                raise RuntimeError(f"ACP agent exited with status {process.returncode}")
            return bridge, response
