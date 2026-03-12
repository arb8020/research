"""OrchestrateEnvironment - multi-agent orchestration via Python DSL.

The primary agent gets a single `orchestrate` tool. It writes async Python code
that calls a `system` API to spawn subthreads, run queries, share documents, and
log progress. The code runs in a real async context (trio), so `await` and
`trio.open_nursery()` work natively.

## System API

```python
# Spawn a subagent and await its result.
result = await system.thread(
    id,           # str | None — alias for session resumption
    task,         # str — task prompt sent to the subagent
    capabilities=None,  # list[str] | None — "read","write","grep","terminal","websearch"
    kind=None,    # str | None — native rollouts env kind by default; special: "claude_code", "codex"
    *,
    docs=None,    # list[str | dict] | None — document names to inject
    traces=None,  # list[ThreadResult | QueryResult] | None — prior results as context
)
# → ThreadResult(status, output, reason, trace, duration_ms)

# Single-shot query with no tools.
result = await system.query(prompt, *, docs=None, traces=None)
# → QueryResult(output, trace, duration_ms, error)

# Retrieve a prior thread/query result by alias (synchronous).
prior = system.fromId(alias)
# → ThreadResult | QueryResult | None

# Allocate a named document in the shared DocumentStore (synchronous).
system.allocate(name)
# → name (str)

# Append to the execution log (synchronous).
system.log(message, data=None)
```

## Parallelism

Use trio nurseries:

```python
async with trio.open_nursery() as n:
    results = {}
    async def run(key, task):
        results[key] = await system.thread(key, task)
    n.start_soon(run, "a", "Analyse X")
    n.start_soon(run, "b", "Analyse Y")
```

Or the simpler pattern using a list:

```python
async def gather(*coros):
    results = [None] * len(coros)
    async with trio.open_nursery() as n:
        for i, coro in enumerate(coros):
            async def _run(i=i, coro=coro):
                results[i] = await coro
            n.start_soon(_run)
    return results

a, b = await gather(
    system.thread("a", "Analyse X"),
    system.thread("b", "Analyse Y"),
)
```

## Stop tools for subthreads

Subthreads signal completion by calling one of these tools in their final message.
The system prompt injected into each subthread explains them:

- `complete(result="...")` — task succeeded, result is the output string
- `abort(reason="...")` — task failed, reason explains why
- `escalate(problem="...")` — blocked, needs planner intervention

If a subthread hits `max_turns` without calling any of these, it is treated as
`status="aborted"` with `reason="Thread timed out or otherwise failed."`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

if TYPE_CHECKING:
    from ..frontends.tui.theme import Theme

from ..agents import (
    Actor,
    AgentState,
    RunConfig,
    compose_handlers,
    handle_stop_max_turns,
    handle_stop_on_empty_message,
    inject_turn_warning,
    run_agent,
)
from ..core import (
    Endpoint,
    Message,
    StopReason,
    Tool,
    ToolCall,
    ToolFunction,
    ToolFunctionParameter,
    ToolResult,
    Trajectory,
)
from ..dtypes import StreamEvent
from ._formatting import get_text_output

# ── Default system prompt injected into every subthread ───────────────────────

SUBTHREAD_SYSTEM_PROMPT = """\
You are a subagent working on a specific task. You have access to tools to \
complete the task.

When you are done, you MUST call one of these completion tools as your final action:

- `complete(result="...")` — task succeeded. `result` should be a concise summary \
of what you accomplished and any key findings.
- `abort(reason="...")` — task failed or is impossible. `reason` should explain \
what went wrong and what was tried.
- `escalate(problem="...")` — you are blocked and need the planner to intervene. \
`problem` should describe exactly what you need help with.

Do not stop without calling one of these tools. If you are unsure, prefer `escalate` \
over `abort`.
"""

# Maximum turns a subthread runs before being killed and marked as timed out.
SUBTHREAD_MAX_TURNS = 30
THREAD_HANDOFF_MESSAGE_THRESHOLD = 24
THREAD_HANDOFF_RECENT_MESSAGES = 10
THREAD_HANDOFF_PREFIX = "# Prior thread handoff"

# ── Result types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThreadResult:
    status: str  # "completed" | "aborted" | "escalated" | "interrupted" | "error"
    output: str | None  # set when status == "completed"
    reason: str | None  # set when status != "completed"
    trace: str  # minimized transcript
    duration_ms: int
    session_id: str | None = None


@dataclass(frozen=True)
class QueryResult:
    output: str | None
    trace: str
    duration_ms: int
    error: str | None = None


@dataclass(frozen=True)
class ChildRunOutput:
    result: ThreadResult
    messages: tuple[Message, ...] = ()


@dataclass
class ExternalEventLimitState:
    max_events: int
    observed_events: int = 0
    exceeded: bool = False


# ── DocumentStore ─────────────────────────────────────────────────────────────


@dataclass
class DocumentStore:
    """Shared in-memory document store for an orchestration session.

    All subthreads share the same store. Documents are injected into subthread
    prompts as explicit instructions (input/output/reference/modify).
    """

    _docs: dict[str, str] = field(default_factory=dict)

    def allocate(self, name: str) -> str:
        """Reserve a document slot. Creates empty doc if not present."""
        if name not in self._docs:
            self._docs[name] = ""
        return name

    def read(self, name: str) -> str | None:
        return self._docs.get(name)

    def write(self, name: str, content: str) -> None:
        self._docs[name] = content

    def names(self) -> list[str]:
        return list(self._docs.keys())


@dataclass
class SharedDocumentEnvironment:
    """Tool environment for shared orchestration documents.

    This is only used inside native orchestrated child threads. It keeps a
    thread-local view of a subset of shared docs, then the orchestrator syncs
    the final state back into the session-wide DocumentStore when the child
    finishes.
    """

    documents: dict[str, str] = field(default_factory=dict)
    permissions: dict[str, str] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)

    def get_name(self) -> str:
        return "orchestrate_documents"

    def get_tools(self) -> list[Tool]:
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="list_documents",
                    description="List shared orchestration documents available to this child thread.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={},
                    ),
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="read_document",
                    description="Read a shared orchestration document by name.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "name": {
                                "type": "string",
                                "description": "Document name returned by list_documents().",
                            }
                        },
                    ),
                    required=["name"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="write_document",
                    description=(
                        "Write or append content to a writable shared orchestration document. "
                        "Only output/modify documents may be written."
                    ),
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "name": {
                                "type": "string",
                                "description": "Document name returned by list_documents().",
                            },
                            "content": {
                                "type": "string",
                                "description": "Text to write to the document.",
                            },
                            "append": {
                                "type": "boolean",
                                "description": "Append to the existing document instead of replacing it.",
                            },
                        },
                    ),
                    required=["name", "content"],
                ),
            ),
        ]

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        return False

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        del current_state, run_config, cancel_scope

        if tool_call.name == "list_documents":
            lines = []
            for name in sorted(self.documents):
                permission = self.permissions.get(name, "reference")
                description = self.descriptions.get(name, "")
                suffix = f" - {description}" if description else ""
                lines.append(f"{name} [{permission}]{suffix}")
            content = "\n".join(lines) if lines else "(no documents)"
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=False,
                content=content,
            )

        if tool_call.name == "read_document":
            name = str(tool_call.args.get("name", ""))
            if name not in self.documents:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Unknown document: {name}",
                )
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=False,
                content=self.documents[name],
                details={
                    "document_name": name,
                    "operation": "read_document",
                    "permission": self.permissions.get(name, "reference"),
                },
            )

        if tool_call.name == "write_document":
            name = str(tool_call.args.get("name", ""))
            content = str(tool_call.args.get("content", ""))
            append = bool(tool_call.args.get("append", False))
            permission = self.permissions.get(name)

            if name not in self.documents:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Unknown document: {name}",
                )
            if permission not in {"output", "modify"}:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Document is read-only: {name}",
                )

            prior = self.documents.get(name, "")
            self.documents[name] = prior + content if append else content
            verb = "Appended to" if append else "Wrote"
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=False,
                content=f"{verb} document {name}",
                details={
                    "document_name": name,
                    "operation": "append_document" if append else "write_document",
                    "chars": len(content),
                },
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=True,
            content="",
            error=f"Unknown tool: {tool_call.name}",
        )

    async def serialize(self) -> dict[str, Any]:
        return {
            "env_kind": "orchestrate_documents",
            "version": "1.0.0",
            "documents": self.documents,
            "permissions": self.permissions,
            "descriptions": self.descriptions,
        }

    @staticmethod
    async def deserialize(data: dict[str, Any]) -> SharedDocumentEnvironment:
        assert data.get("env_kind") == "orchestrate_documents"
        return SharedDocumentEnvironment(
            documents=dict(data.get("documents", {})),
            permissions=dict(data.get("permissions", {})),
            descriptions=dict(data.get("descriptions", {})),
        )


# ── System runtime ────────────────────────────────────────────────────────────


class System:
    """The `system` object passed to orchestration code.

    All methods that spawn agents are async. Log and allocate are sync.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        cwd: Path,
        document_store: DocumentStore,
        subthread_system_prompt: str,
        max_turns: int,
        on_event: Callable[[dict[str, Any]], None],
        saved_results: dict[str, dict[str, Any]] | None = None,
        saved_native_messages: dict[str, list[str]] | None = None,
        saved_child_kinds: dict[str, str] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._cwd = cwd
        self._docs = document_store
        self._subthread_system_prompt = subthread_system_prompt
        self._max_turns = max_turns
        self._on_event = on_event

        # alias → ThreadResult | QueryResult
        self._results: dict[str, ThreadResult | QueryResult] = {
            alias: _deserialize_saved_result(result)
            for alias, result in (saved_results or {}).items()
        }
        self._saved_results = saved_results if saved_results is not None else {}
        self._saved_native_messages = (
            saved_native_messages if saved_native_messages is not None else {}
        )
        self._saved_child_kinds = saved_child_kinds if saved_child_kinds is not None else {}
        # execution sequence log
        self._sequence: list[dict[str, Any]] = []

    def _append(self, item: dict[str, Any]) -> None:
        self._sequence.append(item)
        self._on_event(item)

    # ── Public API ─────────────────────────────────────────────────────────

    async def thread(
        self,
        id: str | None,  # noqa: A002
        task: str,
        capabilities: list[str] | None = None,
        kind: str | None = None,
        *,
        docs: list[str | dict[str, Any]] | None = None,
        traces: list[ThreadResult | QueryResult] | None = None,
    ) -> ThreadResult:
        """Spawn a subthread and await its result."""
        t0 = time.perf_counter()
        alias = id
        child_kind = kind or self._saved_child_kinds.get(alias or "", "coding")

        # Build system message content
        system_parts = [self._subthread_system_prompt]

        # Inject traces as prior context
        if traces:
            system_parts.append("# Prior context\n")
            for trace in traces:
                system_parts.append(_render_result(trace))

        # Inject doc instructions
        if docs:
            system_parts.append(_render_doc_instructions(docs))

        # Build task prompt
        full_task = task
        if docs:
            # Remind about doc access
            doc_names = [d if isinstance(d, str) else d.get("name", "") for d in docs]
            full_task += f"\n\nDocuments available: {', '.join(doc_names)}"

        system_content = "\n\n".join(p for p in system_parts if p)

        # Construct AgentState for run_claude autonomous mode
        self._append({"type": "thread_start", "id": alias, "task": task[:200], "kind": child_kind})

        # Null sink for events (subthread events not forwarded to primary)
        async def _noop(event: Any) -> None:
            pass

        run_cfg = RunConfig(on_chunk=_noop)
        existing_messages: list[Message] = []
        if alias and child_kind not in {"claude_code", "codex"}:
            prior_kind = self._saved_child_kinds.get(alias)
            if prior_kind in {None, child_kind}:
                existing_messages = _deserialize_messages(self._saved_native_messages.get(alias))

        child_output: ChildRunOutput | None = None
        try:
            child_output = await _run_child_thread(
                kind=child_kind,
                endpoint=self._endpoint,
                cwd=self._cwd,
                task=full_task,
                system_content=system_content,
                capabilities=capabilities,
                run_cfg=run_cfg,
                document_store=self._docs,
                docs=docs,
                max_turns=self._max_turns,
                existing_messages=existing_messages,
            )
            thread_result = child_output.result
        except Exception as e:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            thread_result = ThreadResult(
                status="error",
                output=None,
                reason=str(e),
                trace="",
                duration_ms=duration_ms,
            )

        if alias:
            self._results[alias] = thread_result
            self._saved_results[alias] = _serialize_result(thread_result)
            self._saved_child_kinds[alias] = child_kind
            if child_kind not in {"claude_code", "codex"} and child_output is not None:
                self._saved_native_messages[alias] = _serialize_messages(child_output.messages)

        self._append({
            "type": "thread_end",
            "id": alias,
            "status": thread_result.status,
            "duration_ms": thread_result.duration_ms,
        })

        return thread_result

    async def query(
        self,
        prompt: str,
        *,
        docs: list[str | dict[str, Any]] | None = None,
        traces: list[ThreadResult | QueryResult] | None = None,
        id: str | None = None,  # noqa: A002
    ) -> QueryResult:
        """Single-shot query with no tools."""
        import shutil
        import subprocess

        import trio.lowlevel

        t0 = time.perf_counter()

        full_prompt = prompt
        if traces:
            ctx = "\n\n".join(_render_result(t) for t in traces)
            full_prompt = f"# Prior context\n\n{ctx}\n\n# Task\n\n{prompt}"
        if docs:
            full_prompt += "\n\n" + _render_doc_instructions(docs)

        self._append({"type": "query_start", "prompt": prompt[:200]})

        claude_bin = shutil.which("claude")
        if claude_bin is None:
            return QueryResult(
                output=None,
                trace="",
                duration_ms=0,
                error="claude CLI not found",
            )

        cmd = [
            claude_bin,
            "--print",
            "--output-format",
            "text",
            "--dangerously-skip-permissions",
            full_prompt,
        ]

        try:
            result = await trio.to_thread.run_sync(
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self._cwd),
                    timeout=120,
                )
            )
            output = result.stdout.strip() if result.returncode == 0 else None
            error = result.stderr.strip() if result.returncode != 0 else None
            duration_ms = int((time.perf_counter() - t0) * 1000)
            query_result = QueryResult(
                output=output,
                trace=output or "",
                duration_ms=duration_ms,
                error=error,
            )
        except Exception as e:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            query_result = QueryResult(output=None, trace="", duration_ms=duration_ms, error=str(e))

        if id:
            self._results[id] = query_result
            self._saved_results[id] = _serialize_result(query_result)

        self._append({
            "type": "query_end",
            "status": "ok" if query_result.error is None else "error",
            "duration_ms": query_result.duration_ms,
        })

        return query_result

    def fromId(self, alias: str) -> ThreadResult | QueryResult | None:  # noqa: N802
        """Retrieve a prior result by alias."""
        return self._results.get(alias)

    def allocate(self, name: str) -> str:
        """Reserve a document slot in the shared DocumentStore."""
        result = self._docs.allocate(name)
        self._append({"type": "allocate", "name": name})
        return result

    def log(self, message: str, data: Any = None) -> None:
        """Append to the execution log."""
        self._append({
            "type": "log",
            "message": message,
            **({"data": data} if data is not None else {}),
        })


# ── Helper functions ───────────────────────────────────────────────────────────


def _capabilities_to_allowed_tools(capabilities: list[str]) -> list[str]:
    mapping: dict[str, list[str]] = {
        "read": ["Read", "LS"],
        "write": ["Write", "Edit", "MultiEdit"],
        "grep": ["Grep", "Glob"],
        "terminal": ["Bash"],
        "websearch": ["WebSearch", "WebFetch"],
    }
    tools: list[str] = []
    for cap in capabilities:
        tools.extend(mapping.get(cap, []))
    return tools


def _capabilities_to_native_tools(capabilities: list[str] | None) -> list[str]:
    caps = capabilities or ["read", "write", "grep", "terminal"]
    tools: list[str] = []

    if "read" in caps and "read" not in tools:
        tools.append("read")
    if "write" in caps:
        if "write" not in tools:
            tools.append("write")
        if "edit" not in tools:
            tools.append("edit")
    if "grep" in caps or "terminal" in caps:
        if "bash" not in tools:
            tools.append("bash")
    if "websearch" in caps and "web_fetch" not in tools:
        tools.append("web_fetch")

    return tools or ["read"]


def _build_external_prompt(system_content: str, task: str) -> str:
    parts = []
    if system_content.strip():
        parts.append("# Instructions")
        parts.append(system_content.strip())
    parts.append("# Task")
    parts.append(task.strip())
    return "\n\n".join(parts)


def _normalize_docs(
    docs: list[str | dict[str, Any]] | None,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in docs or []:
        if isinstance(item, str):
            normalized.append({
                "name": item,
                "purpose": "reference",
                "description": "",
            })
            continue

        name = str(item.get("name", "")).strip()
        if not name:
            continue
        normalized.append({
            "name": name,
            "purpose": str(item.get("purpose", "reference")),
            "description": str(item.get("description", "")),
        })
    return normalized


def _build_document_environment(
    document_store: DocumentStore,
    docs: list[str | dict[str, Any]] | None,
) -> SharedDocumentEnvironment | None:
    normalized = _normalize_docs(docs)
    if not normalized:
        return None

    documents: dict[str, str] = {}
    permissions: dict[str, str] = {}
    descriptions: dict[str, str] = {}

    for spec in normalized:
        name = spec["name"]
        purpose = spec["purpose"]
        document_store.allocate(name)
        documents[name] = document_store.read(name) or ""
        permissions[name] = purpose
        descriptions[name] = spec["description"]

    return SharedDocumentEnvironment(
        documents=documents,
        permissions=permissions,
        descriptions=descriptions,
    )


def _sync_documents_back(
    environment: Any,
    document_store: DocumentStore,
) -> None:
    from .compose import ComposedEnvironment

    if isinstance(environment, SharedDocumentEnvironment):
        doc_env = environment
    elif isinstance(environment, ComposedEnvironment):
        doc_env = next(
            (env for env in environment.environments if isinstance(env, SharedDocumentEnvironment)),
            None,
        )
    else:
        doc_env = None

    if doc_env is None:
        return

    for name, content in doc_env.documents.items():
        if doc_env.permissions.get(name) in {"output", "modify"}:
            document_store.write(name, content)


def _make_child_run_config(base_run_cfg: RunConfig, max_turns: int) -> RunConfig:
    warning_handler = (
        inject_turn_warning(max_turns=max_turns, warning_at=min(2, max_turns - 1))
        if max_turns > 1
        else (lambda state: state)
    )
    stop_handler = compose_handlers([
        handle_stop_max_turns(max_turns),
        handle_stop_on_empty_message(),
    ])
    return RunConfig(
        on_chunk=base_run_cfg.on_chunk,
        on_input=base_run_cfg.on_input,
        confirm_tool=base_run_cfg.confirm_tool,
        handle_tool_error=base_run_cfg.handle_tool_error,
        on_step_start=warning_handler,
        handle_stop=stop_handler,
        handle_no_tool=base_run_cfg.handle_no_tool,
        user_message_for_thinking=base_run_cfg.user_message_for_thinking,
        inline_thinking=base_run_cfg.inline_thinking,
        show_progress=base_run_cfg.show_progress,
        cancel_scope=base_run_cfg.cancel_scope,
        interrupt_flag=base_run_cfg.interrupt_flag,
        session_store=base_run_cfg.session_store,
        api_limiter=base_run_cfg.api_limiter,
        tool_limiter=base_run_cfg.tool_limiter,
    )


def _is_external_counted_event(event: StreamEvent) -> bool:
    event_type = getattr(event, "type", "")
    return event_type in {"text_end", "toolcall_end", "done"}


def _make_external_run_config(
    base_run_cfg: RunConfig,
    max_events: int,
    *,
    mode: str,
    interrupt_flag: list[bool] | None = None,
    cancel_scope: trio.CancelScope | None = None,
) -> tuple[RunConfig, ExternalEventLimitState]:
    state = ExternalEventLimitState(max_events=max_events)

    async def wrapped_on_chunk(event: StreamEvent) -> None:
        await base_run_cfg.on_chunk(event)

        if state.exceeded:
            return

        if _is_external_counted_event(event):
            state.observed_events += 1
            if state.observed_events >= max_events:
                state.exceeded = True
                if mode == "claude_code" and interrupt_flag is not None:
                    interrupt_flag[0] = True
                if mode == "codex" and cancel_scope is not None:
                    cancel_scope.cancel()

    cfg = RunConfig(
        on_chunk=wrapped_on_chunk,
        on_input=base_run_cfg.on_input,
        confirm_tool=base_run_cfg.confirm_tool,
        handle_tool_error=base_run_cfg.handle_tool_error,
        on_step_start=base_run_cfg.on_step_start,
        handle_stop=base_run_cfg.handle_stop,
        handle_no_tool=base_run_cfg.handle_no_tool,
        user_message_for_thinking=base_run_cfg.user_message_for_thinking,
        inline_thinking=base_run_cfg.inline_thinking,
        show_progress=base_run_cfg.show_progress,
        cancel_scope=cancel_scope or base_run_cfg.cancel_scope,
        interrupt_flag=interrupt_flag or base_run_cfg.interrupt_flag,
        session_store=base_run_cfg.session_store,
        api_limiter=base_run_cfg.api_limiter,
        tool_limiter=base_run_cfg.tool_limiter,
    )
    return cfg, state


def _get_last_assistant_text(messages: list[Message]) -> str | None:
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if hasattr(part, "type") and part.type == "text":
                    text = getattr(part, "text", "")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text.strip())
            if text_parts:
                return "\n".join(text_parts)
    return None


def _serialize_messages(messages: list[Message] | tuple[Message, ...]) -> list[str]:
    return [message.to_json() for message in messages]


def _deserialize_messages(payloads: list[str] | None) -> list[Message]:
    if not payloads:
        return []
    return [Message.from_json(payload) for payload in payloads]


def _serialize_result(result: ThreadResult | QueryResult) -> dict[str, Any]:
    if isinstance(result, ThreadResult):
        return {
            "result_type": "thread",
            "status": result.status,
            "output": result.output,
            "reason": result.reason,
            "trace": result.trace,
            "duration_ms": result.duration_ms,
            "session_id": result.session_id,
        }

    return {
        "result_type": "query",
        "output": result.output,
        "trace": result.trace,
        "duration_ms": result.duration_ms,
        "error": result.error,
    }


def _deserialize_saved_result(data: dict[str, Any]) -> ThreadResult | QueryResult:
    result_type = data.get("result_type", "thread")
    if result_type == "query":
        return QueryResult(
            output=data.get("output"),
            trace=data.get("trace", ""),
            duration_ms=int(data.get("duration_ms", 0)),
            error=data.get("error"),
        )

    return ThreadResult(
        status=data.get("status", "error"),
        output=data.get("output"),
        reason=data.get("reason"),
        trace=data.get("trace", ""),
        duration_ms=int(data.get("duration_ms", 0)),
        session_id=data.get("session_id"),
    )


def _is_handoff_message(message: Message) -> bool:
    return (
        message.role == "system"
        and isinstance(message.content, str)
        and message.content.startswith(THREAD_HANDOFF_PREFIX)
    )


def _truncate_text(text: str, *, max_chars: int = 1200, max_lines: int = 16) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    clipped = lines[:max_lines]
    output = "\n".join(clipped)
    if len(lines) > max_lines:
        output += f"\n... ({len(lines) - max_lines} more lines)"
    if len(output) > max_chars:
        output = output[:max_chars].rstrip() + "..."
    return output.strip()


def _summarize_messages_for_handoff(
    prior_summary: str | None,
    target_messages: list[Message],
    recent_messages: list[Message],
) -> str:
    parts: list[str] = [THREAD_HANDOFF_PREFIX]

    if prior_summary:
        parts.append("## Earlier summary")
        parts.append(
            _truncate_text(
                prior_summary.removeprefix(THREAD_HANDOFF_PREFIX).strip(), max_chars=1500
            )
        )

    target_transcript = _generate_transcript(target_messages)
    if target_transcript:
        parts.append("## Completed work")
        parts.append(_truncate_text(target_transcript, max_chars=3000, max_lines=40))

    recent_transcript = _generate_transcript(recent_messages)
    if recent_transcript:
        parts.append("## Recent context")
        parts.append(_truncate_text(recent_transcript, max_chars=1200, max_lines=16))

    if len(parts) == 1:
        parts.append("No prior work was preserved.")

    return "\n\n".join(part for part in parts if part.strip())


def _compact_thread_messages(messages: list[Message]) -> list[Message]:
    if len(messages) <= THREAD_HANDOFF_MESSAGE_THRESHOLD:
        return list(messages)

    compacted = list(messages)
    system_prefix: list[Message] = []
    while compacted and compacted[0].role == "system" and not _is_handoff_message(compacted[0]):
        system_prefix.append(compacted.pop(0))

    prior_handoff: str | None = None
    if compacted and _is_handoff_message(compacted[0]):
        handoff_msg = compacted.pop(0)
        prior_handoff = handoff_msg.content if isinstance(handoff_msg.content, str) else None

    if len(compacted) <= THREAD_HANDOFF_MESSAGE_THRESHOLD:
        rebuilt = [*system_prefix]
        if prior_handoff:
            rebuilt.append(Message(role="system", content=prior_handoff))
        rebuilt.extend(compacted)
        return rebuilt

    recent_count = min(THREAD_HANDOFF_RECENT_MESSAGES, len(compacted))
    target_messages = compacted[:-recent_count]
    recent_messages = compacted[-recent_count:]
    handoff_summary = _summarize_messages_for_handoff(
        prior_handoff, target_messages, recent_messages
    )

    return [
        *system_prefix,
        Message(role="system", content=handoff_summary),
        *recent_messages,
    ]


def _thread_result_from_state(final_state: AgentState, duration_ms: int) -> ThreadResult:
    messages_out = list(final_state.actor.trajectory.messages)
    result = _extract_result(messages_out)
    trace = _generate_transcript(messages_out)
    session_id = final_state.driver_session_id or final_state.session_id

    if (
        result["status"] != "aborted"
        or result.get("reason") != "Thread timed out or otherwise failed."
    ):
        return ThreadResult(
            status=result["status"],
            output=result.get("output"),
            reason=result.get("reason"),
            trace=trace,
            duration_ms=duration_ms,
            session_id=session_id,
        )

    if final_state.stop == StopReason.MAX_TURNS:
        return ThreadResult(
            status="aborted",
            output=None,
            reason="Thread exceeded max_turns without calling complete/abort/escalate.",
            trace=trace,
            duration_ms=duration_ms,
            session_id=session_id,
        )

    final_text = _get_last_assistant_text(messages_out)
    if final_text:
        return ThreadResult(
            status="completed",
            output=final_text,
            reason=None,
            trace=trace,
            duration_ms=duration_ms,
            session_id=session_id,
        )

    if final_state.stop == StopReason.ABORTED:
        return ThreadResult(
            status="interrupted",
            output=None,
            reason="Interrupted",
            trace=trace or "[Interrupted by user]",
            duration_ms=duration_ms,
            session_id=session_id,
        )

    if final_state.stop == StopReason.PROVIDER_ERROR:
        return ThreadResult(
            status="error",
            output=None,
            reason=final_state.error or "Provider error",
            trace=trace or f"ERROR: {final_state.error or 'Provider error'}",
            duration_ms=duration_ms,
            session_id=session_id,
        )

    return ThreadResult(
        status="aborted",
        output=None,
        reason=result.get("reason"),
        trace=trace,
        duration_ms=duration_ms,
        session_id=session_id,
    )


async def _run_native_thread(
    *,
    kind: str,
    endpoint: Endpoint,
    cwd: Path,
    task: str,
    system_content: str,
    capabilities: list[str] | None,
    run_cfg: RunConfig,
    document_store: DocumentStore,
    docs: list[str | dict[str, Any]] | None,
    max_turns: int,
    existing_messages: list[Message] | None = None,
) -> ChildRunOutput:
    from .calculator import CalculatorEnvironment
    from .coding import LocalFilesystemEnvironment
    from .compose import ComposedEnvironment
    from .git_worktree import GitWorktreeEnvironment

    if kind == "coding":
        env = LocalFilesystemEnvironment(
            working_dir=cwd,
            tools=_capabilities_to_native_tools(capabilities),
        )
    elif kind == "search":
        env = LocalFilesystemEnvironment(
            working_dir=cwd,
            tools=["read", "bash"],
        )
    elif kind == "git":
        env = GitWorktreeEnvironment(working_dir=cwd)
    elif kind == "calculator":
        env = CalculatorEnvironment()
    else:
        raise ValueError(f"Unsupported native child kind: {kind}")

    doc_env = _build_document_environment(document_store, docs)
    if doc_env is not None:
        env = ComposedEnvironment([env, doc_env])

    messages = _compact_thread_messages(list(existing_messages or []))
    if messages:
        resume_task = task
        if system_content.strip():
            resume_task = (
                f"[Thread context refresh]\n{system_content.strip()}\n\n[New task]\n{task.strip()}"
            )
        messages.append(Message(role="user", content=resume_task))
    else:
        if system_content:
            messages.append(Message(role="system", content=system_content))
        messages.append(Message(role="user", content=task))

    initial_state = AgentState(
        actor=Actor(
            trajectory=Trajectory(messages=messages),
            endpoint=endpoint,
            tools=env.get_tools(),
        ),
        environment=env,
    )
    t0 = time.perf_counter()
    final_states = await run_agent(initial_state, _make_child_run_config(run_cfg, max_turns))
    final_state = final_states[-1] if final_states else initial_state
    if final_state.environment is not None:
        _sync_documents_back(final_state.environment, document_store)
    duration_ms = int((time.perf_counter() - t0) * 1000)
    final_messages = _compact_thread_messages(list(final_state.actor.trajectory.messages))
    result = _thread_result_from_state(final_state, duration_ms)
    return ChildRunOutput(result=result, messages=tuple(final_messages))


async def _run_claude_code_thread(
    *,
    endpoint: Endpoint,
    cwd: Path,
    task: str,
    system_content: str,
    capabilities: list[str] | None,
    run_cfg: RunConfig,
    max_turns: int,
) -> ChildRunOutput:
    from ..drivers.run_claude import run_claude

    prompt = _build_external_prompt(system_content, task)
    allowed_tools = _capabilities_to_allowed_tools(
        capabilities or ["read", "write", "grep", "terminal"]
    )
    initial_state = AgentState(
        actor=Actor(
            trajectory=Trajectory(messages=[Message(role="user", content=prompt)]),
            endpoint=endpoint,
            tools=[],
        ),
        environment=None,
    )
    model = "sonnet"
    if endpoint.model and endpoint.provider == "anthropic":
        raw = endpoint.model
        model = raw.split("/", 1)[1] if "/" in raw else raw
    interrupt_flag = [False]
    limited_run_cfg, limit_state = _make_external_run_config(
        run_cfg,
        max_turns,
        mode="claude_code",
        interrupt_flag=interrupt_flag,
    )
    t0 = time.perf_counter()
    final_states = await run_claude(
        initial_state,
        limited_run_cfg,
        model=model,
        cwd=cwd,
        autonomous=True,
        allowed_tools=allowed_tools,
    )
    final_state = final_states[-1] if final_states else initial_state
    duration_ms = int((time.perf_counter() - t0) * 1000)
    messages = tuple(final_state.actor.trajectory.messages)
    result = _thread_result_from_state(final_state, duration_ms)
    if limit_state.exceeded:
        result = ThreadResult(
            status="aborted",
            output=None,
            reason=f"Thread exceeded observed event limit ({max_turns}) in claude_code runtime.",
            trace=result.trace,
            duration_ms=result.duration_ms,
            session_id=result.session_id,
        )
    return ChildRunOutput(result=result, messages=messages)


async def _run_codex_thread(
    *,
    endpoint: Endpoint,
    cwd: Path,
    task: str,
    system_content: str,
    run_cfg: RunConfig,
    max_turns: int,
) -> ChildRunOutput:
    from ..drivers.run_codex import run_codex

    prompt = _build_external_prompt(system_content, task)
    initial_state = AgentState(
        actor=Actor(
            trajectory=Trajectory(messages=[Message(role="user", content=prompt)]),
            endpoint=endpoint,
            tools=[],
        ),
        environment=None,
    )
    t0 = time.perf_counter()
    with trio.CancelScope() as cancel_scope:
        limited_run_cfg, limit_state = _make_external_run_config(
            run_cfg,
            max_turns,
            mode="codex",
            cancel_scope=cancel_scope,
        )
        final_states = await run_codex(
            initial_state,
            limited_run_cfg,
            model="o3",
            cwd=cwd,
            autonomous=True,
        )
    final_state = final_states[-1] if final_states else initial_state
    duration_ms = int((time.perf_counter() - t0) * 1000)
    messages = tuple(final_state.actor.trajectory.messages)
    result = _thread_result_from_state(final_state, duration_ms)
    if limit_state.exceeded:
        result = ThreadResult(
            status="aborted",
            output=None,
            reason=f"Thread exceeded observed event limit ({max_turns}) in codex runtime.",
            trace=result.trace,
            duration_ms=result.duration_ms,
            session_id=result.session_id,
        )
    return ChildRunOutput(result=result, messages=messages)


async def _run_child_thread(
    *,
    kind: str,
    endpoint: Endpoint,
    cwd: Path,
    task: str,
    system_content: str,
    capabilities: list[str] | None,
    run_cfg: RunConfig,
    document_store: DocumentStore,
    docs: list[str | dict[str, Any]] | None,
    max_turns: int,
    existing_messages: list[Message] | None = None,
) -> ChildRunOutput:
    if kind == "claude_code":
        return await _run_claude_code_thread(
            endpoint=endpoint,
            cwd=cwd,
            task=task,
            system_content=system_content,
            capabilities=capabilities,
            run_cfg=run_cfg,
            max_turns=max_turns,
        )
    if kind == "codex":
        return await _run_codex_thread(
            endpoint=endpoint,
            cwd=cwd,
            task=task,
            system_content=system_content,
            run_cfg=run_cfg,
            max_turns=max_turns,
        )
    return await _run_native_thread(
        kind=kind,
        endpoint=endpoint,
        cwd=cwd,
        task=task,
        system_content=system_content,
        capabilities=capabilities,
        run_cfg=run_cfg,
        document_store=document_store,
        docs=docs,
        max_turns=max_turns,
        existing_messages=existing_messages,
    )


def _extract_result(messages: list[Message]) -> dict[str, Any]:
    """Scan messages backwards for complete/abort/escalate tool call."""
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        content = msg.content
        if not isinstance(content, list):
            continue
        for part in content:
            if not hasattr(part, "type"):
                continue
            if part.type == "toolCall":
                name = part.name
                args = part.arguments if isinstance(part.arguments, dict) else {}
                if name == "complete":
                    return {"status": "completed", "output": args.get("result", "")}
                if name == "abort":
                    return {"status": "aborted", "reason": args.get("reason", "")}
                if name == "escalate":
                    return {"status": "escalated", "reason": args.get("problem", "")}

    # Also check raw string content for tool calls (non-streaming claude output)
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        content = msg.content
        if not isinstance(content, str):
            continue
        # Look for JSON tool call patterns in text
        for name, key in [("complete", "result"), ("abort", "reason"), ("escalate", "problem")]:
            if f'"{name}"' in content or f"'{name}'" in content:
                status = (
                    "completed"
                    if name == "complete"
                    else ("escalated" if name == "escalate" else "aborted")
                )
                result_key = "output" if name == "complete" else "reason"
                return {"status": status, result_key: ""}

    return {"status": "aborted", "reason": "Thread timed out or otherwise failed."}


def _format_tool_result_for_trace(
    tool_name: str | None,
    message: Message,
) -> list[str]:
    details = message.details or {}
    content = message.content if isinstance(message.content, str) else ""
    lines: list[str] = []
    label = tool_name or "tool"

    if message.tool_call_id:
        lines.append(f"RESULT[{label}#{message.tool_call_id}]:")
    else:
        lines.append(f"RESULT[{label}]:")

    if details.get("diff"):
        lines.append(_truncate_text(str(details["diff"]), max_chars=1600, max_lines=20))
        return lines

    file_path = details.get("file_path")
    operation = details.get("operation")
    if file_path and operation:
        range_bits: list[str] = []
        if details.get("start_line") is not None:
            range_bits.append(str(details["start_line"]))
        if details.get("end_line") is not None:
            range_bits.append(str(details["end_line"]))
        location = f" lines {'-'.join(range_bits)}" if range_bits else ""
        lines.append(f"{operation}: {file_path}{location}")
        if content.strip():
            lines.append(_truncate_text(content, max_chars=600, max_lines=6))
        return lines

    output_file = details.get("output_file")
    if output_file:
        lines.append(f"output_file: {output_file}")

    if content.strip():
        lines.append(_truncate_text(content, max_chars=1400, max_lines=12))
    elif details:
        lines.append(_truncate_text(str(details), max_chars=600, max_lines=8))
    else:
        lines.append("(no output)")
    return lines


def _generate_transcript(messages: list[Message]) -> str:
    """Generate a minimized text transcript from a message list."""
    lines: list[str] = []
    tool_names: dict[str, str] = {}
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "user":
            continue
        if msg.role == "assistant":
            content = msg.content
            if isinstance(content, str) and content.strip():
                lines.append(f'AGENT: "{content.strip()}"')
            elif isinstance(content, list):
                for part in content:
                    if hasattr(part, "type"):
                        if part.type == "text" and getattr(part, "text", "").strip():
                            lines.append(f'AGENT: "{part.text.strip()}"')
                        elif part.type == "thinking" and getattr(part, "thinking", "").strip():
                            lines.append(f"AGENT: <thinking>{part.thinking.strip()}</thinking>")
                        elif part.type == "toolCall":
                            tool_names[part.id] = part.name
                            lines.append(
                                f"TOOL: {part.name}({_fmt_args(getattr(part, 'arguments', {}))})"
                            )
            lines.append("")
        elif msg.role == "tool":
            tool_name = tool_names.get(msg.tool_call_id or "")
            lines.extend(_format_tool_result_for_trace(tool_name, msg))
            lines.append("")
    return "\n".join(lines).strip()


def _fmt_args(args: dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 40:
            parts.append(f'{k}="{v[:40]}..."')
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _render_result(result: ThreadResult | QueryResult) -> str:
    lines: list[str] = []
    if isinstance(result, ThreadResult):
        lines.append(f"STATUS: {result.status}")
        if result.output:
            lines.append(f"OUTPUT: {result.output}")
        if result.reason:
            lines.append(f"REASON: {result.reason}")
        if result.trace:
            lines.append("TRACE:")
            for line in result.trace.splitlines():
                lines.append(f"  {line}")
    else:
        if result.error:
            lines.append(f"ERROR: {result.error}")
        elif result.output:
            lines.append(f"OUTPUT: {result.output}")
    return "\n".join(lines)


def _render_doc_instructions(docs: list[str | dict[str, Any]]) -> str:
    lines = [
        "## Documents",
        "",
        "Native child threads can access these with list_documents(), read_document(name=...), and write_document(name=..., content=...).",
        "",
    ]
    by_purpose: dict[str, list[str]] = {"input": [], "output": [], "modify": [], "reference": []}
    for d in docs:
        if isinstance(d, str):
            by_purpose["reference"].append(d)
        else:
            purpose = d.get("purpose", "reference")
            name = d.get("name", "")
            desc = d.get("description", "")
            label = f"{name}" + (f": {desc}" if desc else "")
            by_purpose.get(purpose, by_purpose["reference"]).append(label)

    labels = {
        "input": ("Input Documents (READ these)", "You MUST read these and use their content."),
        "output": ("Output Documents (WRITE to these)", "You MUST write your results here."),
        "modify": ("Documents to Modify", "Read and update these as needed."),
        "reference": ("Reference Documents", "Consult as needed."),
    }
    for purpose, (heading, instruction) in labels.items():
        items = by_purpose[purpose]
        if items:
            lines.append(f"### {heading}")
            for item in items:
                lines.append(f"- {item}")
            lines.append(instruction)
            lines.append("")
    return "\n".join(lines)


def format_orchestrate(
    tool_name: str,  # noqa: ARG001
    args: dict,
    result: dict | None,
    expanded: bool,
    theme: Theme | None = None,  # noqa: ARG001
) -> str:
    """Format orchestrate tool output for TUI."""
    code = args.get("code", "")
    code_lines = code.strip().split("\n") if code else []
    max_code_lines = len(code_lines) if expanded else 8
    display_code = "\n".join(code_lines[:max_code_lines])
    if code_lines and len(code_lines) > max_code_lines:
        display_code += f"\n# ... ({len(code_lines) - max_code_lines} more lines)"

    text = "orchestrate()"
    if display_code:
        text += f"\n```python\n{display_code}\n```"

    if not result:
        return text

    details = result.get("details", {}) if isinstance(result, dict) else {}
    sequence = details.get("sequence", []) if isinstance(details, dict) else []

    if sequence:
        text += "\n⎿ Sequence:"
        max_events = len(sequence) if expanded else 10
        for idx, event in enumerate(sequence[:max_events], start=1):
            etype = event.get("type", "unknown")
            payload = {k: v for k, v in event.items() if k != "type"}
            summary = ", ".join(f"{k}={v!r}" for k, v in payload.items()) if payload else ""
            text += f"\n  {idx}. {etype}" + (f": {summary}" if summary else "")
        if len(sequence) > max_events:
            text += f"\n  ... ({len(sequence) - max_events} more events)"

    output = get_text_output(result).strip()
    if output:
        lines = output.split("\n")
        max_lines = len(lines) if expanded else 8
        text += "\n⎿ Output:"
        for line in lines[:max_lines]:
            text += f"\n  {line}"
        if len(lines) > max_lines:
            text += f"\n  ... ({len(lines) - max_lines} more lines)"

    return text


# ── Environment ────────────────────────────────────────────────────────────────


@dataclass
class OrchestrateEnvironment:
    """Environment for orchestrating multi-agent execution plans.

    The primary agent gets a single `orchestrate` tool that accepts async Python
    code. The code runs with a `system` object providing thread spawning, queries,
    document sharing, and logging.

    Attributes:
        endpoint: The Endpoint to use for subthreads. If None, inherits from the
            primary agent's state at exec time.
        cwd: Working directory for all subthreads. Defaults to current directory.
        subthread_system_prompt: System prompt injected into every subthread.
            Defaults to SUBTHREAD_SYSTEM_PROMPT.
        max_turns: Maximum turns per subthread before timeout.
    """

    endpoint: Endpoint | None = None
    cwd: Path | None = None
    subthread_system_prompt: str = SUBTHREAD_SYSTEM_PROMPT
    max_turns: int = SUBTHREAD_MAX_TURNS

    # Runtime state (not serialized, reset per orchestrate call)
    _document_store: DocumentStore = field(default_factory=DocumentStore, repr=False)
    _last_sequence: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _saved_results: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _saved_native_messages: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _saved_child_kinds: dict[str, str] = field(default_factory=dict, repr=False)

    def get_name(self) -> str:
        return "orchestrate"

    def get_tools(self) -> list[Tool]:
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="orchestrate",
                    description=(
                        "Execute async Python orchestration code using the `system` API. "
                        "Use trio for parallelism. "
                        "Available: system.thread(), system.query(), system.fromId(), "
                        "system.allocate(), system.log()."
                    ),
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "code": {
                                "type": "string",
                                "description": (
                                    "Async Python code body (contents of an async function). "
                                    "The variable `system` is available. "
                                    "Use `await` for thread/query calls. "
                                    "Use trio.open_nursery() for parallelism. "
                                    "Return a value to set the orchestration output."
                                ),
                            }
                        },
                    ),
                    required=["code"],
                ),
            )
        ]

    def get_system_prompt(self) -> str | None:
        return __doc__  # module docstring serves as the system prompt addition

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        return False

    def get_status_info(self) -> dict[str, str] | None:
        return {
            "docs": str(len(self._document_store.names())),
            "last_events": str(len(self._last_sequence)),
        }

    def get_tool_formatter(
        self, tool_name: str
    ) -> Callable[[str, dict, dict | None, bool, Theme | None], str] | None:
        if tool_name == "orchestrate":
            return format_orchestrate
        return None

    async def on_session_start(self, session_id: str) -> None:
        pass

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        try:
            if tool_call.name == "orchestrate":
                return await self._exec_orchestrate(tool_call, current_state)
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Unknown tool: {tool_call.name}",
            )
        except trio.Cancelled:
            raise
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"{type(e).__name__}: {e}",
            )

    async def _exec_orchestrate(self, tool_call: ToolCall, current_state: AgentState) -> ToolResult:
        code = tool_call.args.get("code", "")
        if not isinstance(code, str) or not code.strip():
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="No code provided",
            )

        # Resolve endpoint: prefer explicit, fall back to primary agent's endpoint
        endpoint = self.endpoint or current_state.actor.endpoint
        cwd = self.cwd or Path.cwd()

        sequence: list[dict[str, Any]] = []

        def on_event(event: dict[str, Any]) -> None:
            sequence.append(event)

        system = System(
            endpoint=endpoint,
            cwd=cwd,
            document_store=self._document_store,
            subthread_system_prompt=self.subthread_system_prompt,
            max_turns=self.max_turns,
            on_event=on_event,
            saved_results=self._saved_results,
            saved_native_messages=self._saved_native_messages,
            saved_child_kinds=self._saved_child_kinds,
        )

        # Wrap code in an async function and exec it
        indented = "\n".join(f"    {line}" for line in code.splitlines())
        fn_src = f"async def _orch(system, trio):\n{indented}\n"

        namespace: dict[str, Any] = {}
        try:
            exec(fn_src, namespace)  # noqa: S102
        except SyntaxError as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"SyntaxError in orchestration code: {e}",
            )

        orch_fn = namespace["_orch"]

        t0 = time.perf_counter()
        output = None
        error_msg = None
        try:
            output = await orch_fn(system, trio)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"

        duration_ms = int((time.perf_counter() - t0) * 1000)
        self._last_sequence = sequence

        # Build result summary
        completed = sum(
            1 for e in sequence if e.get("type") == "thread_end" and e.get("status") == "completed"
        )
        failed = sum(
            1 for e in sequence if e.get("type") == "thread_end" and e.get("status") != "completed"
        )
        thread_total = completed + failed

        if error_msg:
            summary = f"Orchestration error: {error_msg}"
        elif thread_total > 0:
            summary = f"{completed}/{thread_total} threads completed in {duration_ms}ms"
            if failed:
                summary += f", {failed} failed"
        else:
            summary = f"Orchestration completed in {duration_ms}ms"

        if output is not None:
            summary += f"\nOutput: {output}"

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=bool(error_msg),
            content=summary,
            error=error_msg,
            details={
                "sequence": sequence,
                "output": output,
                "duration_ms": duration_ms,
                "docs": self._document_store.names(),
            },
        )

    async def serialize(self) -> dict[str, Any]:
        return {
            "env_kind": "orchestrate",
            "version": "2.1.0",
            "cwd": str(self.cwd) if self.cwd else None,
            "subthread_system_prompt": self.subthread_system_prompt,
            "max_turns": self.max_turns,
            "documents": self._document_store._docs,
            "saved_results": self._saved_results,
            "saved_native_messages": self._saved_native_messages,
            "saved_child_kinds": self._saved_child_kinds,
        }

    @staticmethod
    async def deserialize(data: dict[str, Any]) -> OrchestrateEnvironment:
        assert data.get("env_kind") == "orchestrate"
        env = OrchestrateEnvironment(
            cwd=Path(data["cwd"]) if data.get("cwd") else None,
            subthread_system_prompt=data.get("subthread_system_prompt", SUBTHREAD_SYSTEM_PROMPT),
            max_turns=data.get("max_turns", SUBTHREAD_MAX_TURNS),
        )
        env._document_store._docs = data.get("documents", {})
        env._saved_results = data.get("saved_results", {})
        env._saved_native_messages = data.get("saved_native_messages", {})
        env._saved_child_kinds = data.get("saved_child_kinds", {})
        return env
