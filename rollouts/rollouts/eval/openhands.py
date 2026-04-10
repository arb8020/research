"""Driver for the OpenHands CLI (headless JSON mode)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from functools import partial
from pathlib import Path
from typing import Any

import trio

from ..core import Message, Trajectory
from ..drivers.runner import _make_raw_driver_line_handler
from ..environments.local_workspace_resource import LocalWorkspaceResource
from ..environments.resources import SandboxWorkspaceResource
from ..training.types import Status
from .types import ExternalAttemptArtifact


def _coerce_tool_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_openhands_tools(allowed_tools: list[str] | None) -> str | None:
    if not allowed_tools:
        return None
    normalized = [_coerce_tool_string(tool) for tool in allowed_tools]
    tools = [tool for tool in normalized if tool is not None]
    if not tools:
        return None
    return ",".join(sorted(set(tools)))


def _normalize_openhands_runtime(value: Any) -> str:
    runtime = _coerce_tool_string(value)
    if runtime is None:
        return "docker"
    return runtime


def _normalize_openhands_environment(
    value: Any,
    *,
    allowed_tools: list[str] | None,
) -> str:
    environment = _coerce_tool_string(value)
    if environment is not None:
        return environment
    tool_string = _normalize_openhands_tools(allowed_tools)
    if tool_string is None:
        return "default"
    return f"{tool_string},finish"


def _looks_like_session_id(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F-]{16,}", value))


def _flatten_openhands_content(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    continue
                normalized = str(text).strip()
                if normalized:
                    parts.append(normalized)
                continue
            normalized = str(item).strip()
            if normalized:
                parts.append(normalized)
        if not parts:
            return None
        return "\n".join(parts)
    normalized = str(value).strip()
    return normalized or None


def _extract_json_objects_from_marked_output(
    raw_output: str, *, marker: str
) -> list[dict[str, Any]]:
    lines = raw_output.splitlines()
    payloads: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != marker:
            i += 1
            continue
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines) or lines[i].strip() != "{":
            continue

        json_lines: list[str] = []
        depth = 0
        in_string = False
        escaped = False
        while i < len(lines):
            line = lines[i]
            json_lines.append(line)
            for ch in line:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
            i += 1
            if depth == 0 and json_lines:
                break

        if depth != 0:
            continue
        try:
            payload = json.loads("\n".join(json_lines))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _trajectory_from_openhands_json_output(raw_output: str) -> Trajectory:
    events = _extract_json_objects_from_marked_output(raw_output, marker="--JSON Event--")
    messages: list[Message] = []
    for event in events:
        _append_openhands_event(messages, event)

    if not messages:
        raise RuntimeError("OpenHands produced no importable JSON events")
    return Trajectory(messages=messages)


def _append_openhands_event(messages: list[Message], event: dict[str, Any]) -> None:
    kind = str(event.get("kind") or "")
    source = str(event.get("source") or "").lower()
    if kind == "MessageEvent":
        llm_message = event.get("llm_message")
        if not isinstance(llm_message, dict):
            return
        raw_role = str(llm_message.get("role") or source or "assistant").lower()
        role = raw_role if raw_role in {"system", "user", "assistant", "tool"} else "assistant"
        content = _flatten_openhands_content(llm_message.get("content"))
        if content is None:
            return
        details = {
            "openhands_kind": kind,
            "openhands_source": source or None,
        }
        for key in ("id", "timestamp", "llm_response_id"):
            if event.get(key) is not None:
                details[key] = event[key]
        messages.append(Message(role=role, content=content, details=details))
        return

    if kind == "ActionEvent":
        thought = _flatten_openhands_content(event.get("thought"))
        action = event.get("action")
        action_message = None
        if isinstance(action, dict):
            action_message = _flatten_openhands_content(action.get("message"))
        content_parts = [part for part in (thought, action_message) if part]
        if not content_parts:
            return
        details = {
            "openhands_kind": kind,
            "openhands_source": source or None,
            "action": action,
            "tool_call": event.get("tool_call"),
            "tool_name": event.get("tool_name"),
        }
        for key in ("id", "timestamp", "summary", "reasoning_content", "tool_call_id"):
            if event.get(key) is not None:
                details[key] = event[key]
        messages.append(
            Message(role="assistant", content="\n\n".join(content_parts), details=details)
        )
        return

    if kind == "ObservationEvent":
        observation = event.get("observation")
        content = None
        if isinstance(observation, dict):
            content = _flatten_openhands_content(observation.get("content"))
        if content is None:
            return
        details = {
            "openhands_kind": kind,
            "openhands_source": source or None,
            "observation": observation,
            "tool_name": event.get("tool_name"),
        }
        for key in ("id", "timestamp", "action_id", "tool_call_id"):
            if event.get(key) is not None:
                details[key] = event[key]
        messages.append(
            Message(
                role="tool",
                content=content,
                tool_call_id=event.get("tool_call_id"),
                details=details,
            )
        )


class _OpenHandsJsonEventStreamParser:
    def __init__(self) -> None:
        self._awaiting_marker = True
        self._awaiting_object_start = False
        self._json_lines: list[str] = []
        self._depth = 0
        self._in_string = False
        self._escaped = False

    def feed_line(self, line: str) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        stripped = line.strip()
        if self._awaiting_marker:
            if stripped == "--JSON Event--":
                self._awaiting_marker = False
                self._awaiting_object_start = True
            return payloads

        if self._awaiting_object_start:
            if not stripped:
                return payloads
            if stripped != "{":
                self._awaiting_marker = True
                self._awaiting_object_start = False
                return payloads
            self._awaiting_object_start = False
            self._json_lines = []
            self._depth = 0
            self._in_string = False
            self._escaped = False

        self._json_lines.append(line)
        for ch in line:
            if self._escaped:
                self._escaped = False
                continue
            if ch == "\\":
                self._escaped = True
                continue
            if ch == '"':
                self._in_string = not self._in_string
                continue
            if self._in_string:
                continue
            if ch == "{":
                self._depth += 1
            elif ch == "}":
                self._depth -= 1

        if self._depth != 0:
            return payloads

        try:
            payload = json.loads("\n".join(self._json_lines))
        except json.JSONDecodeError:
            payload = None

        self._awaiting_marker = True
        self._awaiting_object_start = False
        self._json_lines = []
        self._depth = 0
        self._in_string = False
        self._escaped = False
        if isinstance(payload, dict):
            payloads.append(payload)
        return payloads


async def trajectory_from_openhands(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    workspace: SandboxWorkspaceResource | LocalWorkspaceResource | None = None,
    run_config: Any | None = None,
    model: str | None = None,
    timeout_seconds: float = 600.0,
    max_iterations: int | None = None,
    allowed_tools: list[str] | None = None,
    agent_cls: str | None = None,
    api_key_env_var: str | None = None,
    runtime: str = "docker",
    environment: str | None = None,
) -> ExternalAttemptArtifact:
    del sample_data

    if api_key_env_var is not None and not os.environ.get(api_key_env_var):
        raise RuntimeError(f"Required environment variable {api_key_env_var} is not set")

    cli = shutil.which("openhands")
    if cli is None:
        raise RuntimeError(
            "OpenHands CLI not found. Install from https://docs.all-hands.dev/usage/installation"
        )

    workdir = (
        Path(workspace.working_dir).resolve() if workspace is not None else Path.cwd().resolve()
    )
    if allowed_tools:
        raise ValueError(
            "OpenHands CLI no longer accepts benchmark-level allowed tool filtering; "
            "do not pass `allowed_tools` to `trajectory_from_openhands`."
        )
    if agent_cls is not None:
        raise ValueError(
            "OpenHands CLI no longer exposes `--agent-cls` on the top-level headless path; "
            "do not pass `agent_cls` here."
        )
    normalized_runtime = _normalize_openhands_runtime(runtime)
    if normalized_runtime != "docker":
        raise ValueError(
            "OpenHands runtime selection is no longer controlled by a top-level CLI flag; "
            f"got runtime={runtime!r}."
        )
    if environment is not None:
        raise ValueError(
            "OpenHands environment selection is no longer controlled by a top-level CLI flag; "
            f"got environment={environment!r}."
        )
    if max_iterations is not None:
        raise ValueError(
            "OpenHands max-iteration control is not currently wired through the headless JSON CLI."
        )

    cmd = [
        cli,
        "--headless",
        "--json",
        "--always-approve",
        "--override-with-envs",
        "--task",
        prompt,
    ]

    env = os.environ.copy()
    if model is not None:
        env["LLM_MODEL"] = model
    else:
        env.setdefault(
            "LLM_MODEL", os.environ.get("OPENHANDS_DEFAULT_MODEL", "anthropic/claude-sonnet-4-5")
        )
    if api_key_env_var is not None:
        env["LLM_API_KEY"] = os.environ[api_key_env_var]
    elif not env.get("LLM_API_KEY"):
        for fallback_key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            fallback = env.get(fallback_key)
            if fallback:
                env["LLM_API_KEY"] = fallback
                break

    if not env.get("LLM_API_KEY"):
        raise RuntimeError(
            "OpenHands headless JSON mode requires `LLM_API_KEY`; none was provided "
            "via `api_key_env_var`, `LLM_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`."
        )

    raw_line_handler = _make_raw_driver_line_handler(run_config, driver="openhands")
    parser = _OpenHandsJsonEventStreamParser()
    messages: list[Message] = []
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    conversation_id: str | None = None

    async def _consume_stream(
        stream: trio.abc.ReceiveStream | None,
        *,
        sink: list[str],
        parse_events: bool,
    ) -> None:
        nonlocal conversation_id
        if stream is None:
            return
        buffer = ""
        while True:
            chunk = await stream.receive_some(4096)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            while True:
                newline = buffer.find("\n")
                if newline < 0:
                    break
                line = buffer[:newline]
                buffer = buffer[newline + 1 :]
                sink.append(line)
                if parse_events:
                    if raw_line_handler is not None:
                        await raw_line_handler(line)
                    for event in parser.feed_line(line):
                        _append_openhands_event(messages, event)
                    if conversation_id is None:
                        match = re.search(r"Conversation ID:\s*([0-9a-fA-F-]{16,})", line)
                        if match:
                            conversation_id = match.group(1)
        if buffer:
            sink.append(buffer)
            if parse_events:
                if raw_line_handler is not None:
                    await raw_line_handler(buffer)
                for event in parser.feed_line(buffer):
                    _append_openhands_event(messages, event)
                if conversation_id is None:
                    match = re.search(r"Conversation ID:\s*([0-9a-fA-F-]{16,})", buffer)
                    if match:
                        conversation_id = match.group(1)

    proc = await trio.lowlevel.open_process(
        cmd,
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    with trio.move_on_after(timeout_seconds) as cancel_scope:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(
                partial(_consume_stream, proc.stdout, sink=stdout_lines, parse_events=True)
            )
            nursery.start_soon(
                partial(_consume_stream, proc.stderr, sink=stderr_lines, parse_events=False)
            )
            await proc.wait()
            nursery.cancel_scope.cancel()

    if cancel_scope.cancelled_caught:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"OpenHands run timed out after {timeout_seconds:.1f}s")

    stdout_text = "\n".join(stdout_lines).strip()
    stderr_text = "\n".join(stderr_lines).strip()
    combined_output = "\n".join(part for part in (stdout_text, stderr_text) if part).strip()
    if not messages:
        trajectory = _trajectory_from_openhands_json_output(combined_output)
    else:
        trajectory = Trajectory(messages=messages)

    metadata: dict[str, Any] = {
        "runtime": "openhands",
        "driver": "openhands",
        "workspace": str(workdir),
        "returncode": proc.returncode,
    }
    if env.get("LLM_MODEL"):
        metadata["model"] = env["LLM_MODEL"]
    if api_key_env_var is not None:
        metadata["api_key_env_var"] = api_key_env_var
    if conversation_id is not None:
        metadata["conversation_id"] = conversation_id

    status = Status.COMPLETED if proc.returncode == 0 else Status.ABORTED
    if combined_output:
        metadata["openhands_output"] = combined_output[-8000:]

    return ExternalAttemptArtifact(
        trajectory=trajectory,
        metadata=metadata,
        status=status,
    )
