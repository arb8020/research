from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal

from ..core import Message, Trajectory
from ..drivers import (
    ClaudeACPDriver,
    ClaudeDriver,
    CodexACPDriver,
    CodexDriver,
    run_driver_to_trajectory,
)
from ..drivers.runner import _make_raw_driver_line_handler
from ..environments.local_workspace_resource import LocalWorkspaceResource
from ..environments.resources import SandboxWorkspaceResource
from ..training.types import DatasetRow, RowAttempt
from .mini_swe_agent import trajectory_from_mini_swe_agent
from .openhands import trajectory_from_openhands
from .remote_runtime import (
    _make_eval_on_event,
)
from .types import ExternalAttemptArtifact

PromptBuilder = Callable[[dict[str, Any]], str]
# TODO(external-runtime-sum-type): "claude_code", "codex", and "openhands" are
# currently first-class external runtimes. The next likely addition is
# `mini_swe_agent`, which should enter through this shared runtime surface
# rather than via another benchmark-local wrapper. If more runtimes arrive,
# consider replacing this Literal with an explicit sum type plus runtime
# capability metadata instead of growing ad hoc string branching.
ExternalRuntime = Literal[
    "claude_code",
    "claude_acp",
    "codex",
    "codex_acp",
    "openhands",
    "mini_swe_agent",
]

ExternalAttemptResult = ExternalAttemptArtifact | Trajectory

TrajectoryAdapter = Callable[
    ...,
    ExternalAttemptResult | Awaitable[ExternalAttemptResult],
]

WorkspaceResource = SandboxWorkspaceResource | LocalWorkspaceResource


def _trajectory_adapter_kwargs(
    trajectory_adapter: TrajectoryAdapter,
    workspace: WorkspaceResource,
    *,
    run_config: Any | None,
) -> dict[str, Any]:
    signature = inspect.signature(trajectory_adapter)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )

    adapter_kwargs: dict[str, Any] = {}
    if "workspace" in parameters:
        adapter_kwargs["workspace"] = workspace
    elif "cwd" in parameters:
        adapter_kwargs["cwd"] = workspace.working_dir
    elif accepts_var_kwargs:
        adapter_kwargs["workspace"] = workspace

    if "run_config" in parameters or accepts_var_kwargs:
        adapter_kwargs["run_config"] = run_config

    return adapter_kwargs


def _sample_metadata(sample_data: dict[str, Any]) -> dict[str, Any]:
    raw = sample_data.get("metadata")
    return dict(raw) if isinstance(raw, dict) else {}


def _result_from_artifact(
    *,
    sample_data: dict[str, Any],
    sample_id: str,
    artifact: ExternalAttemptArtifact,
) -> RowAttempt:
    problem = DatasetRow(
        problem_id=sample_id,
        payload=dict(sample_data),
        metadata=_sample_metadata(sample_data),
    )
    return RowAttempt(
        attempt_id=sample_id,
        problem=problem,
        trajectory=artifact.trajectory,
        environment_state=artifact.environment_state,
        status=artifact.status,
        metadata=dict(artifact.metadata),
    )


async def execute_external_attempt(
    sample_data: dict[str, Any],
    sample_id: str,
    environment: Any | None,
    run_config: Any,
    *,
    prompt_builder: PromptBuilder,
    trajectory_adapter: TrajectoryAdapter,
) -> RowAttempt:
    # TODO(external-agent-args): once that helper exists, decide whether
    # `AgentRunSpec.external_agent_args` should lower into this path too, or
    # remain explicitly launcher-only. The built-in helper now exists, but
    # benchmark-specific wrappers still pass runtime-specific kwargs manually.
    # TODO(external-session): environment is accepted but immediately discarded.
    # This is the sharpest expression of the external agent path's session gap:
    # the environment exists, carries MCP servers and workspace state, but the
    # external agent never sees it. The trajectory_adapter runs blind — it spawns
    # the CLI, waits for it to finish, and returns a Trajectory with no per-turn
    # persistence and no mid-run feedback from the environment.
    #
    # Fix: thread environment through to the runtime so it can:
    #   1. Pass environment.get_mcp_servers() as --mcp-config to the CLI
    #   2. Write turns to the session store as they arrive (not just at run-end)
    #   3. Resume from a prior session_id on the environment if one exists
    #
    workspace = _coerce_workspace_for_attempt(environment)
    prompt = prompt_builder(sample_data)
    adapter_kwargs = _trajectory_adapter_kwargs(
        trajectory_adapter,
        workspace,
        run_config=run_config,
    )
    artifact_or_trajectory = trajectory_adapter(
        prompt,
        sample_id,
        sample_data,
        **adapter_kwargs,
    )
    if inspect.isawaitable(artifact_or_trajectory):
        artifact_or_trajectory = await artifact_or_trajectory

    if isinstance(artifact_or_trajectory, Trajectory):
        artifact = ExternalAttemptArtifact(trajectory=artifact_or_trajectory)
    else:
        artifact = artifact_or_trajectory

    if not isinstance(artifact, ExternalAttemptArtifact):
        raise TypeError(
            "trajectory_adapter must return Trajectory or ExternalAttemptArtifact "
            f"(got {type(artifact)!r})"
        )

    return _result_from_artifact(sample_data=sample_data, sample_id=sample_id, artifact=artifact)


def _trajectory_adapter_for_runtime(runtime: ExternalRuntime) -> TrajectoryAdapter:
    if runtime == "claude_code":
        return trajectory_from_claude_code
    if runtime == "claude_acp":
        return trajectory_from_claude_acp
    if runtime == "codex":
        return trajectory_from_codex
    if runtime == "codex_acp":
        return trajectory_from_codex_acp
    if runtime == "openhands":
        return trajectory_from_openhands
    if runtime == "mini_swe_agent":
        return trajectory_from_mini_swe_agent
    raise ValueError(f"Unsupported external runtime: {runtime}")


def make_external_attempt_executor(
    runtime: ExternalRuntime,
    *,
    prompt_builder: PromptBuilder,
    **trajectory_kwargs: Any,
) -> Callable[[dict[str, Any], str, Any | None, Any], Awaitable[RowAttempt]]:
    """Build the standard autonomous external-runtime attempt executor.

    Config authors supply the prompt builder, choose a built-in runtime, and
    pass runtime-specific kwargs directly. The shared lowering from prompt ->
    external runtime -> RowAttempt stays inside rollouts.
    """

    trajectory_adapter = partial(
        _trajectory_adapter_for_runtime(runtime),
        **trajectory_kwargs,
    )
    return partial(
        execute_external_attempt,
        prompt_builder=prompt_builder,
        trajectory_adapter=trajectory_adapter,
    )


def make_external_trajectory_adapter(
    runtime: ExternalRuntime,
    **trajectory_kwargs: Any,
) -> Callable[
    [str, str, dict[str, Any], WorkspaceResource, Any], Awaitable[ExternalAttemptArtifact]
]:
    """Build the standard runtime adapter for projected workspaces.

    This is the shared lower-level piece for benchmark-specific wrappers that
    still need to own workspace projection, grading, or other environment-local
    effects before converting the final artifact into their own attempt type.
    """

    base_adapter = _trajectory_adapter_for_runtime(runtime)

    async def projected_adapter(
        prompt: str,
        sample_id: str,
        sample_data: dict[str, Any],
        workspace: WorkspaceResource,
        run_config: Any,
    ) -> ExternalAttemptArtifact:
        adapter_kwargs = dict(
            _trajectory_adapter_kwargs(
                base_adapter,
                workspace,
                run_config=run_config,
            )
        )
        adapter_kwargs.update(trajectory_kwargs)
        artifact_or_trajectory = base_adapter(
            prompt,
            sample_id,
            sample_data,
            **adapter_kwargs,
        )
        if inspect.isawaitable(artifact_or_trajectory):
            artifact_or_trajectory = await artifact_or_trajectory
        if isinstance(artifact_or_trajectory, Trajectory):
            return ExternalAttemptArtifact(trajectory=artifact_or_trajectory)
        return artifact_or_trajectory

    return projected_adapter


def _coerce_workspace_for_attempt(environment: Any | None) -> WorkspaceResource:
    workspace = getattr(environment, "workspace", environment)
    if isinstance(workspace, WorkspaceResource):
        return workspace
    if environment is None:
        return LocalWorkspaceResource.from_existing(Path.cwd())
    raise TypeError(
        "trajectory_adapter requires a workspace-backed environment "
        "with a .workspace attribute or WorkspaceResource itself"
    )


async def trajectory_from_claude_code(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    workspace: WorkspaceResource,
    run_config: Any | None = None,
    model: str = "sonnet",
    include_partial: bool = True,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    timeout_seconds: float = 600.0,
) -> ExternalAttemptArtifact:
    del sample_data
    if not isinstance(workspace, LocalWorkspaceResource):
        raise TypeError(
            "trajectory_from_claude_code requires a LocalWorkspaceResource for local execution"
        )
    cwd = workspace.working_dir
    driver = ClaudeDriver(
        cwd=Path(cwd),
        model=model,
        include_partial=include_partial,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        timeout_seconds=timeout_seconds,
        on_raw_line=_make_raw_driver_line_handler(run_config, driver="claude"),
    )
    on_event = _make_eval_on_event(sample_id, getattr(run_config, "on_chunk", None))
    trajectory = await run_driver_to_trajectory(
        driver,
        prompt,
        sample_id=sample_id,
        on_event=on_event,
    )
    return ExternalAttemptArtifact(
        trajectory=trajectory,
        metadata={
            "runtime": "claude_code",
            "driver": "claude",
            "model": model,
            "cwd": cwd,
            "session_id": driver.session_id,
        },
    )


async def trajectory_from_codex(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    workspace: WorkspaceResource,
    run_config: Any | None = None,
    model: str = "gpt-5.1-codex-mini",
    sandbox: str = "read-only",
    timeout_seconds: float = 600.0,
) -> ExternalAttemptArtifact:
    del sample_data
    if not isinstance(workspace, LocalWorkspaceResource):
        raise TypeError(
            "trajectory_from_codex requires a LocalWorkspaceResource for local execution"
        )
    cwd = workspace.working_dir
    driver = CodexDriver(
        cwd=Path(cwd),
        model=model,
        sandbox=sandbox,
        timeout_seconds=timeout_seconds,
        on_raw_line=_make_raw_driver_line_handler(run_config, driver="codex"),
    )
    on_event = _make_eval_on_event(sample_id, getattr(run_config, "on_chunk", None))
    trajectory = await run_driver_to_trajectory(
        driver,
        prompt,
        sample_id=sample_id,
        on_event=on_event,
    )
    return ExternalAttemptArtifact(
        trajectory=trajectory,
        metadata={
            "runtime": "codex",
            "driver": "codex",
            "model": model,
            "cwd": cwd,
            "sandbox": sandbox,
            "session_id": driver.session_id,
        },
    )


async def trajectory_from_claude_acp(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    workspace: WorkspaceResource,
    run_config: Any | None = None,
    model: str = "claude-agent-acp",
) -> ExternalAttemptArtifact:
    del sample_data
    if not isinstance(workspace, LocalWorkspaceResource):
        raise TypeError(
            "trajectory_from_claude_acp requires a LocalWorkspaceResource for local execution"
        )
    cwd = workspace.working_dir
    driver = ClaudeACPDriver(cwd=cwd, model=model)
    on_event = _make_eval_on_event(sample_id, getattr(run_config, "on_chunk", None))
    trajectory = await run_driver_to_trajectory(
        driver,
        prompt,
        sample_id=sample_id,
        on_event=on_event,
    )
    return ExternalAttemptArtifact(
        trajectory=trajectory,
        metadata={
            "runtime": "claude_acp",
            "driver": "claude_acp",
            "model": model,
            "cwd": cwd,
        },
    )


async def trajectory_from_codex_acp(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    workspace: WorkspaceResource,
    run_config: Any | None = None,
    model: str = "codex-acp",
) -> ExternalAttemptArtifact:
    del sample_data
    if not isinstance(workspace, LocalWorkspaceResource):
        raise TypeError(
            "trajectory_from_codex_acp requires a LocalWorkspaceResource for local execution"
        )
    cwd = workspace.working_dir
    driver = CodexACPDriver(cwd=cwd, model=model)
    on_event = _make_eval_on_event(sample_id, getattr(run_config, "on_chunk", None))
    trajectory = await run_driver_to_trajectory(
        driver,
        prompt,
        sample_id=sample_id,
        on_event=on_event,
    )
    return ExternalAttemptArtifact(
        trajectory=trajectory,
        metadata={
            "runtime": "codex_acp",
            "driver": "codex_acp",
            "model": model,
            "cwd": cwd,
        },
    )


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
