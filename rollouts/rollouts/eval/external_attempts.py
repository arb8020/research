from __future__ import annotations

import inspect
import json
import logging
import os
import re
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trio

from ..core import Message, Trajectory
from ..drivers import ClaudeDriver, CodexDriver, run_driver_to_trajectory
from ..dtypes import StreamChunk
from ..training.types import AttemptRow, ProblemRow, Status

_event_logger = logging.getLogger("rollouts.eval.events")

PromptBuilder = Callable[[dict[str, Any]], str]


def _make_eval_on_event(
    sample_id: str,
    caller_on_event: Callable[[Any], Awaitable[None]] | None,
) -> Callable[[Any], Awaitable[None]]:
    """Return an on_event handler that emits turn events to _event_logger.

    External drivers (ClaudeDriver, CodexDriver) emit typed dataclass events,
    not StreamChunk.  LLMCallStart signals a new turn; ToolCallStart/End give
    finer-grained status.  We forward these to _event_logger so events.jsonl
    gets turn-level progress during external evals.

    Composes with any existing caller on_event so both receive every event.
    """
    from ..dtypes import LLMCallStart, ToolCallEnd, ToolCallStart

    turn: list[int] = [0]  # mutable cell so inner async fn can update it

    async def on_event(event: Any) -> None:
        if isinstance(event, LLMCallStart):
            _event_logger.info(
                "turn",
                extra={"sample_id": sample_id, "turn": turn[0], "status": "streaming..."},
            )
            turn[0] += 1
        elif isinstance(event, ToolCallStart):
            _event_logger.info(
                "turn",
                extra={
                    "sample_id": sample_id,
                    "turn": turn[0],
                    "status": "calling tool...",
                },
            )
        elif isinstance(event, ToolCallEnd):
            _event_logger.info(
                "turn",
                extra={"sample_id": sample_id, "turn": turn[0], "status": "tool done"},
            )
        if caller_on_event is not None:
            await caller_on_event(event)

    return on_event


@dataclass(frozen=True)
class ExternalAttemptArtifact:
    trajectory: Trajectory
    metadata: dict[str, Any] = field(default_factory=dict)
    environment_state: dict[str, Any] | None = None
    status: Status = Status.COMPLETED


ExternalAttemptResult = ExternalAttemptArtifact | Trajectory


TrajectoryAdapter = Callable[
    ...,
    ExternalAttemptResult | Awaitable[ExternalAttemptResult],
]


def _trajectory_adapter_accepts_run_config(trajectory_adapter: TrajectoryAdapter) -> bool:
    params = inspect.signature(trajectory_adapter).parameters.values()
    for param in params:
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "run_config":
            return True
    return False


def _sample_metadata(sample_data: dict[str, Any]) -> dict[str, Any]:
    raw = sample_data.get("metadata")
    return dict(raw) if isinstance(raw, dict) else {}


def _make_raw_driver_line_handler(
    run_config: Any | None,
    *,
    driver: str,
) -> Callable[[str], Awaitable[None]] | None:
    on_chunk = getattr(run_config, "on_chunk", None)
    if on_chunk is None:
        return None

    async def emit(raw_line: str) -> None:
        await on_chunk(
            StreamChunk(
                "raw_driver_line",
                {
                    "driver": driver,
                    "raw_line": raw_line,
                },
            )
        )

    return emit


def _attempt_from_artifact(
    *,
    sample_data: dict[str, Any],
    sample_id: str,
    artifact: ExternalAttemptArtifact,
) -> AttemptRow:
    problem = ProblemRow(
        problem_id=sample_id,
        payload=dict(sample_data),
        metadata=_sample_metadata(sample_data),
    )
    return AttemptRow(
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
) -> AttemptRow:
    del environment
    prompt = prompt_builder(sample_data)
    if _trajectory_adapter_accepts_run_config(trajectory_adapter):
        artifact_or_trajectory = trajectory_adapter(
            prompt,
            sample_id,
            sample_data,
            run_config=run_config,
        )
    else:
        artifact_or_trajectory = trajectory_adapter(prompt, sample_id, sample_data)
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

    return _attempt_from_artifact(sample_data=sample_data, sample_id=sample_id, artifact=artifact)


async def trajectory_from_claude_code(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    cwd: Path,
    run_config: Any | None = None,
    model: str = "sonnet",
    include_partial: bool = True,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    timeout_seconds: float = 600.0,
) -> ExternalAttemptArtifact:
    del sample_data
    driver = ClaudeDriver(
        cwd=cwd,
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
            "cwd": str(cwd),
            "session_id": driver.session_id,
        },
    )


async def trajectory_from_codex(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    cwd: Path,
    run_config: Any | None = None,
    model: str = "gpt-5.1-codex-mini",
    sandbox: str = "read-only",
    timeout_seconds: float = 600.0,
) -> ExternalAttemptArtifact:
    del sample_data
    driver = CodexDriver(
        cwd=cwd,
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
            "cwd": str(cwd),
            "sandbox": sandbox,
            "session_id": driver.session_id,
        },
    )


def _resolve_openhands_api_key(model: str, api_key_env_var: str | None) -> tuple[str, str]:
    if api_key_env_var is not None:
        key = os.environ.get(api_key_env_var, "")
        if not key:
            raise ValueError(f"{api_key_env_var} must be set for OpenHands runtime")
        return key, api_key_env_var

    if model.startswith("claude"):
        env_var = "ANTHROPIC_API_KEY"
    else:
        env_var = "OPENAI_API_KEY"
    key = os.environ.get(env_var, "")
    if not key:
        raise ValueError(f"{env_var} must be set for OpenHands runtime")
    return key, env_var


def _parse_openhands_json_events(stdout_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    marker = "--JSON Event--"
    lines = stdout_text.splitlines()
    i = 0

    while i < len(lines):
        if lines[i].strip() != marker:
            i += 1
            continue

        i += 1
        chunk: list[str] = []
        while i < len(lines):
            candidate = "\n".join(chunk + [lines[i]])
            try:
                event = json.loads(candidate)
            except json.JSONDecodeError:
                chunk.append(lines[i])
                i += 1
                continue
            else:
                events.append(event)
                i += 1
                break
    return events


def _message_from_openhands_event(event: dict[str, Any]) -> Message | None:
    if event.get("kind") != "MessageEvent":
        return None

    llm_message = event.get("llm_message")
    if not isinstance(llm_message, dict):
        return None

    role = llm_message.get("role")
    if role not in {"user", "assistant"}:
        source = event.get("source")
        if source == "agent":
            role = "assistant"
        elif source == "user":
            role = "user"
        else:
            return None

    raw_content = llm_message.get("content", [])
    if not isinstance(raw_content, list):
        return None

    text_parts: list[str] = []
    for block in raw_content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))

    text = "".join(text_parts).strip()
    if not text:
        return None

    return Message(role=role, content=text)


async def trajectory_from_openhands(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    cwd: Path,
    model: str,
    api_key_env_var: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 600.0,
) -> ExternalAttemptArtifact:
    del sample_data

    openhands_bin = shutil.which("openhands") or str(Path.home() / ".local" / "bin" / "openhands")
    if not Path(openhands_bin).exists():
        raise RuntimeError("OpenHands CLI not found. Install with `uv tool install openhands`.")

    api_key, resolved_api_key_env = _resolve_openhands_api_key(model, api_key_env_var)
    env = dict(os.environ)
    env["LLM_API_KEY"] = api_key
    env["LLM_MODEL"] = model
    if base_url is not None:
        env["LLM_BASE_URL"] = base_url

    cmd = [
        openhands_bin,
        "--headless",
        "--json",
        "--override-with-envs",
        "-t",
        prompt,
    ]

    completed = None
    with trio.move_on_after(timeout_seconds) as timeout_scope:
        completed = await trio.run_process(
            cmd,
            capture_stdout=True,
            capture_stderr=True,
            cwd=str(cwd),
            env=env,
            check=False,
        )
    if timeout_scope.cancelled_caught:
        raise TimeoutError(f"OpenHands timed out after {timeout_seconds}s")
    assert completed is not None

    stdout_text = completed.stdout.decode("utf-8", errors="replace")
    stderr_text = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            "OpenHands run failed "
            f"(exit={completed.returncode}). stdout tail={stdout_text[-400:]!r} "
            f"stderr tail={stderr_text[-400:]!r}"
        )

    events = _parse_openhands_json_events(stdout_text)
    messages = [message for event in events if (message := _message_from_openhands_event(event))]
    if not messages:
        raise RuntimeError(
            f"OpenHands produced no parseable messages. stdout tail={stdout_text[-800:]!r}"
        )

    conversation_match = re.search(r"Conversation ID:\s*([0-9a-f-]+)", stdout_text)
    conversation_id = conversation_match.group(1) if conversation_match else None

    trajectory = Trajectory(
        messages=messages,
        metadata={
            "runtime": "openhands",
            "conversation_id": conversation_id,
        },
    )
    return ExternalAttemptArtifact(
        trajectory=trajectory,
        metadata={
            "runtime": "openhands",
            "driver": "openhands",
            "model": model,
            "cwd": str(cwd),
            "conversation_id": conversation_id,
            "api_key_env_var": resolved_api_key_env,
            "stdout_tail": stdout_text[-800:],
            "stderr_tail": stderr_text[-400:],
        },
    )
