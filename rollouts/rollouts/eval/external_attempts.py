from __future__ import annotations

import inspect
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

import trio

from ..core import Message, Trajectory
from ..drivers import ClaudeDriver, CodexDriver, run_driver_to_trajectory
from ..dtypes import StreamChunk
from ..training.types import AttemptResult, ProblemRow, Status

_event_logger = logging.getLogger("rollouts.eval.events")

PromptBuilder = Callable[[dict[str, Any]], str]
ExternalRuntime = Literal["claude_code", "codex", "openhands"]


def _make_eval_on_event(
    sample_id: str,
    caller_on_event: Callable[[Any], Awaitable[None]] | None,
) -> Callable[[Any], Awaitable[None]]:
    """Emit coarse external-driver progress while forwarding real events."""
    from ..dtypes import LLMCallStart, ToolCallEnd, ToolCallStart

    turn: list[int] = [0]

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
                extra={"sample_id": sample_id, "turn": turn[0], "status": "calling tool..."},
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
ProjectedTrajectoryAdapter = Callable[
    [str, str, dict[str, Any], Path, Any],
    Awaitable[ExternalAttemptArtifact],
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


def _result_from_artifact(
    *,
    sample_data: dict[str, Any],
    sample_id: str,
    artifact: ExternalAttemptArtifact,
) -> AttemptResult:
    problem = ProblemRow(
        problem_id=sample_id,
        payload=dict(sample_data),
        metadata=_sample_metadata(sample_data),
    )
    return AttemptResult(
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
) -> AttemptResult:
    # TODO(external-agent-args): once that helper exists, decide whether
    # `AgentRunSpec.external_agent_args` should lower into this path too, or
    # remain explicitly launcher-only. The built-in helper now exists, but
    # benchmark-specific wrappers still pass runtime-specific kwargs manually.
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

    return _result_from_artifact(sample_data=sample_data, sample_id=sample_id, artifact=artifact)


def _trajectory_adapter_for_runtime(runtime: ExternalRuntime) -> TrajectoryAdapter:
    if runtime == "claude_code":
        return trajectory_from_claude_code
    if runtime == "codex":
        return trajectory_from_codex
    if runtime == "openhands":
        return trajectory_from_openhands
    raise ValueError(f"Unsupported external runtime: {runtime}")


def make_external_attempt_executor(
    runtime: ExternalRuntime,
    *,
    prompt_builder: PromptBuilder,
    **trajectory_kwargs: Any,
) -> Callable[[dict[str, Any], str, Any | None, Any], Awaitable[AttemptResult]]:
    """Build the standard autonomous external-runtime attempt executor.

    Config authors supply the prompt builder, choose a built-in runtime, and
    pass runtime-specific kwargs directly. The shared lowering from prompt ->
    external runtime -> AttemptResult stays inside rollouts.
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
) -> ProjectedTrajectoryAdapter:
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
        cwd: Path,
        run_config: Any,
    ) -> ExternalAttemptArtifact:
        artifact_or_trajectory = base_adapter(
            prompt,
            sample_id,
            sample_data,
            cwd=cwd,
            run_config=run_config,
            **trajectory_kwargs,
        )
        if inspect.isawaitable(artifact_or_trajectory):
            artifact_or_trajectory = await artifact_or_trajectory
        if isinstance(artifact_or_trajectory, Trajectory):
            return ExternalAttemptArtifact(trajectory=artifact_or_trajectory)
        return artifact_or_trajectory

    return projected_adapter


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


async def trajectory_from_openhands(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    cwd: Path | None = None,
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
    del sample_data, run_config

    if api_key_env_var is not None and not os.environ.get(api_key_env_var):
        raise RuntimeError(f"Required environment variable {api_key_env_var} is not set")

    cli = shutil.which("openhands")
    if cli is None:
        raise RuntimeError(
            "OpenHands CLI not found. Install from https://docs.all-hands.dev/usage/installation"
        )

    workdir = Path(cwd or Path.cwd()).resolve()
    cmd = [
        cli,
        "--non-interactive",
        "--task",
        prompt,
        "--model",
        model or os.environ.get("OPENHANDS_DEFAULT_MODEL", "anthropic/claude-sonnet-4-5"),
        "--runtime",
        _normalize_openhands_runtime(runtime),
        "--workspace",
        str(workdir),
    ]

    resolved_environment = _normalize_openhands_environment(
        environment,
        allowed_tools=allowed_tools,
    )
    if resolved_environment:
        cmd.extend(["--environment", resolved_environment])

    if agent_cls:
        cmd.extend(["--agent-cls", agent_cls])
    if max_iterations is not None:
        cmd.extend(["--max-iterations", str(max_iterations)])

    proc = await trio.lowlevel.open_process(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_buffer = bytearray()
    stderr_buffer = bytearray()

    async def _read_stream(stream: trio.abc.ReceiveStream | None, sink: bytearray) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.receive_some(4096)
            if not chunk:
                break
            sink.extend(chunk)

    with trio.move_on_after(timeout_seconds) as cancel_scope:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(_read_stream, proc.stdout, stdout_buffer)
            nursery.start_soon(_read_stream, proc.stderr, stderr_buffer)
            await proc.wait()
            nursery.cancel_scope.cancel()

    if cancel_scope.cancelled_caught:
        proc.kill()
        raise TimeoutError(f"OpenHands run timed out after {timeout_seconds:.1f}s")

    stdout_text = stdout_buffer.decode("utf-8", errors="replace")
    stderr_text = stderr_buffer.decode("utf-8", errors="replace")
    combined_output = "\n".join(part for part in (stdout_text, stderr_text) if part).strip()

    session_id = None
    for line in reversed(combined_output.splitlines()):
        candidate = line.strip()
        if _looks_like_session_id(candidate):
            session_id = candidate
            break
    if session_id is None:
        raise RuntimeError(
            "OpenHands did not print a recognizable session id.\n"
            f"stdout:\n{stdout_text}\n\nstderr:\n{stderr_text}"
        )

    api_url = os.environ.get("OPENHANDS_API_URL", "http://localhost:3000")
    trajectory = await trio.to_thread.run_sync(
        _trajectory_from_openhands_session, api_url, session_id
    )

    metadata: dict[str, Any] = {
        "runtime": "openhands",
        "driver": "openhands",
        "session_id": session_id,
        "api_url": api_url,
        "workspace": str(workdir),
    }
    if model:
        metadata["model"] = model
    if api_key_env_var is not None:
        metadata["api_key_env_var"] = api_key_env_var
    if max_iterations is not None:
        metadata["max_iterations"] = max_iterations
    if agent_cls:
        metadata["agent_cls"] = agent_cls
    if resolved_environment:
        metadata["environment"] = resolved_environment
    if proc.returncode is not None:
        metadata["returncode"] = proc.returncode

    status = Status.COMPLETED if proc.returncode in (None, 0) else Status.ABORTED
    if combined_output:
        metadata["openhands_output"] = combined_output[-8000:]

    return ExternalAttemptArtifact(
        trajectory=trajectory,
        metadata=metadata,
        status=status,
    )


def _trajectory_from_openhands_session(api_url: str, session_id: str) -> Trajectory:
    import requests

    response = requests.get(
        f"{api_url.rstrip('/')}/api/conversations/{session_id}/events",
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "events" in payload:
        events = payload["events"]
    else:
        events = payload
    if not isinstance(events, list):
        raise TypeError(f"Unexpected OpenHands events payload: {type(events)!r}")

    messages: list[Message] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        source = str(event.get("source") or event.get("role") or "").lower()
        content = event.get("message") or event.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content if part)
        text = str(content).strip()
        if not text:
            continue

        if source in {"user", "task"}:
            role = "user"
        elif source in {"assistant", "agent"}:
            role = "assistant"
        elif source in {"system"}:
            role = "system"
        else:
            role = "assistant"
        messages.append(Message(role=role, content=text))

    if not messages:
        raise RuntimeError(f"OpenHands session {session_id} returned no importable messages")
    return Trajectory(messages=messages)
