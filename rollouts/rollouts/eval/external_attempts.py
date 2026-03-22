from __future__ import annotations

import inspect
import json
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
from ..drivers.claude import _ClaudeEventParser
from ..drivers.codex import _CodexEventParser
from ..drivers.runner import _EventAccumulator, _FlushAssistantMessage
from ..dtypes import StreamChunk
from ..environments.resources import SandboxWorkspaceResource
from ..training.types import AttemptResult, ProblemRow, Status

_event_logger = logging.getLogger("rollouts.eval.events")

PromptBuilder = Callable[[dict[str, Any]], str]
ExternalRuntime = Literal["claude_code", "codex", "openhands"]
REMOTE_AGENT_USER = "rollouts-agent"


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
RemoteTrajectoryAdapter = Callable[
    [str, str, dict[str, Any], SandboxWorkspaceResource, str, Any],
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


def make_remote_external_trajectory_adapter(
    runtime: ExternalRuntime,
    **trajectory_kwargs: Any,
) -> RemoteTrajectoryAdapter:
    """Build a remote-runtime adapter for real workspace-backed execution.

    TODO(remote-external-runtime): this currently bootstraps Claude Code / Codex
    inside the remote workspace on demand. Once that path is stable, bake the
    CLIs into the runtime image instead of doing per-attempt npm installs.

    TODO(remote-external-runtime-streaming): this only reconstructs the final
    trajectory after the remote process exits. Add true remote stdout streaming
    over the session boundary so live runs get honest incremental progress.
    """

    if runtime == "claude_code":
        return partial(trajectory_from_remote_claude_code, **trajectory_kwargs)
    if runtime == "codex":
        return partial(trajectory_from_remote_codex, **trajectory_kwargs)
    raise ValueError(f"Unsupported remote external runtime: {runtime}")


def _sample_id_slug(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", sample_id).strip("-") or "sample"


def _remote_runtime_state_dir(runtime: str, sample_id: str) -> str:
    return f"/tmp/rollouts-external-runtime/{runtime}/{_sample_id_slug(sample_id)}"


def _remote_runtime_cli(runtime: str) -> tuple[str, str]:
    if runtime == "claude_code":
        return "claude", "@anthropic-ai/claude-code"
    if runtime == "codex":
        return "codex", "@openai/codex"
    raise ValueError(f"Unsupported remote runtime: {runtime}")


def _remote_runtime_env(runtime: str) -> dict[str, str]:
    if runtime == "claude_code":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for remote Claude Code runs")
        return {
            "ANTHROPIC_API_KEY": key,
            "CLAUDE_CODE_ENTRYPOINT": "rollouts-remote-eval",
        }
    if runtime == "codex":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for remote Codex runs")
        return {
            "OPENAI_API_KEY": key,
            "CODEX_ENTRYPOINT": "rollouts-remote-eval",
        }
    raise ValueError(f"Unsupported remote runtime: {runtime}")


def _remote_runtime_bootstrap_command(runtime: str) -> str:
    cli_name, npm_package = _remote_runtime_cli(runtime)
    return (
        "set -euo pipefail\n"
        'NODE_MAJOR="$(node -p \'process.versions.node.split(".")[0]\' 2>/dev/null || echo 0)"\n'
        'if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1 || [ "${NODE_MAJOR}" -lt 18 ]; then\n'
        "  apt-get update\n"
        "  DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates\n"
        "  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -\n"
        "  DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs\n"
        "fi\n"
        f"if ! id -u {REMOTE_AGENT_USER} >/dev/null 2>&1; then\n"
        f"  useradd -m -s /bin/bash {REMOTE_AGENT_USER}\n"
        "fi\n"
        f"if ! command -v {cli_name} >/dev/null 2>&1; then\n"
        f"  npm install -g {npm_package}\n"
        "fi\n"
    )


async def _ensure_remote_runtime_bootstrap(
    workspace: SandboxWorkspaceResource,
    *,
    runtime: str,
    cwd: str,
) -> None:
    marker_dir = f"/tmp/rollouts-external-runtime/{runtime}"
    marker_path = f"{marker_dir}/bootstrap-ready"
    result = await workspace.run(
        (
            "set -euo pipefail\n"
            f"mkdir -p {marker_dir}\n"
            f"if [ ! -f {marker_path} ]; then\n"
            f"{_remote_runtime_bootstrap_command(runtime)}"
            f"  touch {marker_path}\n"
            "fi\n"
        ),
        cwd=cwd,
        timeout=10 * 60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to bootstrap remote {runtime}: {result.stderr or result.stdout}"
        )


async def _build_trajectory_from_remote_jsonl(
    *,
    runtime: Literal["claude_code", "codex"],
    raw_output: str,
    sample_id: str,
    run_config: Any | None,
    driver_name: str,
) -> tuple[Trajectory, str | None]:
    parser = _ClaudeEventParser() if runtime == "claude_code" else _CodexEventParser()
    accumulator = _EventAccumulator()
    on_event = _make_eval_on_event(sample_id, getattr(run_config, "on_chunk", None))
    on_raw_line = _make_raw_driver_line_handler(run_config, driver=driver_name)

    for raw_line in raw_output.splitlines():
        line = raw_line.rstrip("\r")
        if on_raw_line is not None and line:
            await on_raw_line(line)
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        for event in parser.parse(msg):
            if isinstance(event, _FlushAssistantMessage):
                accumulator.handle(event)
                continue
            await on_event(event)
            accumulator.handle(event)

    return Trajectory(messages=accumulator.finalize()), getattr(parser, "_session_id", None)


async def _run_remote_external_runtime(
    *,
    runtime: Literal["claude_code", "codex"],
    prompt: str,
    sample_id: str,
    workspace: SandboxWorkspaceResource,
    cwd: str,
    run_config: Any | None,
    command: list[str],
    timeout_seconds: float,
) -> ExternalAttemptArtifact:
    await _ensure_remote_runtime_bootstrap(workspace, runtime=runtime, cwd=cwd)

    state_dir = _remote_runtime_state_dir(runtime, sample_id)
    prompt_path = f"{state_dir}/prompt.txt"
    env_path = f"{state_dir}/env.json"
    await workspace.run(
        f"mkdir -p {state_dir}",
        cwd=cwd,
        timeout=30.0,
    )
    await workspace.write_file(prompt_path, prompt.encode("utf-8"))
    await workspace.write_file(
        env_path,
        json.dumps(_remote_runtime_env(runtime)).encode("utf-8"),
    )
    await workspace.run(
        f"chown -R {REMOTE_AGENT_USER}:{REMOTE_AGENT_USER} {cwd} {state_dir}",
        cwd=cwd,
        timeout=30.0,
    )

    runner = (
        "python - <<'PY'\n"
        "import json\n"
        "import os\n"
        "import pwd\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"agent = pwd.getpwnam({REMOTE_AGENT_USER!r})\n"
        "env = dict(os.environ)\n"
        f"env.update(json.loads(Path({env_path!r}).read_text()))\n"
        "env.update({\n"
        '    "HOME": agent.pw_dir,\n'
        f'    "USER": {REMOTE_AGENT_USER!r},\n'
        f'    "LOGNAME": {REMOTE_AGENT_USER!r},\n'
        "})\n"
        f"prompt = Path({prompt_path!r}).read_text()\n"
        f"cmd = {command!r}\n"
        "cmd.append(prompt)\n"
        "def demote() -> None:\n"
        "    os.setgid(agent.pw_gid)\n"
        "    os.setuid(agent.pw_uid)\n"
        f"proc = subprocess.run(cmd, cwd={cwd!r}, env=env, text=True, capture_output=True, preexec_fn=demote)\n"
        "sys.stdout.write(proc.stdout)\n"
        "sys.stderr.write(proc.stderr)\n"
        "raise SystemExit(proc.returncode)\n"
        "PY"
    )
    result = await workspace.run(
        runner,
        cwd=cwd,
        timeout=timeout_seconds,
    )
    await workspace.run(
        f"rm -rf {state_dir}",
        cwd=cwd,
        timeout=30.0,
    )
    stdout_text = result.stdout.strip()
    stderr_text = result.stderr.strip()
    if result.returncode == -1:
        detail = stderr_text or "remote command timed out without emitting output"
        raise RuntimeError(f"Remote {runtime} timed out after {timeout_seconds:.0f}s: {detail}")
    if result.returncode != 0 and not stdout_text:
        raise RuntimeError(
            f"Remote {runtime} failed before producing trajectory output: "
            f"{stderr_text or result.stdout}"
        )

    trajectory, session_id = await _build_trajectory_from_remote_jsonl(
        runtime=runtime,
        raw_output=result.stdout,
        sample_id=sample_id,
        run_config=run_config,
        driver_name="claude" if runtime == "claude_code" else "codex",
    )
    metadata: dict[str, Any] = {
        "runtime": runtime,
        "driver": "claude" if runtime == "claude_code" else "codex",
        "cwd": cwd,
        "session_id": session_id,
        "remote_execution": True,
    }
    if result.returncode != 0 and not trajectory.messages:
        detail = stderr_text or "remote runtime returned no usable trajectory output"
        raise RuntimeError(f"Remote {runtime} failed without a usable trajectory: {detail}")
    if result.returncode != 0:
        metadata["remote_stderr"] = result.stderr[-4000:]
    return ExternalAttemptArtifact(
        trajectory=trajectory,
        metadata=metadata,
    )


async def trajectory_from_remote_claude_code(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    workspace: SandboxWorkspaceResource,
    cwd: str,
    run_config: Any | None = None,
    *,
    model: str = "sonnet",
    include_partial: bool = True,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    timeout_seconds: float = 600.0,
) -> ExternalAttemptArtifact:
    del sample_data
    command = [
        "claude",
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
        "--model",
        model,
    ]
    if include_partial:
        command.append("--include-partial-messages")
    if system_prompt:
        command.extend(["--system-prompt", system_prompt])
    if allowed_tools:
        command.extend(["--allowedTools", ",".join(allowed_tools)])
    artifact = await _run_remote_external_runtime(
        runtime="claude_code",
        prompt=prompt,
        sample_id=sample_id,
        workspace=workspace,
        cwd=cwd,
        run_config=run_config,
        command=command,
        timeout_seconds=timeout_seconds,
    )
    artifact.metadata["model"] = model
    return artifact


async def trajectory_from_remote_codex(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    workspace: SandboxWorkspaceResource,
    cwd: str,
    run_config: Any | None = None,
    *,
    model: str = "gpt-5.1-codex-mini",
    sandbox: str = "read-only",
    timeout_seconds: float = 600.0,
) -> ExternalAttemptArtifact:
    del sample_data
    command = [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--model",
        model,
        "--sandbox",
        sandbox,
    ]
    artifact = await _run_remote_external_runtime(
        runtime="codex",
        prompt=prompt,
        sample_id=sample_id,
        workspace=workspace,
        cwd=cwd,
        run_config=run_config,
        command=command,
        timeout_seconds=timeout_seconds,
    )
    artifact.metadata["model"] = model
    artifact.metadata["sandbox"] = sandbox
    return artifact


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
