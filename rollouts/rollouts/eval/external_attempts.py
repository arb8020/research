from __future__ import annotations

import inspect
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

import trio

from ..core import Message, Trajectory
from ..drivers import (
    ClaudeACPDriver,
    ClaudeDriver,
    CodexACPDriver,
    CodexDriver,
    run_driver_to_trajectory,
)
from ..drivers.claude import _ClaudeEventParser
from ..drivers.codex import _CodexEventParser
from ..drivers.runner import _EventAccumulator, _FlushAssistantMessage
from ..dtypes import StreamChunk
from ..environments.resources import SandboxWorkspaceResource, SessionExecSpec
from ..training.types import AttemptResult, DatasetRow, Status

_event_logger = logging.getLogger("rollouts.eval.events")

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
REMOTE_AGENT_USER = "rollouts-agent"

# TODO(runtime-hooks): make runtime capabilities explicit enough for topology-
# aware environments to reject unsupported pairings honestly. Near-term target:
# - LocalSync environments: runtime only needs local workspace execution
# - RemoteResident environments: runtime must support launch in authoritative remote workspace
# - defer local-loop + hijacked-remote-tools until runtimes expose real tool interception hooks


async def _workspace_exec(
    workspace: SandboxWorkspaceResource,
    command: str,
    *,
    cwd: str,
    timeout: float,
) -> Any:
    return await workspace.exec(
        SessionExecSpec(
            command=command,
            cwd=cwd,
            timeout=timeout,
        )
    )


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


@dataclass(frozen=True)
class RemoteRuntimePreparation:
    mode: Literal["preinstalled", "uv"] = "uv"
    uv_config_toml: str | None = None
    npmrc: str | None = "min-release-age=8\nignore-scripts=true\n"


# TODO(remote-runtime-preparation): Modal currently uses per-attempt uv-based
# preparation for remote ACP runs (node/npm + ACP Python deps + npx package
# fetch). That is semantically acceptable for disposable sandboxes but too
# expensive and too mutation-heavy to be the long-term default. Move Modal
# toward a preinstalled runtime/image path, then give SSH/bare-metal a separate
# user-space or preinstalled preparation story instead of inheriting this path.


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
    # TODO(observability): `raw_driver_line` is a raw boundary observation, not
    # a normalized domain event. Keep it available for parser/runtime debugging,
    # but move the canonical info-level journal toward parsed StreamEvents with
    # an explicit event envelope (`source` / `kind` / `payload`) instead of
    # treating escaped vendor JSON as a primary analysis surface.
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


def _make_external_progress_emitter(
    run_config: Any | None,
    *,
    driver: str,
) -> Callable[[str, dict[str, Any]], Awaitable[None]] | None:
    on_chunk = getattr(run_config, "on_chunk", None)
    if on_chunk is None:
        return None

    async def emit(kind: str, payload: dict[str, Any]) -> None:
        await on_chunk(
            StreamChunk(
                kind,
                {
                    "driver": driver,
                    **payload,
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
    problem = DatasetRow(
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
    trajectory after the remote process exits. Replace _run_remote_external_runtime
    with a session-file polling approach: launch the CLI in the background inside
    the sandbox, then poll the native session file it writes
    (~/.claude/projects/.../session.jsonl or ~/.codex/sessions/.../rollout-*.jsonl)
    via repeated workspace.exec(SessionExecSpec(command="tail -c +{offset} {path}", ...))
    calls, feeding each new
    line through the canonical _ClaudeEventParser / _CodexEventParser from
    drivers/claude.py and drivers/codex.py. This gives live progress and uses the
    authoritative session file (which has complete tool arguments) rather than the
    buffered stdout. Falls back to full stdout parse if the session file is not found.
    """

    if runtime == "claude_code":
        return partial(trajectory_from_remote_claude_code, **trajectory_kwargs)
    if runtime == "claude_acp":
        return partial(trajectory_from_remote_claude_acp, **trajectory_kwargs)
    if runtime == "codex":
        return partial(trajectory_from_remote_codex, **trajectory_kwargs)
    if runtime == "codex_acp":
        return partial(trajectory_from_remote_codex_acp, **trajectory_kwargs)
    raise ValueError(f"Unsupported remote external runtime: {runtime}")


def _sample_id_slug(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", sample_id).strip("-") or "sample"


def _remote_runtime_state_dir(runtime: str, sample_id: str) -> str:
    return f"/tmp/rollouts-external-runtime/{runtime}/{_sample_id_slug(sample_id)}"


def _remote_runtime_cli(runtime: str) -> tuple[str, str]:
    if runtime == "claude_code":
        return "claude", "@anthropic-ai/claude-code"
    if runtime == "claude_acp":
        return "claude-agent-acp", "@agentclientprotocol/claude-agent-acp"
    if runtime == "codex":
        return "codex", "@openai/codex"
    if runtime == "codex_acp":
        return "codex-acp", "@zed-industries/codex-acp"
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
    if runtime == "claude_acp":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for remote Claude ACP runs")
        return {
            "ANTHROPIC_API_KEY": key,
        }
    if runtime == "codex":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for remote Codex runs")
        return {
            "CODEX_API_KEY": key,
            "OPENAI_API_KEY": key,
            "CODEX_ENTRYPOINT": "rollouts-remote-eval",
        }
    if runtime == "codex_acp":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for remote Codex ACP runs")
        return {
            "CODEX_API_KEY": key,
            "OPENAI_API_KEY": key,
        }
    raise ValueError(f"Unsupported remote runtime: {runtime}")


def _remote_acp_python(runtime: str) -> str:
    del runtime
    return "/opt/venvs/rollouts/bin/python"


def _remote_codex_acp_auth_payload(api_key: str) -> dict[str, Any]:
    return {
        "auth_mode": "apikey",
        "OPENAI_API_KEY": api_key,
        "tokens": None,
        "last_refresh": None,
    }


def _remote_acp_validate_command(runtime: str) -> str:
    cli_name, _npm_package = _remote_runtime_cli(runtime)
    python_bin = _remote_acp_python(runtime)
    return (
        "set -euo pipefail\n"
        'NODE_MAJOR="$(node -p \'process.versions.node.split(".")[0]\' 2>/dev/null || echo 0)"\n'
        'if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1 || [ "${NODE_MAJOR}" -lt 18 ]; then\n'
        '  echo "missing required Node/npm runtime (need node>=18 and npm)" >&2\n'
        "  exit 1\n"
        "fi\n"
        f"if ! command -v {cli_name} >/dev/null 2>&1; then\n"
        f'  echo "missing preinstalled ACP CLI: {cli_name}" >&2\n'
        "  exit 1\n"
        "fi\n"
        f'{python_bin} -c "import acp, trio_asyncio" >/dev/null 2>&1\n'
    )


def _emit_remote_runtime_stage(
    *,
    runtime: str,
    sample_id: str | None,
    stage: str,
    status: str,
) -> None:
    _event_logger.info(
        "remote_runtime_stage",
        extra={
            "runtime": runtime,
            "sample_id": sample_id,
            "stage": stage,
            "status": status,
        },
    )


def _remote_acp_ensure_node_command() -> str:
    return (
        "set -euo pipefail\n"
        'NODE_MAJOR="$(node -p \'process.versions.node.split(".")[0]\' 2>/dev/null || echo 0)"\n'
        'if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1 || [ "${NODE_MAJOR}" -lt 18 ]; then\n'
        "  apt-get update\n"
        "  DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates\n"
        "  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -\n"
        "  DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs\n"
        "fi\n"
    )


def _remote_acp_ensure_uv_command() -> str:
    return (
        "set -euo pipefail\n"
        "if ! command -v uv >/dev/null 2>&1; then\n"
        '  if [ -x "$HOME/.local/bin/uv" ]; then\n'
        '    export PATH="$HOME/.local/bin:$PATH"\n'
        "  else\n"
        "    if ! command -v curl >/dev/null 2>&1; then\n"
        "      apt-get update\n"
        "      DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates\n"
        "    fi\n"
        "    curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        '    export PATH="$HOME/.local/bin:$PATH"\n'
        "  fi\n"
        "fi\n"
        "command -v uv >/dev/null 2>&1\n"
    )


def _remote_acp_uv_prepare_command(
    runtime: str,
    *,
    uv_config_path: str | None = None,
    npmrc_path: str | None = None,
) -> str:
    python_bin = _remote_acp_python(runtime)
    uv_args = f"--config-file {shlex.quote(uv_config_path)} " if uv_config_path else ""
    npm_env = f"export NPM_CONFIG_USERCONFIG={shlex.quote(npmrc_path)}\n" if npmrc_path else ""
    return (
        "set -euo pipefail\n"
        f"{npm_env}"
        'if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then\n'
        '  export PATH="$HOME/.local/bin:$PATH"\n'
        "fi\n"
        f'{python_bin} -c "import acp, trio_asyncio" >/dev/null 2>&1 || '
        f"uv pip install {uv_args}--python {shlex.quote(python_bin)} "
        "'agent-client-protocol>=0.8.1' 'trio-asyncio>=0.15.0'\n"
    )


def _remote_npm_registry_probe_command(
    npm_package: str,
    *,
    npmrc_path: str | None = None,
) -> str:
    npm_env = f"export NPM_CONFIG_USERCONFIG={shlex.quote(npmrc_path)}\n" if npmrc_path else ""
    return f"set -euo pipefail\n{npm_env}npm view {shlex.quote(npm_package)} version >/dev/null\n"


def _remote_acp_run_env_exports(*, npmrc_path: str | None = None) -> str:
    exports: list[str] = []
    if npmrc_path is not None:
        exports.append(f"export NPM_CONFIG_USERCONFIG={shlex.quote(npmrc_path)}")
    # TODO(remote-pnpm-preparation): if the ACP adapter path ever switches from
    # npm/npx to pnpm dlx, materialize a remote pnpm rc file here and export the
    # corresponding config env so remote release-age policy stays explicit.
    # TODO(remote-bun-preparation): if the ACP adapter path ever switches from
    # npm/npx to bunx, materialize a remote bunfig.toml here and export BUN_CONFIG
    # (or pass --config) explicitly rather than relying on ambient host config.
    if not exports:
        return ""
    return "".join(f"{line}\n" for line in exports)


def _remote_acp_command(
    runtime: Literal["claude_acp", "codex_acp"],
    *,
    preparation: RemoteRuntimePreparation,
    extra_args: list[str] | None = None,
) -> list[str]:
    cli_name, npm_package = _remote_runtime_cli(runtime)
    base = [cli_name] if preparation.mode == "preinstalled" else ["npx", "-y", npm_package]
    return [*base, *(extra_args or [])]


async def _ensure_remote_acp_uv_preparation(
    workspace: SandboxWorkspaceResource,
    *,
    runtime: Literal["claude_acp", "codex_acp"],
    cwd: str,
    preparation: RemoteRuntimePreparation,
    sample_id: str | None = None,
) -> None:
    _cli_name, npm_package = _remote_runtime_cli(runtime)
    marker_dir = f"/tmp/rollouts-external-runtime/{runtime}"
    marker_path = f"{marker_dir}/bootstrap-ready"
    await _workspace_exec(workspace, f"mkdir -p {marker_dir}", cwd=cwd, timeout=30.0)

    marker_result = await _workspace_exec(
        workspace,
        f"test -f {marker_path}",
        cwd=cwd,
        timeout=10.0,
    )
    if marker_result.returncode == 0:
        _emit_remote_runtime_stage(
            runtime=runtime,
            sample_id=sample_id,
            stage="prepare_cached",
            status="success",
        )
        return

    uv_config_path: str | None = None
    npmrc_path: str | None = None
    if preparation.uv_config_toml is not None:
        uv_config_path = f"{marker_dir}/uv.toml"
        await workspace.write_file(uv_config_path, preparation.uv_config_toml.encode("utf-8"))
    if preparation.npmrc is not None:
        npmrc_path = f"{marker_dir}/.npmrc"
        await workspace.write_file(npmrc_path, preparation.npmrc.encode("utf-8"))

    stages = (
        ("node_runtime", _remote_acp_ensure_node_command(), 5 * 60),
        ("uv_runtime", _remote_acp_ensure_uv_command(), 5 * 60),
        (
            "python_deps",
            _remote_acp_uv_prepare_command(
                runtime,
                uv_config_path=uv_config_path,
                npmrc_path=npmrc_path,
            ),
            10 * 60,
        ),
        (
            "npm_registry",
            _remote_npm_registry_probe_command(npm_package, npmrc_path=npmrc_path),
            60.0,
        ),
        ("mark_ready", f"touch {marker_path}", 10.0),
    )

    for stage_name, command, timeout in stages:
        _emit_remote_runtime_stage(
            runtime=runtime,
            sample_id=sample_id,
            stage=stage_name,
            status="start",
        )
        result = await _workspace_exec(
            workspace,
            command,
            cwd=cwd,
            timeout=timeout,
        )
        if result.returncode != 0:
            _emit_remote_runtime_stage(
                runtime=runtime,
                sample_id=sample_id,
                stage=stage_name,
                status="error",
            )
            raise RuntimeError(
                f"Failed to prepare remote {runtime} during {stage_name}: "
                f"{result.stderr or result.stdout}"
            )
        _emit_remote_runtime_stage(
            runtime=runtime,
            sample_id=sample_id,
            stage=stage_name,
            status="success",
        )


async def _ensure_remote_acp_preinstalled(
    workspace: SandboxWorkspaceResource,
    *,
    runtime: Literal["claude_acp", "codex_acp"],
    cwd: str,
) -> None:
    result = await _workspace_exec(
        workspace,
        _remote_acp_validate_command(runtime),
        cwd=cwd,
        timeout=60.0,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Remote {runtime} preinstalled runtime is incomplete: {result.stderr or result.stdout}"
        )


async def _prepare_remote_acp_runtime(
    workspace: SandboxWorkspaceResource,
    *,
    runtime: Literal["claude_acp", "codex_acp"],
    cwd: str,
    preparation: RemoteRuntimePreparation,
    sample_id: str | None = None,
) -> None:
    if preparation.mode == "preinstalled":
        await _ensure_remote_acp_preinstalled(workspace, runtime=runtime, cwd=cwd)
        return
    if preparation.mode == "uv":
        await _ensure_remote_acp_uv_preparation(
            workspace,
            runtime=runtime,
            cwd=cwd,
            preparation=preparation,
            sample_id=sample_id,
        )
        return
    raise AssertionError(f"Unhandled remote ACP preparation mode: {preparation.mode!r}")


def _assistant_block_from_text(idx: int, text: str) -> tuple[int, dict[str, Any]]:
    return idx, {"type": "text", "text": text}


def _assistant_block_from_thinking(idx: int, text: str) -> tuple[int, dict[str, Any]]:
    return idx, {"type": "thinking", "thinking": text}


def _assistant_block_from_tool_call(
    idx: int,
    *,
    tool_call_id: str,
    name: str,
    args: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    return idx, {"type": "toolCall", "id": tool_call_id, "name": name, "arguments": args}


def _trajectory_from_remote_acp_payload(payload: dict[str, Any]) -> Trajectory:
    messages: list[Message] = []
    completed_blocks: list[tuple[int, dict[str, Any]]] = []

    def flush_assistant() -> None:
        nonlocal completed_blocks
        if not completed_blocks:
            return
        completed_blocks.sort(key=lambda item: item[0])
        messages.append(Message(role="assistant", content=[block for _, block in completed_blocks]))
        completed_blocks = []

    for entry in payload.get("events", []):
        event_type = entry.get("event")
        if event_type == "text":
            completed_blocks.append(
                _assistant_block_from_text(int(entry["content_index"]), str(entry["content"]))
            )
            continue
        if event_type == "thinking":
            completed_blocks.append(
                _assistant_block_from_thinking(int(entry["content_index"]), str(entry["content"]))
            )
            continue
        if event_type == "tool_call":
            completed_blocks.append(
                _assistant_block_from_tool_call(
                    int(entry["content_index"]),
                    tool_call_id=str(entry["tool_call_id"]),
                    name=str(entry["tool_name"]),
                    args=dict(entry.get("args", {})),
                )
            )
            continue
        if event_type == "tool_result":
            flush_assistant()
            messages.append(
                Message(
                    role="tool",
                    content=str(entry.get("content", "")),
                    tool_call_id=str(entry["tool_call_id"]),
                )
            )
            continue

    flush_assistant()
    return Trajectory(messages=messages)


async def _run_remote_acp_runtime(
    *,
    runtime: Literal["claude_acp", "codex_acp"],
    prompt: str,
    sample_id: str,
    workspace: SandboxWorkspaceResource,
    cwd: str,
    run_config: Any | None,
    command: list[str],
    timeout_seconds: float,
    preparation: RemoteRuntimePreparation,
) -> ExternalAttemptArtifact:
    del run_config
    await _prepare_remote_acp_runtime(
        workspace,
        runtime=runtime,
        cwd=cwd,
        preparation=preparation,
        sample_id=sample_id,
    )

    state_dir = _remote_runtime_state_dir(runtime, sample_id)
    prompt_path = f"{state_dir}/prompt.txt"
    env_path = f"{state_dir}/env.json"
    output_path = f"{state_dir}/result.json"
    python_bin = _remote_acp_python(runtime)
    npmrc_path = f"/tmp/rollouts-external-runtime/{runtime}/.npmrc"
    codex_home_path: str | None = None
    await _workspace_exec(workspace, f"mkdir -p {state_dir}", cwd=cwd, timeout=30.0)
    await workspace.write_file(prompt_path, prompt.encode("utf-8"))
    env_payload = _remote_runtime_env(runtime)
    if runtime == "codex_acp":
        codex_home_path = f"{state_dir}/codex-home"
        auth_path = f"{codex_home_path}/auth.json"
        await _workspace_exec(workspace, f"mkdir -p {codex_home_path}", cwd=cwd, timeout=30.0)
        await workspace.write_file(
            auth_path,
            json.dumps(_remote_codex_acp_auth_payload(str(env_payload["OPENAI_API_KEY"]))).encode(
                "utf-8"
            ),
        )
        env_payload["CODEX_HOME"] = codex_home_path
    await workspace.write_file(
        env_path,
        json.dumps(env_payload).encode("utf-8"),
    )

    runner = (
        "cat <<'PY' > "
        f"{state_dir}/run_remote_acp.py\n"
        "from __future__ import annotations\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import trio\n"
        "import trio_asyncio\n"
        "from acp import InitializeResponse, PromptResponse, spawn_agent_process\n"
        "from acp.schema import ClientCapabilities, Implementation, TextContentBlock\n"
        "class Bridge:\n"
        "    def __init__(self):\n"
        "        self.events = []\n"
        "        self._content_index = 0\n"
        "        self._pending = {}\n"
        "    def _next_idx(self):\n"
        "        idx = self._content_index\n"
        "        self._content_index += 1\n"
        "        return idx\n"
        "    def on_connect(self, conn):\n"
        "        return None\n"
        "    async def request_permission(self, options, session_id, tool_call, **kwargs):\n"
        "        del session_id, tool_call, kwargs\n"
        "        from acp import RequestPermissionResponse\n"
        "        from acp.schema import AllowedOutcome, DeniedOutcome\n"
        "        allowed = next((opt for opt in options if getattr(opt, 'kind', None) in {'allow_once', 'allow_always'}), None)\n"
        "        if allowed is None:\n"
        "            return RequestPermissionResponse(outcome=DeniedOutcome(outcome='cancelled'))\n"
        "        return RequestPermissionResponse(outcome=AllowedOutcome(optionId=allowed.optionId, outcome='selected'))\n"
        "    async def session_update(self, session_id, update, **kwargs):\n"
        "        del session_id, kwargs\n"
        "        update_type = getattr(update, 'sessionUpdate', None)\n"
        "        if update_type == 'agent_message_chunk':\n"
        "            text = self._content_to_text(getattr(update, 'content', None))\n"
        "            if text:\n"
        "                self.events.append({'event': 'text', 'content_index': self._next_idx(), 'content': text})\n"
        "            return\n"
        "        if update_type == 'agent_thought_chunk':\n"
        "            text = self._content_to_text(getattr(update, 'content', None))\n"
        "            if text:\n"
        "                self.events.append({'event': 'thinking', 'content_index': self._next_idx(), 'content': text})\n"
        "            return\n"
        "        if update_type not in {'tool_call', 'tool_call_update'}:\n"
        "            return\n"
        "        tool_call_id = getattr(update, 'toolCallId', None)\n"
        "        if not tool_call_id:\n"
        "            return\n"
        "        pending = self._pending.get(tool_call_id)\n"
        "        name = self._tool_name(update)\n"
        "        args = self._json_safe(getattr(update, 'rawInput', None))\n"
        "        if not isinstance(args, dict):\n"
        "            args = {'raw_input': args}\n"
        "        if pending is None:\n"
        "            pending = {'content_index': self._next_idx(), 'tool_name': name, 'args': args}\n"
        "            self._pending[tool_call_id] = pending\n"
        "        else:\n"
        "            pending['tool_name'] = name or pending['tool_name']\n"
        "            pending['args'] = args or pending['args']\n"
        "        status = getattr(update, 'status', None)\n"
        "        if status not in {'completed', 'failed'}:\n"
        "            return\n"
        "        self.events.append({'event': 'tool_call', 'content_index': pending['content_index'], 'tool_call_id': tool_call_id, 'tool_name': pending['tool_name'], 'args': pending['args']})\n"
        "        raw_output = self._json_safe(getattr(update, 'rawOutput', None))\n"
        "        content = self._json_safe(getattr(update, 'content', None))\n"
        "        result = self._stringify(raw_output) if raw_output is not None else self._stringify(content)\n"
        "        self.events.append({'event': 'tool_result', 'tool_call_id': tool_call_id, 'content': result, 'is_error': status == 'failed'})\n"
        "        self._pending.pop(tool_call_id, None)\n"
        "    def _tool_name(self, update):\n"
        "        title = getattr(update, 'title', None)\n"
        "        if title:\n"
        "            return str(title)\n"
        "        kind = getattr(update, 'kind', None)\n"
        "        if kind:\n"
        "            return str(kind)\n"
        "        return 'tool'\n"
        "    def _json_safe(self, value):\n"
        "        if isinstance(value, list):\n"
        "            return [self._json_safe(item) for item in value]\n"
        "        if isinstance(value, tuple):\n"
        "            return [self._json_safe(item) for item in value]\n"
        "        if isinstance(value, dict):\n"
        "            return {str(k): self._json_safe(v) for k, v in value.items() if not (k == 'field_meta' and v is None)}\n"
        "        if hasattr(value, 'model_dump'):\n"
        "            return self._json_safe(value.model_dump())\n"
        "        if hasattr(value, '__dict__') and not isinstance(value, (str, bytes, bytearray)):\n"
        "            try:\n"
        "                return self._json_safe(vars(value))\n"
        "            except TypeError:\n"
        "                pass\n"
        "        try:\n"
        "            json.dumps(value)\n"
        "            return value\n"
        "        except TypeError:\n"
        "            return repr(value)\n"
        "    def _content_to_text(self, content):\n"
        "        if content is None:\n"
        "            return ''\n"
        "        if isinstance(content, list):\n"
        "            return '\\n'.join(filter(None, (self._content_to_text(item) for item in content)))\n"
        "        block_type = getattr(content, 'type', None)\n"
        "        if block_type == 'text':\n"
        "            return str(getattr(content, 'text', ''))\n"
        "        return json.dumps(self._json_safe(content), indent=2)\n"
        "    def _stringify(self, value):\n"
        "        if isinstance(value, str):\n"
        "            return value\n"
        "        try:\n"
        "            return json.dumps(value, indent=2)\n"
        "        except TypeError:\n"
        "            return repr(value)\n"
        "async def main():\n"
        f"    prompt = Path({prompt_path!r}).read_text()\n"
        f"    env = dict(os.environ)\n"
        f"    env.update(json.loads(Path({env_path!r}).read_text()))\n"
        f"    if Path({npmrc_path!r}).exists():\n"
        f"        env['NPM_CONFIG_USERCONFIG'] = {npmrc_path!r}\n"
        f"    command = {command!r}\n"
        "    async def run_once_asyncio():\n"
        "        bridge = Bridge()\n"
        "        async with spawn_agent_process(bridge, command[0], *command[1:], env=env, cwd=os.getcwd()) as (conn, process):\n"
        "            init = await conn.initialize(protocol_version=1, client_capabilities=ClientCapabilities(terminal=False), client_info=Implementation(name='rollouts-remote', version='0.1.0'))\n"
        "            if not isinstance(init, InitializeResponse):\n"
        "                raise RuntimeError(f'unexpected initialize response: {init!r}')\n"
        "            session = await conn.new_session(cwd=os.getcwd(), mcp_servers=[])\n"
        "            response = await conn.prompt([TextContentBlock(type='text', text=prompt)], session_id=session.sessionId)\n"
        "            if not isinstance(response, PromptResponse):\n"
        "                raise RuntimeError(f'unexpected prompt response: {response!r}')\n"
        "            if process.returncode is None:\n"
        "                process.terminate()\n"
        "                await process.wait()\n"
        "            return bridge, response, process.returncode\n"
        "    async with trio_asyncio.open_loop():\n"
        "        bridge, response, returncode = await trio_asyncio.aio_as_trio(run_once_asyncio)()\n"
        "    payload = {'events': bridge.events, 'stop_reason': response.stopReason, 'returncode': returncode}\n"
        f"    Path({output_path!r}).write_text(json.dumps(payload), encoding='utf-8')\n"
        "trio.run(main)\n"
        "PY\n"
        f"{_remote_acp_run_env_exports(npmrc_path=npmrc_path)}"
        f"{python_bin} {state_dir}/run_remote_acp.py\n"
    )
    result = await _workspace_exec(
        workspace,
        runner,
        cwd=cwd,
        timeout=timeout_seconds,
    )
    output_result = await _workspace_exec(
        workspace,
        f"cat {output_path}",
        cwd=cwd,
        timeout=30.0,
    )
    await _workspace_exec(workspace, f"rm -rf {state_dir}", cwd=cwd, timeout=30.0)
    if result.returncode != 0:
        raise RuntimeError(
            f"Remote {runtime} failed before producing trajectory output: "
            f"{result.stderr or result.stdout}"
        )
    payload = json.loads(output_result.stdout)
    trajectory = _trajectory_from_remote_acp_payload(payload)
    metadata: dict[str, Any] = {
        "runtime": runtime,
        "driver": runtime,
        "cwd": cwd,
        "remote_execution": True,
        "source": "remote_acp_buffered",
        "stop_reason": payload.get("stop_reason"),
    }
    if payload.get("returncode") not in (None, 0):
        metadata["remote_returncode"] = payload["returncode"]
    return ExternalAttemptArtifact(trajectory=trajectory, metadata=metadata)


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
    result = await _workspace_exec(
        workspace,
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


# TODO(remote-external-runtime-streaming): This function is the fallback for
# the buffered remote execution path. Once session-file polling lands, this
# should only be called when polling fails to find the session file (e.g. the
# CLI crashed before writing anything). The primary path should be live polling.
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


def _remote_session_file_discovery_script(runtime: str, agent_home: str, cwd: str) -> str:
    """Return a shell snippet that prints the session file path to stdout.

    For claude_code: looks in ~/.claude/projects/<escaped-cwd>/ for the newest .jsonl.
    For codex: looks in ~/.codex/sessions/YYYY/MM/DD/ for the newest rollout-*.jsonl.
    Prints nothing (exits 0) if not found yet.
    """
    if runtime == "claude_code":
        # Claude Code escapes cwd by replacing / with -
        # e.g. /workspace -> -workspace
        # We use Python inline to avoid bash quoting hell with arbitrary cwd paths.
        return (
            f'python3 -c "\n'
            f"import os, pathlib\n"
            f"cwd = {cwd!r}\n"
            f"escaped = cwd.replace('/', '-')\n"
            f"projects = pathlib.Path({agent_home!r}) / '.claude' / 'projects' / escaped\n"
            f"if not projects.exists():\n"
            f"    exit(0)\n"
            f"files = sorted(projects.glob('*.jsonl'), key=lambda p: p.stat().st_mtime)\n"
            f"if files:\n"
            f"    print(files[-1])\n"
            f'"'
        )
    if runtime == "codex":
        return (
            f'python3 -c "\n'
            f"import pathlib\n"
            f"sessions = pathlib.Path({agent_home!r}) / '.codex' / 'sessions'\n"
            f"if not sessions.exists():\n"
            f"    exit(0)\n"
            f"files = sorted(sessions.rglob('rollout-*.jsonl'), key=lambda p: p.stat().st_mtime)\n"
            f"if files:\n"
            f"    print(files[-1])\n"
            f'"'
        )
    raise ValueError(f"Unsupported runtime: {runtime}")


async def _run_remote_external_runtime_session_file(
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
    """Launch the CLI in the background and poll its native session file for progress.

    Gives live per-line visibility into the run as the agent writes its session file,
    rather than waiting for the process to exit before seeing any output.
    """
    from ..drivers.session_adapter import claude_message_to_rollouts, codex_message_to_rollouts

    await _ensure_remote_runtime_bootstrap(workspace, runtime=runtime, cwd=cwd)

    state_dir = _remote_runtime_state_dir(runtime, sample_id)
    prompt_path = f"{state_dir}/prompt.txt"
    env_path = f"{state_dir}/env.json"
    pid_path = f"{state_dir}/agent.pid"
    stdout_path = f"{state_dir}/agent.stdout"
    stderr_path = f"{state_dir}/agent.stderr"

    await _workspace_exec(workspace, f"mkdir -p {state_dir}", cwd=cwd, timeout=30.0)
    await workspace.write_file(prompt_path, prompt.encode("utf-8"))
    await workspace.write_file(
        env_path,
        json.dumps(_remote_runtime_env(runtime)).encode("utf-8"),
    )
    await _workspace_exec(
        workspace,
        f"chown -R {REMOTE_AGENT_USER}:{REMOTE_AGENT_USER} {cwd} {state_dir}",
        cwd=cwd,
        timeout=30.0,
    )

    # Get the agent user's home directory
    home_result = await _workspace_exec(
        workspace,
        f"getent passwd {REMOTE_AGENT_USER} | cut -d: -f6",
        cwd=cwd,
        timeout=10.0,
    )
    agent_home = home_result.stdout.strip() or f"/home/{REMOTE_AGENT_USER}"

    # Launch CLI in background as agent user, capturing stdout/stderr to files.
    # Build the inner command string that su -c will execute:
    #   ENV=val ... HOME=... <cli> <args> <prompt> >stdout 2>stderr & echo $!
    env_export = " ".join(f"{k}={shlex.quote(v)}" for k, v in _remote_runtime_env(runtime).items())
    inner_cmd = (
        env_export
        + f" HOME={shlex.quote(agent_home)}"
        + " "
        + shlex.join(command + [prompt])
        + f" >{stdout_path} 2>{stderr_path} & echo $!"
    )
    launch_script = f"su -s /bin/bash {REMOTE_AGENT_USER} -c {shlex.quote(inner_cmd)} >{pid_path}"
    launch_result = await _workspace_exec(workspace, launch_script, cwd=cwd, timeout=30.0)
    if launch_result.returncode != 0:
        raise RuntimeError(
            f"Failed to launch remote {runtime}: {launch_result.stderr or launch_result.stdout}"
        )

    # Poll for session file and read lines as they appear
    messages: list[Any] = []
    session_file: str | None = None
    session_id: str | None = None
    poll_interval = 2.0
    elapsed = 0.0
    file_offset = 0  # bytes consumed so far
    assistant_turn = 0

    discovery_script = _remote_session_file_discovery_script(runtime, agent_home, cwd)

    while elapsed < timeout_seconds:
        await trio.sleep(poll_interval)
        elapsed += poll_interval

        # Discover session file on first appearance
        if session_file is None:
            disc = await _workspace_exec(workspace, discovery_script, cwd=cwd, timeout=10.0)
            path = disc.stdout.strip()
            if path:
                session_file = path
                _event_logger.info(
                    "session_file_found",
                    extra={"runtime": runtime, "sample_id": sample_id, "path": path},
                )

        # Read new bytes from session file
        if session_file is not None:
            read_result = await _workspace_exec(
                workspace,
                f"tail -c +{file_offset + 1} {session_file}",
                cwd=cwd,
                timeout=10.0,
            )
            new_bytes = read_result.stdout
            if new_bytes:
                file_offset += len(new_bytes.encode("utf-8", errors="replace"))
                for raw_line in new_bytes.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if runtime == "claude_code":
                        # Session file uses claude_message_to_rollouts format
                        if session_id is None:
                            session_id = entry.get("sessionId")
                        msg = claude_message_to_rollouts(entry)
                        if msg is not None:
                            messages.append(msg)
                            if msg.role == "assistant":
                                _event_logger.info(
                                    "turn",
                                    extra={
                                        "sample_id": sample_id,
                                        "turn": assistant_turn,
                                        "status": "streaming...",
                                    },
                                )
                                assistant_turn += 1
                            elif msg.role == "tool":
                                _event_logger.info(
                                    "turn",
                                    extra={
                                        "sample_id": sample_id,
                                        "turn": assistant_turn,
                                        "status": "tool done",
                                    },
                                )
                    else:
                        # Codex session file uses _CodexEventParser format
                        if session_id is None and entry.get("type") == "session_meta":
                            session_id = entry.get("payload", {}).get("id")
                        msg = codex_message_to_rollouts(entry)
                        if msg is not None:
                            messages.append(msg)
                            if msg.role == "assistant":
                                _event_logger.info(
                                    "turn",
                                    extra={
                                        "sample_id": sample_id,
                                        "turn": assistant_turn,
                                        "status": "streaming...",
                                    },
                                )
                                assistant_turn += 1

        # Check if background process has exited
        alive_result = await _workspace_exec(
            workspace,
            f"kill -0 $(cat {pid_path} 2>/dev/null) 2>/dev/null && echo alive || echo dead",
            cwd=cwd,
            timeout=10.0,
        )
        if alive_result.stdout.strip() == "dead":
            # One final read to catch any trailing lines written before exit
            if session_file is not None:
                final_result = await _workspace_exec(
                    workspace,
                    f"tail -c +{file_offset + 1} {session_file}",
                    cwd=cwd,
                    timeout=10.0,
                )
                new_bytes = final_result.stdout
                if new_bytes:
                    for raw_line in new_bytes.splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if runtime == "claude_code":
                            if session_id is None:
                                session_id = entry.get("sessionId")
                            msg = claude_message_to_rollouts(entry)
                            if msg is not None:
                                messages.append(msg)
                        else:
                            if session_id is None and entry.get("type") == "session_meta":
                                session_id = entry.get("payload", {}).get("id")
                            msg = codex_message_to_rollouts(entry)
                            if msg is not None:
                                messages.append(msg)
            break
    else:
        raise RuntimeError(
            f"Remote {runtime} timed out after {timeout_seconds:.0f}s "
            f"(session_file={session_file!r})"
        )

    stdout_result = await _workspace_exec(
        workspace, f"cat {stdout_path} 2>/dev/null || true", cwd=cwd, timeout=10.0
    )
    stderr_result = await _workspace_exec(
        workspace, f"cat {stderr_path} 2>/dev/null || true", cwd=cwd, timeout=10.0
    )
    await _workspace_exec(workspace, f"rm -rf {state_dir}", cwd=cwd, timeout=30.0)

    has_agent_output = any(
        isinstance(msg, Message) and msg.role in {"assistant", "tool"} for msg in messages
    )
    if not has_agent_output:
        trajectory, fallback_session_id = await _build_trajectory_from_remote_jsonl(
            runtime=runtime,
            raw_output=stdout_result.stdout,
            sample_id=sample_id,
            run_config=run_config,
            driver_name="claude" if runtime == "claude_code" else "codex",
        )
        fallback_has_agent_output = any(
            isinstance(msg, Message) and msg.role in {"assistant", "tool"}
            for msg in trajectory.messages
        )
        if fallback_has_agent_output:
            return ExternalAttemptArtifact(
                trajectory=trajectory,
                metadata={
                    "runtime": runtime,
                    "driver": "claude" if runtime == "claude_code" else "codex",
                    "cwd": cwd,
                    "session_id": fallback_session_id,
                    "remote_execution": True,
                    "source": "stream_json_fallback",
                },
            )
        raise RuntimeError(
            f"Remote {runtime} exited without producing agent output. "
            f"stderr: {stderr_result.stdout[-2000:]}"
        )

    trajectory = Trajectory(messages=list(messages))
    metadata: dict[str, Any] = {
        "runtime": runtime,
        "driver": "claude" if runtime == "claude_code" else "codex",
        "cwd": cwd,
        "session_id": session_id,
        "remote_execution": True,
        "source": "session_file",
    }
    return ExternalAttemptArtifact(trajectory=trajectory, metadata=metadata)


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
    await _workspace_exec(
        workspace,
        f"mkdir -p {state_dir}",
        cwd=cwd,
        timeout=30.0,
    )
    await workspace.write_file(prompt_path, prompt.encode("utf-8"))
    await workspace.write_file(
        env_path,
        json.dumps(_remote_runtime_env(runtime)).encode("utf-8"),
    )
    await _workspace_exec(
        workspace,
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
    result = await _workspace_exec(
        workspace,
        runner,
        cwd=cwd,
        timeout=timeout_seconds,
    )
    await _workspace_exec(
        workspace,
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
    source: Literal["stream_json", "session_file"] = "stream_json",
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
    if source == "session_file":
        artifact = await _run_remote_external_runtime_session_file(
            runtime="claude_code",
            prompt=prompt,
            sample_id=sample_id,
            workspace=workspace,
            cwd=cwd,
            run_config=run_config,
            command=command,
            timeout_seconds=timeout_seconds,
        )
    else:
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
    source: Literal["stream_json", "session_file"] = "stream_json",
) -> ExternalAttemptArtifact:
    del sample_data
    command = [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--model",
        model,
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    if source == "session_file":
        artifact = await _run_remote_external_runtime_session_file(
            runtime="codex",
            prompt=prompt,
            sample_id=sample_id,
            workspace=workspace,
            cwd=cwd,
            run_config=run_config,
            command=command,
            timeout_seconds=timeout_seconds,
        )
    else:
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


async def trajectory_from_remote_claude_acp(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    workspace: SandboxWorkspaceResource,
    cwd: str,
    run_config: Any | None = None,
    *,
    model: str = "claude-agent-acp",
    timeout_seconds: float = 600.0,
    preparation: RemoteRuntimePreparation = RemoteRuntimePreparation(mode="uv"),
) -> ExternalAttemptArtifact:
    del sample_data
    artifact = await _run_remote_acp_runtime(
        runtime="claude_acp",
        prompt=prompt,
        sample_id=sample_id,
        workspace=workspace,
        cwd=cwd,
        run_config=run_config,
        command=_remote_acp_command("claude_acp", preparation=preparation),
        timeout_seconds=timeout_seconds,
        preparation=preparation,
    )
    artifact.metadata["model"] = model
    return artifact


async def trajectory_from_remote_codex_acp(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    workspace: SandboxWorkspaceResource,
    cwd: str,
    run_config: Any | None = None,
    *,
    model: str = "codex-acp",
    timeout_seconds: float = 600.0,
    preparation: RemoteRuntimePreparation = RemoteRuntimePreparation(mode="uv"),
) -> ExternalAttemptArtifact:
    del sample_data
    artifact = await _run_remote_acp_runtime(
        runtime="codex_acp",
        prompt=prompt,
        sample_id=sample_id,
        workspace=workspace,
        cwd=cwd,
        run_config=run_config,
        command=_remote_acp_command(
            "codex_acp",
            preparation=preparation,
            extra_args=[
                "-c",
                'forced_login_method="api"',
                "-c",
                'preferred_auth_method="apikey"',
                "-c",
                'approval_policy="never"',
                "-c",
                'sandbox_mode="workspace-write"',
            ],
        ),
        timeout_seconds=timeout_seconds,
        preparation=preparation,
    )
    artifact.metadata["model"] = model
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


async def trajectory_from_claude_acp(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    cwd: Path,
    run_config: Any | None = None,
    model: str = "claude-agent-acp",
) -> ExternalAttemptArtifact:
    del sample_data
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
            "cwd": str(cwd),
        },
    )


async def trajectory_from_codex_acp(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    cwd: Path,
    run_config: Any | None = None,
    model: str = "codex-acp",
) -> ExternalAttemptArtifact:
    del sample_data
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
            "cwd": str(cwd),
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


def _mini_swe_agent_output_to_trajectory(payload: dict[str, Any]) -> Trajectory:
    messages: list[Message] = []
    for raw_msg in payload.get("messages", []):
        if not isinstance(raw_msg, dict):
            continue
        raw_role = str(raw_msg.get("role") or "").lower()
        if raw_role == "exit":
            continue
        role = raw_role if raw_role in {"system", "user", "assistant", "tool"} else "assistant"
        raw_content = raw_msg.get("content")
        if isinstance(raw_content, str) or raw_content is None:
            content = raw_content
        else:
            content = json.dumps(raw_content, ensure_ascii=False)
        details = {k: v for k, v in raw_msg.items() if k not in {"role", "content", "tool_call_id"}}
        messages.append(
            Message(
                role=role,
                content=content,
                tool_call_id=raw_msg.get("tool_call_id"),
                details=details or None,
            )
        )

    if not messages:
        raise RuntimeError("mini-swe-agent produced no importable messages")
    return Trajectory(messages=messages)


async def trajectory_from_mini_swe_agent(
    prompt: str,
    sample_id: str,
    sample_data: dict[str, Any],
    *,
    cwd: Path | None = None,
    run_config: Any | None = None,
    model: str | None = None,
    timeout_seconds: float = 600.0,
    config_spec: list[str] | None = None,
    agent_class: str | None = None,
    environment_class: str | None = None,
    yolo: bool = True,
) -> ExternalAttemptArtifact:
    del sample_data

    workdir = Path(cwd or Path.cwd()).resolve()
    output_path = workdir / f".mini-swe-agent-{_sample_id_slug(sample_id)}.traj.json"
    if output_path.exists():
        output_path.unlink()

    cli = shutil.which("mini")
    if cli is not None:
        cmd = [cli]
    elif shutil.which("uvx") is not None:
        cmd = ["uvx", "--from", "mini-swe-agent", "mini"]
    else:
        raise RuntimeError("mini-swe-agent CLI not found. Install `mini` or make `uvx` available.")

    cmd.extend(["--task", prompt, "--output", str(output_path)])
    if model is not None:
        cmd.extend(["--model", model])
    if yolo:
        cmd.append("--yolo")
    cmd.append("--exit-immediately")
    for spec in config_spec or ():
        cmd.extend(["--config", spec])
    if agent_class:
        cmd.extend(["--agent-class", agent_class])
    if environment_class:
        cmd.extend(["--environment-class", environment_class])

    env = os.environ.copy()
    env.setdefault("MSWEA_CONFIGURED", "true")
    if model is not None:
        env["MSWEA_MODEL_NAME"] = model
    raw_line_handler = _make_raw_driver_line_handler(run_config, driver="mini_swe_agent")
    progress_emitter = _make_external_progress_emitter(run_config, driver="mini_swe_agent")
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    latest_payload: dict[str, Any] | None = None
    latest_messages = 0

    async def _consume_stream(
        stream: trio.abc.ReceiveStream | None,
        *,
        sink: list[str],
    ) -> None:
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
                if raw_line_handler is not None:
                    await raw_line_handler(line)
        if buffer:
            sink.append(buffer)
            if raw_line_handler is not None:
                await raw_line_handler(buffer)

    async def _poll_output_file() -> None:
        nonlocal latest_payload, latest_messages
        last_mtime_ns: int | None = None
        while True:
            await trio.sleep(1.0)
            if not output_path.is_file():
                continue
            stat = output_path.stat()
            if stat.st_mtime_ns == last_mtime_ns:
                continue
            last_mtime_ns = stat.st_mtime_ns
            try:
                payload = json.loads(output_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue
            latest_payload = payload
            latest_messages = len(payload.get("messages", []))
            if progress_emitter is not None:
                await progress_emitter(
                    "external_artifact_progress",
                    {
                        "trajectory_path": str(output_path),
                        "messages": latest_messages,
                        "bytes": stat.st_size,
                    },
                )

    proc = await trio.lowlevel.open_process(
        cmd,
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    with trio.move_on_after(timeout_seconds) as cancel_scope:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(_consume_stream, proc.stdout, sink=stdout_lines))
            nursery.start_soon(partial(_consume_stream, proc.stderr, sink=stderr_lines))
            nursery.start_soon(_poll_output_file)
            await proc.wait()
            nursery.cancel_scope.cancel()

    if cancel_scope.cancelled_caught:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"mini-swe-agent run timed out after {timeout_seconds:.1f}s")

    stdout_text = "\n".join(stdout_lines).strip()
    stderr_text = "\n".join(stderr_lines).strip()
    combined_output = "\n".join(part for part in (stdout_text, stderr_text) if part).strip()

    if not output_path.is_file():
        raise RuntimeError(
            "mini-swe-agent did not write a trajectory file.\n"
            f"stdout:\n{stdout_text}\n\nstderr:\n{stderr_text}"
        )

    if latest_payload is None:
        payload = json.loads(output_path.read_text())
    else:
        payload = latest_payload
    trajectory = _mini_swe_agent_output_to_trajectory(payload)
    info = payload.get("info", {})
    metadata: dict[str, Any] = {
        "runtime": "mini_swe_agent",
        "driver": "mini_swe_agent",
        "cwd": str(workdir),
        "trajectory_path": str(output_path),
        "exit_status": info.get("exit_status"),
        "submission": info.get("submission"),
    }
    if model is not None:
        metadata["model"] = model
    metadata["returncode"] = proc.returncode
    metadata["message_count"] = latest_messages or len(payload.get("messages", []))
    if combined_output:
        metadata["mini_swe_agent_output"] = combined_output[-8000:]

    status = Status.COMPLETED if proc.returncode == 0 else Status.ABORTED
    return ExternalAttemptArtifact(
        trajectory=trajectory,
        metadata=metadata,
        status=status,
    )


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
    del sample_data

    if api_key_env_var is not None and not os.environ.get(api_key_env_var):
        raise RuntimeError(f"Required environment variable {api_key_env_var} is not set")

    cli = shutil.which("openhands")
    if cli is None:
        raise RuntimeError(
            "OpenHands CLI not found. Install from https://docs.all-hands.dev/usage/installation"
        )

    workdir = Path(cwd or Path.cwd()).resolve()
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
