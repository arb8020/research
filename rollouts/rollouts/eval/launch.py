from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import trio

from ..core import Endpoint, TrajectoryEnvironment, TrajectorySession
from ..core.session import EnvironmentConfig
from ..drivers.session_adapter import (
    claude_session_to_messages,
    codex_session_to_messages,
    find_claude_session,
    read_codex_session_id,
)
from ..dtypes import StopReason, Trajectory
from ..store import FileSessionStore
from ..training.scoring import attach_score
from ..training.types import AttemptResult, ProblemRow, ScoringContext, Status
from .configs import EvalOutputConfig, EvalRunConfig, resolve_eval_run_spec, resolve_eval_task_spec
from .native import _compute_score
from .run import load_tasks_from_module


def _resolve_sample(tasks: list[dict[str, Any]], selector: str) -> tuple[str, dict[str, Any]]:
    if not tasks:
        raise ValueError("Eval config contains no tasks")

    if selector.isdigit():
        index = int(selector)
        if 0 <= index < len(tasks):
            row = tasks[index]
            return _sample_id_for_row(row, index), row

    for index, row in enumerate(tasks):
        sample_id = _sample_id_for_row(row, index)
        candidates = {
            sample_id,
            str(row.get("id", "")),
            str(row.get("problem_id", "")),
            str(row.get("name", "")),
        }
        if selector in candidates:
            return sample_id, row

    raise ValueError(
        f"Could not find sample {selector!r}. "
        "Use a zero-based index or an id/problem_id/name present in the dataset."
    )


def _sample_id_for_row(row: dict[str, Any], index: int) -> str:
    for key in ("id", "problem_id", "name"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return str(index)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            thinking = getattr(block, "thinking", None)
            if isinstance(thinking, str):
                parts.append(thinking)
                continue
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("thinking"), str):
                    parts.append(block["thinking"])
        return "\n".join(part for part in parts if part)
    return str(content) if content is not None else ""


def _build_launch_prompt(config_module: Any, sample_data: dict[str, Any]) -> str:
    run_spec = resolve_eval_run_spec(config_module)
    prepare_messages = run_spec.prepare_messages
    if prepare_messages is None:
        raise ValueError(
            "rollouts eval launch requires prepare_messages(...) so the launcher can "
            "derive the task prompt for external runtimes."
        )

    messages = prepare_messages(sample_data)
    if not isinstance(messages, list) or not messages:
        raise ValueError("prepare_messages(...) must return a non-empty message list")

    rendered: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        if role not in {"system", "user", "developer"}:
            continue
        text = _message_text(content).strip()
        if not text:
            continue
        rendered.append(f"{str(role).upper()}:\n{text}")

    if not rendered:
        raise ValueError("prepare_messages(...) produced no system/user/developer text to launch")

    return "\n\n".join(rendered)


def _build_session_endpoint(runtime: str, runtime_model: str | None) -> Endpoint:
    model = runtime_model or ("claude-code" if runtime == "claude_code" else "codex")
    if "/" not in model:
        provider = "anthropic" if runtime == "claude_code" else "openai"
        model = f"{provider}/{model}"

    return Endpoint(
        model=model,
        base_url=f"cli://{runtime}",
        api_format="external-cli",
    )


def _build_environment_bundle(
    env: Any | None,
    environment_state: dict[str, Any] | None,
) -> TrajectoryEnvironment | None:
    if env is None:
        return None

    env_kind = type(env).__name__
    env_config: dict[str, Any] = {}
    if environment_state is not None:
        env_kind = str(environment_state.get("env_kind", env_kind))
        env_config = dict(environment_state)
    return TrajectoryEnvironment.from_session_parts(
        EnvironmentConfig(type=env_kind, config=env_config),
        environment_state,
    )


def _json_safe_environment_state(environment_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if environment_state is None:
        return None
    try:
        json.dumps(environment_state)
    except TypeError:
        return None
    return environment_state


def _external_agent_args(config_module: Any) -> dict[str, Any]:
    return dict(resolve_eval_run_spec(config_module).external_agent_args)


def _interactive_claude_launch_kwargs(
    config_module: Any,
    *,
    cwd: Path,
    model: str,
) -> dict[str, Any]:
    external_agent_args = _external_agent_args(config_module)
    return {
        **external_agent_args,
        "cwd": cwd,
        "model": model,
    }


def _interactive_codex_launch_kwargs(
    config_module: Any,
    *,
    cwd: Path,
    model: str,
    default_sandbox: str,
) -> dict[str, Any]:
    external_agent_args = _external_agent_args(config_module)
    return {
        **external_agent_args,
        "cwd": cwd,
        "model": model,
        "sandbox": external_agent_args.get("sandbox", default_sandbox),
    }


async def _score_attempt(config_module: Any, env: Any | None, attempt: AttemptResult) -> None:
    eval_task = resolve_eval_task_spec(config_module)
    scorer = eval_task.scorer
    if scorer is None:
        raise ValueError("launch requires an explicit scorer")

    score = await _compute_score(
        attempt,
        scorer=scorer,
        scoring_context=ScoringContext(environment=env),
    )
    attach_score(attempt, score)


async def _initialize_environment_for_launch(env: Any | None, sample_id: str) -> None:
    if env is None:
        return
    initialize = getattr(env, "initialize", None)
    if not callable(initialize):
        return

    initialized = initialize(sample_id)
    if hasattr(initialized, "__await__"):
        await initialized


async def _collect_environment_launch_metadata(
    env: Any | None,
    trajectory: Trajectory,
) -> dict[str, Any]:
    if env is None:
        return {}

    metadata: dict[str, Any] = {}

    get_runtime_metadata = getattr(env, "get_runtime_metadata", None)
    if callable(get_runtime_metadata):
        runtime_metadata = get_runtime_metadata()
        if isinstance(runtime_metadata, dict):
            metadata.update(runtime_metadata)

    finalize_launch = getattr(env, "finalize_launch", None)
    if callable(finalize_launch):
        finalized = finalize_launch(trajectory)
        if hasattr(finalized, "__await__"):
            finalized = await finalized
        if isinstance(finalized, dict):
            metadata.update(finalized)

    return metadata


def _build_attempt(
    *,
    sample_id: str,
    sample_data: dict[str, Any],
    trajectory: Trajectory,
    environment_state: dict[str, Any] | None,
    runtime: str,
    control_mode: str,
    metadata: dict[str, Any] | None = None,
) -> AttemptResult:
    return AttemptResult(
        attempt_id=sample_id,
        problem=ProblemRow(
            problem_id=sample_id,
            payload=dict(sample_data),
            metadata=(
                dict(sample_data.get("metadata", {}))
                if isinstance(sample_data.get("metadata"), dict)
                else {}
            ),
        ),
        trajectory=trajectory,
        environment_state=environment_state,
        status=Status.COMPLETED,
        metadata={
            "runtime": runtime,
            "control_mode": control_mode,
            **(metadata or {}),
        },
    )


def _codex_sessions_dir() -> Path:
    return Path.home() / ".codex" / "sessions"


def _snapshot_codex_sessions() -> dict[Path, float]:
    sessions_dir = _codex_sessions_dir()
    if not sessions_dir.exists():
        return {}
    return {path: path.stat().st_mtime for path in sessions_dir.rglob("*.jsonl") if path.is_file()}


def _codex_session_cwd(session_path: Path) -> str | None:
    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "session_meta":
                continue
            payload = entry.get("payload", {})
            cwd = payload.get("cwd")
            return cwd if isinstance(cwd, str) else None
    return None


def _find_interactive_codex_session(
    before: dict[Path, float],
    cwd: Path,
) -> Path | None:
    sessions_dir = _codex_sessions_dir()
    if not sessions_dir.exists():
        return None

    resolved_cwd = str(cwd.resolve())
    candidates: list[tuple[float, Path]] = []
    for session_path in sessions_dir.rglob("*.jsonl"):
        if not session_path.is_file():
            continue
        stat = session_path.stat()
        if session_path not in before or stat.st_mtime > before[session_path]:
            session_cwd = _codex_session_cwd(session_path)
            if session_cwd == resolved_cwd:
                candidates.append((stat.st_mtime, session_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


async def _run_attached_process(cmd: list[str], cwd: Path) -> int:
    completed = await trio.to_thread.run_sync(
        lambda: subprocess.run(
            cmd,
            cwd=str(cwd),
            env=os.environ.copy(),
            check=False,
        )
    )
    return completed.returncode


async def _launch_interactive_claude(
    *,
    prompt: str,
    cwd: Path,
    model: str,
    allowed_tools: list[str] | None = None,
    system_prompt: str | None = None,
    dangerously_skip_permissions: bool = True,
) -> tuple[Trajectory, dict[str, Any], StopReason]:
    claude_bin = shutil.which("claude")
    if claude_bin is None:
        raise RuntimeError(
            "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
        )

    session_id = str(uuid.uuid4())
    cmd = [
        claude_bin,
        "--model",
        model,
        "--session-id",
        session_id,
    ]
    if dangerously_skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    if system_prompt is not None:
        cmd.extend(["--system-prompt", system_prompt])
    if allowed_tools:
        cmd.extend(["--allowedTools", ",".join(allowed_tools)])
    cmd.append(prompt)

    print(
        f"Launching Claude Code interactively in {cwd}.\n"
        "Exit Claude Code when you want rollouts to import and score the session.",
        file=sys.stderr,
    )
    returncode = await _run_attached_process(cmd, cwd)

    session_path = find_claude_session(session_id)
    if session_path is None:
        raise RuntimeError(
            "Claude Code interactive run finished, but rollouts could not find the session file "
            f"for session_id={session_id}."
        )

    messages = claude_session_to_messages(session_path)
    if not messages:
        raise RuntimeError(f"Claude Code session {session_id} produced no importable messages.")

    stop_reason = StopReason.TASK_COMPLETED if returncode == 0 else StopReason.ABORTED
    trajectory = Trajectory(messages=messages)
    metadata = {
        "runtime": "claude_code",
        "driver": "interactive-cli",
        "model": model,
        "cwd": str(cwd),
        "session_id": session_id,
        "session_path": str(session_path),
        "returncode": returncode,
    }
    if allowed_tools:
        metadata["allowed_tools"] = list(allowed_tools)
    if system_prompt is not None:
        metadata["system_prompt"] = system_prompt
    return trajectory, metadata, stop_reason


async def _launch_interactive_codex(
    *,
    prompt: str,
    cwd: Path,
    model: str,
    sandbox: str = "workspace-write",
    ask_for_approval: str = "never",
) -> tuple[Trajectory, dict[str, Any], StopReason]:
    codex_bin = shutil.which("codex")
    if codex_bin is None:
        raise RuntimeError("Codex CLI not found. Install from: https://github.com/openai/codex")

    before = _snapshot_codex_sessions()
    cmd = [
        codex_bin,
        "--model",
        model,
        "--sandbox",
        sandbox,
        "--ask-for-approval",
        ask_for_approval,
        prompt,
    ]

    print(
        f"Launching Codex interactively in {cwd}.\n"
        "Exit Codex when you want rollouts to import and score the session.",
        file=sys.stderr,
    )
    returncode = await _run_attached_process(cmd, cwd)

    session_path = _find_interactive_codex_session(before, cwd)
    if session_path is None:
        raise RuntimeError(
            "Codex interactive run finished, but rollouts could not identify the session file "
            f"for cwd={cwd}."
        )

    messages = codex_session_to_messages(session_path)
    if not messages:
        raise RuntimeError(f"Codex session at {session_path} produced no importable messages.")

    session_id = read_codex_session_id(session_path)
    stop_reason = StopReason.TASK_COMPLETED if returncode == 0 else StopReason.ABORTED
    trajectory = Trajectory(messages=messages)
    metadata = {
        "runtime": "codex",
        "driver": "interactive-cli",
        "model": model,
        "cwd": str(cwd),
        "sandbox": sandbox,
        "session_id": session_id,
        "session_path": str(session_path),
        "returncode": returncode,
    }
    return trajectory, metadata, stop_reason


async def launch_sample(
    *,
    config_module: Any,
    config_path: Path,
    sample_selector: str,
    runtime: str,
    run_config: EvalRunConfig,
    output_config: EvalOutputConfig,
    model: str | None = None,
    session_store: FileSessionStore | None = None,
) -> tuple[AttemptResult, str]:
    if runtime not in {"claude_code", "codex"}:
        raise ValueError(f"Unsupported runtime for eval launch: {runtime}")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError(
            "Interactive eval launch requires a real terminal (stdin/stdout must be TTYs)."
        )

    tasks = load_tasks_from_module(config_module)
    sample_id, sample_data = _resolve_sample(tasks, sample_selector)

    run_spec = resolve_eval_run_spec(config_module)
    environment_factory = run_spec.environment_factory

    env = None
    environment_state = None
    try:
        if environment_factory is not None:
            env_candidate = environment_factory(sample_data)
            if hasattr(env_candidate, "__await__"):
                env = await env_candidate
            else:
                env = env_candidate

        await _initialize_environment_for_launch(env, sample_id)

        if env is not None and hasattr(env, "serialize"):
            environment_state = await env.serialize()
        environment_state = _json_safe_environment_state(environment_state)

        prompt = _build_launch_prompt(config_module, sample_data)

        cwd = Path.cwd()
        if environment_state is not None:
            for key in ("working_dir", "workspace_dir", "cwd"):
                value = environment_state.get(key)
                if isinstance(value, str) and value:
                    cwd = Path(value)
                    break

        runtime_model = model or ("sonnet" if runtime == "claude_code" else "gpt-5.1-codex-mini")
        session_stop_reason = StopReason.TASK_COMPLETED
        if runtime == "claude_code":
            (
                base_trajectory,
                artifact_metadata,
                session_stop_reason,
            ) = await _launch_interactive_claude(
                prompt=prompt,
                **_interactive_claude_launch_kwargs(
                    config_module,
                    cwd=cwd,
                    model=runtime_model,
                ),
            )
        else:
            (
                base_trajectory,
                artifact_metadata,
                session_stop_reason,
            ) = await _launch_interactive_codex(
                prompt=prompt,
                **_interactive_codex_launch_kwargs(
                    config_module,
                    cwd=cwd,
                    model=runtime_model,
                    default_sandbox="workspace-write",
                ),
            )

        now = datetime.now().isoformat()
        endpoint = _build_session_endpoint(runtime, artifact_metadata.get("model"))
        trajectory = replace(
            base_trajectory,
            session=TrajectorySession(
                endpoint=endpoint,
                stop_reason=session_stop_reason,
                created_at=now,
                updated_at=now,
                tags={
                    "launched_from": "eval",
                    "eval_config": str(config_path),
                    "sample_id": sample_id,
                    "runtime": runtime,
                    "control_mode": "interactive",
                },
            ),
            environment=_build_environment_bundle(env, environment_state),
            metadata={
                **artifact_metadata,
                **base_trajectory.metadata,
                "sample_id": sample_id,
                "sample_data": sample_data,
                "eval_name": output_config.experiment_name,
                "eval_config_path": str(config_path),
            },
        )

        env_launch_metadata = await _collect_environment_launch_metadata(env, trajectory)
        if env_launch_metadata:
            trajectory = replace(
                trajectory,
                metadata={
                    **trajectory.metadata,
                    **env_launch_metadata,
                },
            )

        if env is not None and hasattr(env, "serialize"):
            environment_state = await env.serialize()
        environment_state = _json_safe_environment_state(environment_state)

        attempt = _build_attempt(
            sample_id=sample_id,
            sample_data=sample_data,
            trajectory=trajectory,
            environment_state=environment_state,
            runtime=runtime,
            control_mode="interactive",
            metadata={
                **(
                    dict(sample_data.get("metadata", {}))
                    if isinstance(sample_data.get("metadata"), dict)
                    else {}
                ),
                **trajectory.metadata,
            },
        )
        if session_stop_reason is not StopReason.TASK_COMPLETED:
            attempt.status = Status.ABORTED
        await _score_attempt(config_module, env, attempt)

        if session_store is None:
            session_store = FileSessionStore()
        saved_session, err = await session_store.save_trajectory(trajectory)
        if err is not None or saved_session is None:
            raise RuntimeError(err or "Failed to save launched eval session")

        return attempt, saved_session.session_id
    finally:
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                await close()
