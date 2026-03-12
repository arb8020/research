from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from ..core import Endpoint, Trajectory
from ..frontends.runner import RunFn


@dataclass(frozen=True)
class DriverBuildRequest:
    driver: str
    endpoint: Endpoint
    cwd: str | None
    working_dir: Path
    trajectory: Trajectory
    session_id: str | None
    cursor_api_key: str | None = None
    autonomous: bool = False


def _prepare_claude_resume_session(
    trajectory: Trajectory,
    session_id: str | None,
    working_dir: Path,
) -> str | None:
    if not session_id or not trajectory.messages:
        return None

    from .session_adapter import get_claude_session_path, write_claude_session

    messages = [message for message in trajectory.messages if message.role != "system"]
    if not messages:
        return None

    claude_resume_session_id = str(uuid.uuid4())
    cwd = str(working_dir.resolve())
    output_path = get_claude_session_path(claude_resume_session_id, cwd)
    write_claude_session(messages, claude_resume_session_id, output_path, cwd)
    return claude_resume_session_id


def _resolve_claude_model(endpoint: Endpoint, autonomous: bool) -> str:
    if autonomous:
        return "sonnet"

    claude_model = "sonnet"
    if endpoint.model and endpoint.provider == "anthropic":
        raw = endpoint.model
        claude_model = raw.split("/", 1)[1] if "/" in raw else raw
    return claude_model


def create_run_fn(request: DriverBuildRequest) -> RunFn | None:
    if request.driver == "sdk":
        return None

    if request.driver == "claude":
        from .run_claude import run_claude

        resume_session_id = _prepare_claude_resume_session(
            request.trajectory,
            request.session_id,
            request.working_dir,
        )
        return partial(
            run_claude,
            model=_resolve_claude_model(request.endpoint, request.autonomous),
            cwd=request.cwd,
            resume_session_id=resume_session_id,
            autonomous=request.autonomous,
        )

    if request.driver == "codex":
        from .run_codex import run_codex

        model = "o3" if request.autonomous else None
        return partial(
            run_codex,
            model=model,
            cwd=request.cwd,
            autonomous=request.autonomous,
        )

    if request.driver == "cursor":
        from .run_cursor import run_cursor

        return partial(
            run_cursor,
            model=None,
            cwd=request.cwd,
            api_key=request.cursor_api_key,
            autonomous=request.autonomous,
        )

    raise ValueError(f"Unknown driver: {request.driver}")
