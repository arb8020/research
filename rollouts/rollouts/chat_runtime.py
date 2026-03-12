from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trio

from .core import Endpoint, Environment, Trajectory
from .drivers.factory import DriverBuildRequest, create_run_fn
from .frontends import run_interactive
from .frontends.factory import create_interactive_frontend, create_print_frontend
from .frontends.runner import RunnerConfig
from .store import FileSessionStore


@dataclass(frozen=True)
class ChatRunRequest:
    endpoint: Endpoint
    environment: Environment | None
    working_dir: Path
    cwd: str | None
    driver: str
    cursor_api_key: str | None
    session_store: FileSessionStore | None
    frontend: str
    theme: str
    debug: bool
    debug_layout: bool
    stream_json: bool = False
    quiet: bool = False
    confirm_tools: bool = False
    detached: bool = False


def create_session_store(
    *,
    no_session: bool,
    env_name: str,
    environment: Environment | None,
) -> FileSessionStore | None:
    if no_session:
        return None

    if env_name == "tbench" and environment is not None:
        logging_dir = getattr(environment, "logging_dir", None)
        if logging_dir is not None:
            agent_dir = Path(logging_dir) / "agent"
            return FileSessionStore(
                base_dir=agent_dir / "sessions",
                atif_filename="trajectory.json",
                atif_output_path=agent_dir / "trajectory.json",
            )

    return FileSessionStore()


async def cleanup_environment(environment: Environment | None) -> None:
    if environment is None or not hasattr(environment, "cleanup"):
        return
    await environment.cleanup()


def get_tbench_agent_timeout_sec(environment: Environment | None) -> float | None:
    if environment is None:
        return None
    timeout = getattr(environment, "max_agent_timeout_sec", None)
    if timeout is None:
        return None
    return float(timeout)


async def finalize_tbench_run(
    environment: Environment | None,
    *,
    agent_timed_out: bool = False,
) -> int | None:
    if environment is None or not hasattr(environment, "run_tests"):
        return None

    result = await environment.run_tests()
    success = result.success and not agent_timed_out
    failure_reason = "AGENT_TIMEOUT" if agent_timed_out else result.failure_reason

    payload = {
        "score": result.score,
        "success": success,
        "failure_reason": failure_reason,
        "agent_timed_out": agent_timed_out,
    }
    logging_dir = getattr(environment, "logging_dir", None)
    if logging_dir is not None:
        logging_path = Path(logging_dir)
        logging_path.mkdir(parents=True, exist_ok=True)
        (logging_path / "results.json").write_text(json.dumps(payload, indent=2))
        if result.output:
            (logging_path / "test_output.txt").write_text(result.output)

    summary = (
        f"Terminal-Bench result: success={payload['success']} "
        f"score={payload['score']:.3f} reason={payload['failure_reason'] or 'OK'}"
    )
    print(summary, file=sys.stderr)
    return 0 if payload["success"] else 1


async def _run_with_optional_timeout(
    request: ChatRunRequest,
    *,
    trajectory: Trajectory,
    runner_config: RunnerConfig,
    frontend: Any,
) -> tuple[int | None, bool]:
    timeout_sec = get_tbench_agent_timeout_sec(request.environment)
    timed_out = False

    try:
        if timeout_sec is None:
            await run_interactive(
                trajectory,
                request.endpoint,
                frontend=frontend,
                environment=request.environment,
                config=runner_config,
            )
        else:
            with trio.fail_after(timeout_sec):
                await run_interactive(
                    trajectory,
                    request.endpoint,
                    frontend=frontend,
                    environment=request.environment,
                    config=runner_config,
                )
    except trio.TooSlowError:
        timed_out = True
        print(
            f"Terminal-Bench agent timed out after {timeout_sec:.1f}s",
            file=sys.stderr,
        )

    result = await finalize_tbench_run(request.environment, agent_timed_out=timed_out)
    return result, timed_out


async def run_print_mode(
    request: ChatRunRequest,
    *,
    trajectory: Trajectory,
    session_id: str | None,
    bootstrap_input: str | None,
    query: str | None,
) -> int:
    if query == "-":
        query = bootstrap_input or ""
        if not query:
            print("Error: no input from stdin", file=sys.stderr)
            return 1

    frontend = create_print_frontend(
        stream_json=request.stream_json,
        quiet=request.quiet,
        frontend_name=request.frontend,
        environment=request.environment,
        endpoint=request.endpoint,
    )

    run_fn = create_run_fn(
        DriverBuildRequest(
            driver=request.driver,
            endpoint=request.endpoint,
            cwd=request.cwd,
            working_dir=request.working_dir,
            trajectory=trajectory,
            session_id=session_id,
            cursor_api_key=request.cursor_api_key,
            autonomous=True,
        )
    )

    runner_config = RunnerConfig(
        session_store=request.session_store,
        session_id=session_id,
        bootstrap_input=query,
        single_turn=True,
        run_fn=run_fn,
    )

    try:
        result, _ = await _run_with_optional_timeout(
            request,
            trajectory=trajectory,
            runner_config=runner_config,
            frontend=frontend,
        )
        return result if result is not None else 0
    except KeyboardInterrupt:
        result = await finalize_tbench_run(request.environment)
        return result if result is not None else 0
    except Exception as exc:
        result = await finalize_tbench_run(request.environment)
        if request.stream_json:
            print(json.dumps({"type": "error", "error": str(exc)}), flush=True)
        else:
            print(f"\nError: {exc}", file=sys.stderr)
        return result if result is not None else 1


async def run_interactive_mode(
    request: ChatRunRequest,
    *,
    trajectory: Trajectory,
    session_id: str | None,
    parent_session_id: str | None,
    branch_point: int | None,
    bootstrap_input: str | None,
) -> int:
    try:
        frontend = create_interactive_frontend(
            frontend_name=request.frontend,
            environment=request.environment,
            endpoint=request.endpoint,
            theme=request.theme,
            debug=request.debug,
            debug_layout=request.debug_layout,
            driver=request.driver,
            detached=request.detached,
        )
    except ValueError:
        return 1

    run_fn = create_run_fn(
        DriverBuildRequest(
            driver=request.driver,
            endpoint=request.endpoint,
            cwd=request.cwd,
            working_dir=request.working_dir,
            trajectory=trajectory,
            session_id=session_id,
            cursor_api_key=request.cursor_api_key,
            autonomous=False,
        )
    )

    runner_config = RunnerConfig(
        session_store=request.session_store,
        session_id=session_id,
        parent_session_id=parent_session_id,
        branch_point=branch_point,
        confirm_tools=request.confirm_tools,
        bootstrap_input=bootstrap_input,
        detached=request.detached,
        cwd=request.cwd,
        run_fn=run_fn,
    )

    try:
        result, _ = await _run_with_optional_timeout(
            request,
            trajectory=trajectory,
            runner_config=runner_config,
            frontend=frontend,
        )
        return result if result is not None else 0
    except KeyboardInterrupt:
        print("\n\n✅ Agent stopped")
        result = await finalize_tbench_run(request.environment)
        return result if result is not None else 0
