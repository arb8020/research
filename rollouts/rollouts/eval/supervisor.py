from __future__ import annotations

import argparse
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

import trio

from argus.event_log import RunEventSinks, build_jsonl_run_event_sinks, emit_run_event
from rollouts.eval.configs import EndpointConfig, resolve_eval_task_spec
from rollouts.eval.endpoint_realization import realize_worker_backed_endpoint
from rollouts.eval.run import REPO_ROOT, load_config_module


def _resolve_eval_worker(config_module: Any) -> Any | None:
    worker_topology = getattr(config_module, "worker_topology", None)
    if worker_topology is None:
        return None
    return worker_topology.get_worker_for_role("actor")


def _child_command(*, config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "rollouts.eval.run",
        "--config",
        str(config_path),
    ]


def _setup_run_logging(output_dir: Path) -> RunEventSinks:
    return build_jsonl_run_event_sinks(output_dir / "control.jsonl")


def _last_nonempty_log_line(path: Path) -> str | None:
    if not path.exists():
        return None
    last_nonempty: str | None = None
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped:
            last_nonempty = stripped
    return last_nonempty


def _emit_child_failure_summary(
    *, run_logger: RunEventSinks, output_dir: Path, exit_code: int
) -> None:
    stderr_log = output_dir / "stderr.log"
    stdout_log = output_dir / "stdout.log"
    error_summary = _last_nonempty_log_line(stderr_log) or _last_nonempty_log_line(stdout_log)
    emit_run_event(
        run_logger,
        "eval_child_failed",
        exit_code=exit_code,
        error_summary=error_summary,
        stderr_log=str(stderr_log),
        stdout_log=str(stdout_log),
    )


def _spawn_child(
    *, command: list[str], output_dir: Path, child_env: dict[str, str]
) -> subprocess.Popen[bytes]:
    stdout_log = output_dir / "stdout.log"
    stderr_log = output_dir / "stderr.log"
    stdout_handle = stdout_log.open("a")
    stderr_handle = stderr_log.open("a")
    try:
        return subprocess.Popen(
            command,
            cwd=str(REPO_ROOT.parent),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()


async def _wait_for_process(proc: subprocess.Popen[bytes]) -> int:
    return await trio.to_thread.run_sync(proc.wait)


async def _run_eval(
    *,
    config_path: Path,
    output_dir: Path,
    max_samples: int | None,
    force_deploy_committed: bool,
) -> int:
    run_logger = _setup_run_logging(output_dir)
    config_module = load_config_module(config_path)
    eval_task = resolve_eval_task_spec(config_module)
    endpoint_config = eval_task.run_spec.endpoint
    if endpoint_config is None and eval_task.run_spec.attempt_executor is None:
        endpoint_config = EndpointConfig()

    child_env = os.environ.copy()
    child_env["ROLLOUTS_OUTPUT_DIR"] = str(output_dir)
    command = _child_command(config_path=config_path)

    # TODO(eval-cli): `rollouts.eval.run` still has no explicit max-samples CLI.
    # Keep the supervisor contract narrow for now and let config own sample count.
    del max_samples

    worker = _resolve_eval_worker(config_module)
    if endpoint_config is not None and endpoint_config.requires_server:
        async with realize_worker_backed_endpoint(
            endpoint_config=endpoint_config,
            output_dir=output_dir,
            hardware_config=eval_task.hardware,
            server_config=eval_task.server,
            worker=worker,
            run_name=output_dir.name,
            force_deploy_committed=force_deploy_committed,
            run_logger=run_logger,
        ) as realized:
            child_env["ROLLOUTS_ENDPOINT_BASE_URL"] = realized.endpoint_config.base_url or ""
            proc = _spawn_child(command=command, output_dir=output_dir, child_env=child_env)
            return await _wait_for_process(proc)

    proc = _spawn_child(command=command, output_dir=output_dir, child_env=child_env)
    return await _wait_for_process(proc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Argus-owned evaluation supervisor")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-deploy-committed", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_logger = _setup_run_logging(output_dir)
    try:
        exit_code = trio.run(
            partial(
                _run_eval,
                config_path=config_path,
                output_dir=output_dir,
                max_samples=args.max_samples,
                force_deploy_committed=args.force_deploy_committed,
            )
        )
        if exit_code != 0:
            _emit_child_failure_summary(
                run_logger=run_logger, output_dir=output_dir, exit_code=exit_code
            )
        emit_run_event(
            run_logger,
            "run_end",
            status="ok" if exit_code == 0 else "failed",
            exit_code=exit_code,
        )
        return exit_code
    except Exception as exc:
        emit_run_event(run_logger, "run_end", status="failed", error=str(exc))
        print(f"Argus eval supervisor failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
