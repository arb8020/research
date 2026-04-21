from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Any

import trio

from argus.event_log import RunEventSinks, build_jsonl_run_event_sinks, emit_run_event
from rollouts.eval.endpoint_realization import realize_worker_backed_endpoint
from rollouts.eval.run import load_config_module
from rollouts.remote_runtime import resolve_consumer_project

from .configs import resolve_serving_scenario


def _child_command(*, config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "rollouts.serving.run",
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
    *,
    run_logger: RunEventSinks,
    output_dir: Path,
    exit_code: int,
) -> None:
    stderr_log = output_dir / "stderr.log"
    stdout_log = output_dir / "stdout.log"
    error_summary = _last_nonempty_log_line(stderr_log) or _last_nonempty_log_line(stdout_log)
    emit_run_event(
        run_logger,
        "serving_child_failed",
        exit_code=exit_code,
        error_summary=error_summary,
        stderr_log=str(stderr_log),
        stdout_log=str(stdout_log),
    )


def _spawn_child(
    *,
    command: list[str],
    output_dir: Path,
    child_env: dict[str, str],
    cwd: Path,
) -> subprocess.Popen[bytes]:
    stdout_log = output_dir / "stdout.log"
    stderr_log = output_dir / "stderr.log"
    stdout_handle = stdout_log.open("a")
    stderr_handle = stderr_log.open("a")
    try:
        return subprocess.Popen(
            command,
            cwd=str(cwd),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()


async def _wait_for_process(proc: subprocess.Popen[bytes]) -> int:
    """Await child exit with frequent checkpoints so trio can be cancelled.

    Using `trio.to_thread.run_sync(proc.wait)` would block a worker thread
    indefinitely and prevent cancellation from ever landing. We poll instead
    so the outer scope can tear this down when shutdown is signalled.
    """
    while proc.poll() is None:
        await trio.sleep(0.1)
    return proc.returncode


# Lifecycle helpers -----------------------------------------------------------
# These implement the "drain on shutdown" semantics described in ServingRun.
# The eval child is a subprocess (rollouts.serving.run), so our drain mechanism
# is: send SIGTERM, wait up to drain_timeout for natural exit, then SIGKILL.
# We rely on the child's own SIGINT handling (rollouts/eval/run.py) to cancel
# its in-flight work cleanly; SIGTERM gets mapped to that same path below.


async def _duration_watchdog(
    *,
    duration_s: float,
    shutdown_reason: list[str],
    shutdown_event: trio.Event,
    run_logger: RunEventSinks,
) -> None:
    """Signal shutdown when the configured duration elapses.

    Fires shutdown_event, which the child supervisor observes and turns into
    a drain of the eval subprocess. This is the "stop accepting new work"
    trigger, not a hard cancel.
    """
    await trio.sleep(duration_s)
    if not shutdown_event.is_set():
        shutdown_reason.append("duration_elapsed")
        emit_run_event(run_logger, "serving_duration_elapsed", duration_s=duration_s)
        shutdown_event.set()


async def _drain_child(
    *,
    proc: subprocess.Popen[bytes],
    drain_timeout_s: float | None,
    run_logger: RunEventSinks,
) -> int:
    """Signal the eval child to stop and wait for it to exit.

    First we send SIGINT (which the child's eval loop turns into a cancel of
    its own trio scope — clean drain). If drain_timeout_s elapses without the
    child exiting, we escalate to SIGTERM, then SIGKILL. Returns the child's
    exit code.
    """
    if proc.poll() is not None:
        return proc.returncode

    emit_run_event(
        run_logger,
        "serving_drain_start",
        child_pid=proc.pid,
        drain_timeout_s=drain_timeout_s,
    )
    # SIGINT matches the child's existing handler (rollouts/eval/run.py
    # installs one that cancels its trio scope). SIGTERM would work too, but
    # SIGINT is what the child is already equipped to translate into a clean
    # drain of in-flight samples.
    try:
        proc.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return proc.returncode

    if drain_timeout_s is None:
        # No drain window — caller wants immediate cancel. Escalate now.
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
        return await _wait_for_process(proc)

    with trio.move_on_after(drain_timeout_s):
        exit_code = await _wait_for_process(proc)
        emit_run_event(
            run_logger,
            "serving_drain_clean",
            child_pid=proc.pid,
            exit_code=exit_code,
        )
        return exit_code

    # Drain timeout expired. Escalate to SIGTERM, then SIGKILL.
    emit_run_event(
        run_logger,
        "serving_drain_timeout",
        child_pid=proc.pid,
        drain_timeout_s=drain_timeout_s,
    )
    try:
        proc.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return proc.returncode
    with trio.move_on_after(5.0):
        return await _wait_for_process(proc)
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    emit_run_event(run_logger, "serving_drain_sigkill", child_pid=proc.pid)
    return await _wait_for_process(proc)


async def _supervise_child(
    *,
    proc: subprocess.Popen[bytes],
    shutdown_event: trio.Event,
    drain_timeout_s: float | None,
    run_logger: RunEventSinks,
    exit_code_slot: list[int | None],
) -> None:
    """Wait for child to exit, or drain it if shutdown is signalled first."""
    async with trio.open_nursery() as nursery:
        # One task waits for the child to exit on its own.
        # Another waits for shutdown and initiates drain.
        async def _wait_natural() -> None:
            exit_code_slot[0] = await _wait_for_process(proc)
            nursery.cancel_scope.cancel()

        async def _wait_shutdown() -> None:
            await shutdown_event.wait()
            exit_code_slot[0] = await _drain_child(
                proc=proc,
                drain_timeout_s=drain_timeout_s,
                run_logger=run_logger,
            )
            nursery.cancel_scope.cancel()

        nursery.start_soon(_wait_natural)
        nursery.start_soon(_wait_shutdown)


async def _run_serving(
    *,
    config_path: Path,
    output_dir: Path,
    consumer_project_root: Path,
    force_deploy_committed: bool,
    shutdown_event: trio.Event,
    shutdown_reason: list[str],
) -> int:
    run_logger = _setup_run_logging(output_dir)
    config_module = load_config_module(config_path)
    scenario = resolve_serving_scenario(config_module)
    endpoint_config = scenario.endpoint
    child_env = os.environ.copy()
    child_env["ROLLOUTS_OUTPUT_DIR"] = str(output_dir)
    command = _child_command(config_path=config_path)

    duration_s = scenario.duration.total_seconds() if scenario.duration is not None else None
    drain_timeout_s = (
        scenario.drain_timeout.total_seconds() if scenario.drain_timeout is not None else None
    )

    # TODO(on-engine-crash=restart): when scenario.on_engine_crash == "restart",
    # this block needs to wrap the endpoint realize-and-wait loop, restarting
    # the endpoint if it exits unexpectedly. The eval child would also need to
    # tolerate base_url changing mid-run, which it doesn't today (URL is
    # threaded via env var at spawn). Left as a follow-up — current behavior
    # matches the old one-shot: endpoint boots, workloads run, teardown.
    if scenario.on_engine_crash == "restart":
        emit_run_event(
            run_logger,
            "serving_on_engine_crash_restart_not_implemented",
            note="on_engine_crash='restart' accepted by config but not yet wired; behaving as 'fail'.",
        )

    async def _run_with_endpoint() -> int:
        exit_code_slot: list[int | None] = [None]
        async with trio.open_nursery() as nursery:
            if duration_s is not None:
                nursery.start_soon(
                    partial(
                        _duration_watchdog,
                        duration_s=duration_s,
                        shutdown_reason=shutdown_reason,
                        shutdown_event=shutdown_event,
                        run_logger=run_logger,
                    )
                )

            proc = _spawn_child(
                command=command,
                output_dir=output_dir,
                child_env=child_env,
                cwd=consumer_project_root,
            )
            nursery.start_soon(
                partial(
                    _supervise_child,
                    proc=proc,
                    shutdown_event=shutdown_event,
                    drain_timeout_s=drain_timeout_s,
                    run_logger=run_logger,
                    exit_code_slot=exit_code_slot,
                )
            )

        assert exit_code_slot[0] is not None, "child supervision exited without an exit code"
        return exit_code_slot[0]

    if endpoint_config.requires_server:
        async with realize_worker_backed_endpoint(
            endpoint_config=endpoint_config,
            output_dir=output_dir,
            hardware_config=scenario.hardware,
            server_config=scenario.server,
            worker=None,
            run_name=output_dir.name,
            force_deploy_committed=force_deploy_committed,
            run_logger=run_logger,
            consumer_project_root=consumer_project_root,
            reuse_running_endpoint=scenario.reuse_running_endpoint,
            leave_endpoint_running=scenario.leave_endpoint_running,
        ) as realized:
            child_env["ROLLOUTS_ENDPOINT_BASE_URL"] = realized.endpoint_config.base_url or ""
            return await _run_with_endpoint()

    return await _run_with_endpoint()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Argus-owned serving scenario supervisor")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-deploy-committed", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    consumer_project_root = Path(resolve_consumer_project(config_path).local_root)

    run_logger = _setup_run_logging(output_dir)

    # Signal-driven shutdown. SIGTERM/SIGINT set the event; the child
    # supervisor turns that into a drain of the eval subprocess. Second
    # signal escalates to immediate teardown via KeyboardInterrupt — the
    # outer trio.run unwinds and realize_worker_backed_endpoint's __aexit__
    # tears the remote endpoint down.
    shutdown_event = trio.Event()
    shutdown_reason: list[str] = []
    signal_count = {"n": 0}
    original_handlers: dict[int, Any] = {}

    def _handle_shutdown_signal(signum: int, frame: FrameType | None) -> None:
        del frame
        signal_count["n"] += 1
        sig_name = signal.Signals(signum).name
        if signal_count["n"] == 1:
            if not shutdown_event.is_set():
                shutdown_reason.append(f"signal:{sig_name}")
                emit_run_event(run_logger, "serving_signal_shutdown", signal=sig_name)
            # Python delivers signals on the main thread between bytecode
            # instructions. trio.Event.set() is synchronous state mutation
            # plus a wake of pending tasks, which is safe here — trio's
            # scheduler observes the set at the next checkpoint, and
            # _wait_for_process polls frequently enough for the signal to
            # propagate within ~100ms.
            shutdown_event.set()
            return
        # Second signal: user is insisting. Restore default handler and
        # re-raise so the default action (KeyboardInterrupt / terminate)
        # takes over and the outer trio.run unwinds fast.
        emit_run_event(run_logger, "serving_signal_force", signal=sig_name)
        signal.signal(signum, original_handlers.get(signum, signal.SIG_DFL))
        os.kill(os.getpid(), signum)

    original_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
    original_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    try:
        exit_code = trio.run(
            partial(
                _run_serving,
                config_path=config_path,
                output_dir=output_dir,
                consumer_project_root=consumer_project_root,
                force_deploy_committed=args.force_deploy_committed,
                shutdown_event=shutdown_event,
                shutdown_reason=shutdown_reason,
            )
        )
        if exit_code != 0:
            _emit_child_failure_summary(
                run_logger=run_logger,
                output_dir=output_dir,
                exit_code=exit_code,
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
        print(f"Argus serving supervisor failed: {exc}", file=sys.stderr)
        raise
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
