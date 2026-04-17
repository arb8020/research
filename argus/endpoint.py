"""argus endpoint — long-lived owned endpoints with manual lifecycle.

Closes the "API hole" in argus: `argus run` owns the full {launch, eval,
teardown} cycle, which tears down the container after every eval and makes
interactive debugging (curl, repeat evals against a warm server) painful.

Subcommands:
    argus endpoint up --config <cfg> [--name <name>] [--force-deploy-committed]
        Launch an OwnedEndpoint in a detached process. Prints the name and
        local tunnel URL when ready.

    argus endpoint down <name>
        SIGTERM the detached process. Waits for clean teardown.

    argus endpoint list
        Show active endpoints with URL and age.

Handles live at `~/.argus/endpoints/<name>.json`.

Note: This imports the private `_realize_ssh_endpoint`-adjacent code via
`realize_worker_backed_endpoint` (which is public-named). If the underlying
context manager needs renaming later, update this file accordingly.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HANDLES_DIR = Path.home() / ".argus" / "endpoints"
# Max seconds the `up` parent process waits for the child to write a URL to
# its handle file. OwnedEndpoint.startup_timeout is the real ceiling; we just
# need to outlast the typical boot time (2-3 min for DeepSeek V3.2 with
# cached weights) so the user sees the URL when it's ready.
STARTUP_WAIT_S = 1800


def _handles_dir() -> Path:
    HANDLES_DIR.mkdir(parents=True, exist_ok=True)
    return HANDLES_DIR


def _handle_path(name: str) -> Path:
    return _handles_dir() / f"{name}.json"


def _read_handle(name: str) -> dict[str, Any] | None:
    path = _handle_path(name)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_handle(name: str, data: dict[str, Any]) -> None:
    _handle_path(name).write_text(json.dumps(data, indent=2))


def _remove_handle(name: str) -> None:
    p = _handle_path(name)
    if p.exists():
        p.unlink()


def _process_alive(pid: int | None) -> bool:
    # Guard against None / invalid pids. os.kill(-1, 0) would signal the whole
    # process group, which is catastrophically wrong.
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _load_eval_task(config_path: Path) -> Any:
    """Import the config module and return its eval_task."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_argus_endpoint_cfg", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from rollouts.eval.configs import resolve_eval_task_spec

    return resolve_eval_task_spec(module)


# ──────────────────────── Child (detached worker) ────────────────────────────


async def _run_detached(
    name: str,
    config_path: Path,
    force_deploy_committed: bool,
    events: Any,
) -> None:
    """Body of the detached process.

    Enters the realize_worker_backed_endpoint context, writes the tunneled URL
    into the handle file, then blocks until SIGTERM. On SIGTERM the context
    exits cleanly (docker rm, tunnel close).

    All lifecycle transitions are emitted to the JSONL journal via `events`.
    realize_worker_backed_endpoint also receives `events` as run_logger so its
    own per-step events (image pull, container launch, readiness poll, tunnel
    open) land in the same journal — one place to look when debugging.
    """
    import trio

    from rollouts.eval.endpoint_realization import realize_worker_backed_endpoint

    from .event_log import emit_run_event

    eval_task = _load_eval_task(config_path)
    endpoint_config = eval_task.run_spec.endpoint
    hardware_config = eval_task.hardware
    server_config = eval_task.server

    output_dir = Path.cwd() / "results" / "endpoint" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    emit_run_event(
        events,
        "endpoint_child_started",
        name=name,
        config_path=str(config_path),
        force_deploy_committed=force_deploy_committed,
    )

    # Use trio's signal handling so the async context manager gets cancelled
    # cleanly rather than the process being SIGKILL'd mid-teardown. Signal
    # handler cancels the nursery scope; trio unwinds through the async-with
    # stack, which runs realize_worker_backed_endpoint's teardown (stop
    # service → docker rm, close paramiko tunnel).
    async with trio.open_nursery() as nursery:

        async def wait_for_signal() -> None:
            with trio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as sigs:
                async for sig in sigs:
                    emit_run_event(events, "endpoint_signal_received", signal=int(sig))
                    nursery.cancel_scope.cancel()
                    return

        async def run_endpoint() -> None:
            emit_run_event(events, "endpoint_realize_start")
            async with realize_worker_backed_endpoint(
                endpoint_config=endpoint_config,
                output_dir=output_dir,
                hardware_config=hardware_config,
                server_config=server_config,
                run_name=name,
                force_deploy_committed=force_deploy_committed,
                run_logger=events,
            ) as realized:
                url = realized.endpoint_config.base_url
                emit_run_event(events, "endpoint_ready", url=url, metadata=realized.metadata)
                # Publish URL to handle file so parent process can return.
                handle = _read_handle(name) or {}
                handle["url"] = url
                handle["status"] = "ready"
                handle["ready_at"] = datetime.now(timezone.utc).isoformat()
                _write_handle(name, handle)

                # Block until nursery is cancelled (SIGTERM).
                await trio.sleep_forever()

            emit_run_event(events, "endpoint_teardown_done")

        nursery.start_soon(wait_for_signal)
        nursery.start_soon(run_endpoint)

    emit_run_event(events, "endpoint_child_exiting")
    # Clean up handle file on normal exit. (It may already be gone if `down`
    # removed it after we exited the context.)
    _remove_handle(name)


def _run_detached_main(argv: list[str]) -> int:
    """Entry for the detached child. Not user-facing.

    All lifecycle is captured in a JSONL event journal at
    ~/.argus/endpoints/logs/<name>.jsonl — human-readable via `jq`, diffable,
    queryable by status/time. stdout/stderr from third-party libraries is
    redirected to .out/.err as a fallback for anything that bypasses our
    event stream (e.g. uncaught library prints, tracebacks from
    pre-trio-run code).
    """
    import traceback

    import trio

    from .event_log import build_jsonl_run_event_sinks, emit_run_event

    parser = argparse.ArgumentParser(prog="argus endpoint __child__")
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force-deploy-committed", action="store_true")
    args = parser.parse_args(argv)

    logs_dir = Path.home() / ".argus" / "endpoints" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Fallback capture for stdout/stderr from library code that bypasses the
    # event log. Our own messages go through emit_run_event below.
    sys.stdout = open(logs_dir / f"{args.name}.out", "a", buffering=1)
    sys.stderr = open(logs_dir / f"{args.name}.err", "a", buffering=1)

    events = build_jsonl_run_event_sinks(logs_dir / f"{args.name}.jsonl")

    try:
        trio.run(
            _run_detached,
            args.name,
            args.config,
            args.force_deploy_committed,
            events,
        )
    except Exception as exc:
        emit_run_event(
            events,
            "endpoint_child_crashed",
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        # Annotate the handle so the user can see the failure status.
        handle = _read_handle(args.name) or {}
        handle["status"] = "crashed"
        handle["error"] = repr(exc)
        _write_handle(args.name, handle)
        return 1
    return 0


# ──────────────────────── Subcommands ────────────────────────────────────────


def _cmd_up(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    name = args.name or config_path.stem
    if _read_handle(name) is not None:
        existing = _read_handle(name)
        if existing and _process_alive(existing.get("pid", -1)):
            print(
                f"Endpoint '{name}' already running (pid {existing['pid']}). "
                f"Use `argus endpoint down {name}` first, or pass --name for a new instance.",
                file=sys.stderr,
            )
            return 1
        _remove_handle(name)

    # Spawn detached child. start_new_session=True detaches from the parent's
    # process group so the child survives parent exit.
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "argus.endpoint_child",
        "--name",
        name,
        "--config",
        str(config_path),
    ]
    if args.force_deploy_committed:
        cmd.append("--force-deploy-committed")

    # Initial handle so `up` and `down` have something to coordinate on even
    # before the child writes its ready state.
    _write_handle(
        name,
        {
            "name": name,
            "pid": None,  # Set by child below.
            "config_path": str(config_path),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "starting",
            "url": None,
        },
    )

    proc = subprocess.Popen(cmd, start_new_session=True)
    handle = _read_handle(name) or {}
    handle["pid"] = proc.pid
    _write_handle(name, handle)

    logs_dir = Path.home() / ".argus" / "endpoints" / "logs"
    print(f"Endpoint '{name}' launching (pid {proc.pid})")
    print(f"  config: {config_path}")
    print(f"  events: {logs_dir / f'{name}.jsonl'}")
    print(f"  stderr: {logs_dir / f'{name}.err'}")
    print(f"  waiting for URL (up to {STARTUP_WAIT_S}s)...")

    # Poll the handle until URL appears or child dies.
    deadline = time.monotonic() + STARTUP_WAIT_S
    while time.monotonic() < deadline:
        if not _process_alive(proc.pid):
            handle = _read_handle(name) or {}
            print(
                f"\nDetached process exited before ready (status={handle.get('status')}, "
                f"error={handle.get('error')}). See logs.",
                file=sys.stderr,
            )
            return 1
        handle = _read_handle(name) or {}
        if handle.get("status") == "ready" and handle.get("url"):
            print(f"\nReady: {handle['url']}")
            print(f"  teardown: argus endpoint down {name}")
            return 0
        time.sleep(2)

    print(
        f"\nTimed out after {STARTUP_WAIT_S}s. Endpoint may still come up; check "
        f"`argus endpoint list` or the logs.",
        file=sys.stderr,
    )
    return 1


def _cmd_down(args: argparse.Namespace) -> int:
    handle = _read_handle(args.name)
    if handle is None:
        print(f"No endpoint named '{args.name}'", file=sys.stderr)
        return 1

    pid = handle.get("pid")
    if pid is None or not _process_alive(pid):
        print(f"Process for '{args.name}' already gone; clearing handle.")
        _remove_handle(args.name)
        return 0

    print(f"Stopping endpoint '{args.name}' (pid {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _remove_handle(args.name)
        return 0

    # Wait for clean exit (async teardown runs docker rm, tunnel close).
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            _remove_handle(args.name)
            print("Stopped.")
            return 0
        time.sleep(1)

    print(f"Process {pid} did not exit within 120s. It may be stuck in teardown.", file=sys.stderr)
    print("Sending SIGKILL.", file=sys.stderr)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _remove_handle(args.name)
    return 1


def _cmd_logs(args: argparse.Namespace) -> int:
    logs_dir = Path.home() / ".argus" / "endpoints" / "logs"
    jsonl_path = logs_dir / f"{args.name}.jsonl"
    if not jsonl_path.exists():
        print(f"No event log for '{args.name}' at {jsonl_path}", file=sys.stderr)
        return 1
    with jsonl_path.open() as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(line, end="")
                continue
            ts = entry.pop("ts", "")
            event = entry.pop("event", "?")
            rest = json.dumps(entry, default=str)
            print(f"{ts}  {event}  {rest}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    handles = sorted(_handles_dir().glob("*.json"))
    if not handles:
        print("No endpoints.")
        return 0

    for h in handles:
        data = json.loads(h.read_text())
        name = data.get("name", h.stem)
        pid = data.get("pid")
        status = data.get("status", "?")
        url = data.get("url", "<no url yet>")
        alive = _process_alive(pid) if pid else False
        print(f"{name}\t{status}\t{'alive' if alive else 'dead'}\tpid={pid}\t{url}")
    return 0


# ──────────────────────── Entrypoint ─────────────────────────────────────────


def endpoint_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="argus endpoint",
        description="Manage long-lived owned endpoints outside of eval runs.",
    )
    sub = parser.add_subparsers(dest="subcmd", required=True)

    up = sub.add_parser("up", help="Launch an endpoint in a detached process")
    up.add_argument("--config", type=Path, required=True, help="Path to an eval config")
    up.add_argument("--name", default=None, help="Endpoint name (default: config stem)")
    up.add_argument("--force-deploy-committed", action="store_true")
    up.set_defaults(func=_cmd_up)

    down = sub.add_parser("down", help="Tear down a running endpoint")
    down.add_argument("name")
    down.set_defaults(func=_cmd_down)

    ls = sub.add_parser("list", help="List active endpoints")
    ls.set_defaults(func=_cmd_list)

    logs = sub.add_parser("logs", help="Show the JSONL event log for an endpoint")
    logs.add_argument("name")
    logs.set_defaults(func=_cmd_logs)

    args = parser.parse_args(argv)
    return args.func(args)
