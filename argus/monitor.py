"""Argus monitor.

Argus owns control-plane concerns:

- resolve run identity
- attach/sync remote artifacts when needed
- expose launcher/control-plane status
- tail or poll local journals

Local monitor TUI support has been removed. The remaining monitor surfaces are:

- local `--tail`
- local `--wait-for-event`
- remote `--attach --tail`
- remote `--attach --tail-lines`
- remote `--attach --sync-only`
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from argus.tail import tail_run

LAUNCHES_DIR = Path.home() / ".argus" / "launches"
_DEFAULT_WAIT_FAILURE_EVENTS = frozenset({"run_failed"})
_MONITOR_LOG: Path | None = None
_MODAL_ATTACH_FILES = (
    "control.jsonl",
    "training.jsonl",
    "metrics.jsonl",
    "rollouts.jsonl",
    "engine.jsonl",
    "environment.jsonl",
)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _print_launches() -> int:
    if not LAUNCHES_DIR.exists():
        print("No active launcher records found in ~/.argus/launches")
        return 0

    records: list[dict[str, Any]] = []
    for path in sorted(LAUNCHES_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        pid = payload.get("pid")
        payload["alive"] = isinstance(pid, int) and _process_alive(pid)
        records.append(payload)

    if not records:
        print("No active launcher records found in ~/.argus/launches")
        return 0

    print(f"{'LAUNCHER ID':<36} {'PID':<8} {'ALIVE':<6} {'PROVIDER':<10} {'CONFIG'}")
    print("-" * 140)
    for record in records:
        print(
            f"{record.get('launcher_id', ''):<36} "
            f"{str(record.get('pid', '')):<8} "
            f"{str(record.get('alive', False)):<6} "
            f"{str(record.get('provider', '')):<10} "
            f"{record.get('config_path', '')}"
        )
    return 0


def _log(event: str, **data: Any) -> None:
    """Append a structured log event to monitor.jsonl in the run directory."""
    if _MONITOR_LOG is None:
        return
    entry = {
        "ts": datetime.now().isoformat(),
        "source": "argus.monitor",
        "event": event,
        **data,
    }
    try:
        with open(_MONITOR_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _broker_credentials() -> dict[str, str]:
    """Load broker credentials: shared.config (if available) -> env vars."""
    try:
        from infra_utils.config import (
            get_digitalocean_key,
            get_lambda_key,
            get_prime_key,
            get_runpod_key,
            get_vast_key,
        )

        credentials: dict[str, str] = {}
        if key := get_runpod_key():
            credentials["runpod"] = key
        if key := get_prime_key():
            credentials["primeintellect"] = key
        if key := get_lambda_key():
            credentials["lambdalabs"] = key
        if key := get_vast_key():
            credentials["vast"] = key
        if key := get_digitalocean_key():
            credentials["digitalocean"] = key
        return credentials
    except ImportError:
        pass

    credentials = {}
    env_map = [
        ("RUNPOD_API_KEY", "runpod"),
        ("PRIME_API_KEY", "primeintellect"),
        ("LAMBDA_API_KEY", "lambdalabs"),
        ("VAST_API_KEY", "vast"),
        ("DIGITALOCEAN_API_KEY", "digitalocean"),
    ]
    for env_key, provider in env_map:
        if val := os.environ.get(env_key):
            credentials[provider] = val
    return credentials


def _get_instance(provider: str, node_id: str) -> object | None:
    """Query broker for a single instance. Returns ClientGPUInstance or None."""
    import trio

    from broker.client import GPUClient

    credentials = _broker_credentials()
    assert credentials, "No broker credentials found. Set RUNPOD_API_KEY."

    async def _fetch() -> object | None:
        client = GPUClient(credentials=credentials)
        return await client.get_instance(node_id, provider)

    return trio.run(_fetch)


def _print_runs(*, probe: bool) -> int:
    """Print known remote jobs, optionally probing LogsServer reachability."""
    from rollouts.jobs import list_jobs

    jobs = list_jobs()
    if not jobs:
        print("No rollouts jobs found (no pods with 'rollouts/' name prefix).")
        return 0

    header = f"{'JOB ID':<30} {'NODE':<25}"
    if probe:
        header += f" {'STATUS':<10}"
    print(header)
    print("-" * len(header))

    for job in jobs:
        node_str = ", ".join(job.node_ids) if job.nodes else "?"
        row = f"{job.job_id:<30} {node_str:<25}"

        if probe:
            status = "alive"
            node = job.nodes[0]
            try:
                inst = _get_instance(node.provider, node.node_id)
                if inst:
                    logs_host, logs_port = _resolve_logs_endpoint(inst)
                    if logs_host and logs_port:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(2.0)
                        sock.connect((logs_host, int(logs_port)))
                        sock.close()
                        status = "logs ok"
            except (OSError, ValueError, Exception):
                status = "no logs"

            row += f" {status:<10}"

        print(row)

    return 0


def _resolve_logs_endpoint(instance: object) -> tuple[str | None, int | None]:
    """Extract LogsServer host:port from instance runtime port mapping."""
    raw = instance.raw_data or {}
    runtime = raw.get("runtime") or {}
    runtime_ports = runtime.get("ports") or []

    for p in runtime_ports:
        if p.get("privatePort") == 9100 and p.get("isIpPublic"):
            return p["ip"], p["publicPort"]
    return None, None


def _resolve_job_connection(job_id: str | None) -> dict:
    """Resolve a job to its LogsServer connection info."""
    from dotenv import load_dotenv

    from rollouts.jobs import get_job, get_latest_job

    load_dotenv()
    job = get_latest_job() if job_id is None else get_job(job_id)
    assert job.nodes, f"Job {job.job_id} has no nodes"
    node = job.nodes[0]
    node_id_str = f"{node.provider}:{node.node_id}"

    if node.provider == "modal":
        return {
            "run_id": job.job_id,
            "provider": node.provider,
            "node_id": node_id_str,
            "log_path": job.log_path,
            "logs_host": None,
            "logs_port": None,
        }

    instance = _get_instance(node.provider, node.node_id)
    if instance is None:
        return {
            "run_id": job.job_id,
            "provider": node.provider,
            "node_id": node_id_str,
            "log_path": job.log_path,
            "logs_host": None,
            "logs_port": None,
        }

    logs_host, logs_port = _resolve_logs_endpoint(instance)
    return {
        "run_id": job.job_id,
        "provider": node.provider,
        "node_id": node_id_str,
        "log_path": job.log_path,
        "logs_host": logs_host,
        "logs_port": logs_port,
    }


def _get_modal_sandbox(sandbox_id: str) -> Any:
    import modal

    return modal.Sandbox.from_id(sandbox_id)


def _modal_output_dir_candidates(run_id: str, log_path: str | None) -> tuple[str, ...]:
    suffixes = [f"results/rl/{run_id}"]
    if log_path:
        suffixes.insert(0, log_path.strip("/"))

    candidates: list[str] = []
    prefixes = (
        "/workspace/research/rollouts",
        "/workspace/research",
        "/root/.bifrost/workspaces/rollouts-rl/rollouts",
        "/root/.bifrost/workspaces/rollouts-rl",
    )
    for suffix in suffixes:
        for prefix in prefixes:
            candidates.append(f"{prefix}/{suffix}".replace("//", "/"))
    return tuple(dict.fromkeys(candidates))


def _resolve_modal_output_dir(sandbox: Any, run_id: str, log_path: str | None) -> str:
    for candidate in _modal_output_dir_candidates(run_id, log_path):
        try:
            sandbox.ls(candidate)
            return candidate
        except Exception:
            continue
    raise FileNotFoundError(f"Could not find remote output dir for run {run_id!r} in Modal sandbox")


def _read_modal_text_file(sandbox: Any, path: str) -> str | None:
    try:
        handle = sandbox.open(path, "r")
    except Exception:
        return None
    try:
        return handle.read()
    finally:
        handle.close()


def _sync_modal_files_once(
    *,
    sandbox: Any,
    remote_output_dir: str,
    local_sync_dir: Path,
    line_offsets: dict[str, int],
) -> tuple[tuple[str, ...], int]:
    files: list[str] = []
    total_new_lines = 0

    for filename in _MODAL_ATTACH_FILES:
        text = _read_modal_text_file(sandbox, f"{remote_output_dir}/{filename}")
        if text is None:
            continue
        files.append(filename)
        lines = text.splitlines()
        previous = line_offsets.get(filename, 0)
        if previous > len(lines):
            previous = 0
            local_path = local_sync_dir / filename
            if local_path.exists():
                local_path.unlink()
        new_lines = lines[previous:]
        if new_lines:
            local_path = local_sync_dir / filename
            with open(local_path, "a", encoding="utf-8") as f:
                for line in new_lines:
                    f.write(line + "\n")
            total_new_lines += len(new_lines)
        line_offsets[filename] = len(lines)

    return tuple(files), total_new_lines


def _open_ssh_tunnel(
    node_id: str,
    remote_port: int = 9100,
) -> tuple[int, Callable[[], None]]:
    """Open an SSH tunnel to a remote LogsServer."""
    import paramiko
    import trio
    from dotenv import load_dotenv

    from broker.client import GPUClient

    provider, instance_id = node_id.split(":", 1)
    load_dotenv()

    credentials = _broker_credentials()
    client = GPUClient(credentials=credentials)
    instance = trio.run(client.get_instance, instance_id, provider)
    assert instance is not None, f"Instance not found: {node_id}"

    ssh_key = client.get_ssh_key_path(provider) or os.path.expanduser("~/.ssh/id_ed25519")
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(
        hostname=instance.public_ip,
        port=instance.ssh_port,
        username="root",
        key_filename=ssh_key,
        timeout=30,
    )

    transport = ssh_client.get_transport()
    assert transport is not None

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    local_port = server.getsockname()[1]
    server.listen(4)

    def _forward(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(4096)
                if not data:
                    break
                dst.sendall(data)
        except (OSError, EOFError):
            pass
        finally:
            try:
                src.close()
            except (OSError, EOFError):
                pass
            try:
                dst.close()
            except (OSError, EOFError):
                pass

    def _tunnel_accept_loop() -> None:
        while True:
            try:
                client_sock, addr = server.accept()
                channel = transport.open_channel(
                    "direct-tcpip",
                    ("127.0.0.1", remote_port),
                    addr,
                )
                threading.Thread(target=_forward, args=(client_sock, channel), daemon=True).start()
                threading.Thread(target=_forward, args=(channel, client_sock), daemon=True).start()
            except Exception:
                break

    threading.Thread(target=_tunnel_accept_loop, daemon=True).start()

    def cleanup() -> None:
        try:
            server.close()
        except OSError:
            pass
        ssh_client.close()

    return local_port, cleanup


def _cancel_job(run_id: str) -> int:
    """Cancel a running job by killing its tmux session."""
    import trio

    from bifrost import BifrostClient
    from broker.client import GPUClient
    from rollouts.jobs import get_job

    try:
        job = get_job(run_id)
    except AssertionError:
        print(f"Job not found: {run_id}")
        print("Use 'python -m argus monitor --runs' to list jobs")
        return 1

    if not job.nodes:
        print(f"No nodes found for job: {run_id}")
        return 1

    node = job.nodes[0]
    provider = node.provider
    node_id = node.node_id
    print(f"Cancelling job {run_id} on {provider}:{node_id}...")

    try:
        credentials = _broker_credentials()
        client = GPUClient(credentials=credentials)
        instance = trio.run(client.get_instance, node_id, provider)
        if not instance:
            print(f"Instance not found: {provider}:{node_id}")
            print("Instance may have been terminated.")
            return 1

        ssh_key = client.get_ssh_key_path(provider) or os.path.expanduser("~/.ssh/id_ed25519")
        ssh_connection = f"root@{instance.public_ip}:{instance.ssh_port}"
        bifrost = BifrostClient(ssh_connection, ssh_key_path=ssh_key)
        result = bifrost.exec(
            f"tmux kill-session -t bifrost-job-{run_id} 2>/dev/null && echo 'killed' || echo 'no session'"
        )
        output = result.stdout.strip() if result.stdout else ""

        if "killed" in output:
            print(f"Job cancelled: {run_id}")
            print(
                f"  Instance {provider}:{node_id} is still running (use 'broker terminate' to stop)"
            )
        else:
            print(f"No active tmux session found for job {run_id}")
            print("Job may have already completed or failed.")
        return 0
    except Exception as e:
        print(f"Failed to cancel job: {e}")
        return 1


def _fetch_and_print_logs_server_log(node_id: str | None, run_id: str) -> None:
    """Fetch logs_server.log from remote to help debug connection failures."""
    if not node_id:
        return

    try:
        import trio

        from bifrost import BifrostClient
        from broker.client import GPUClient

        provider, instance_id = node_id.split(":", 1)
        credentials = _broker_credentials()
        client = GPUClient(credentials=credentials)
        instance = trio.run(client.get_instance, instance_id, provider)
        if instance is None:
            print(f"Cannot fetch logs: instance {node_id} not found")
            return

        ssh_key = client.get_ssh_key_path(provider) or os.path.expanduser("~/.ssh/id_ed25519")
        ssh_connection = f"root@{instance.public_ip}:{instance.ssh_port}"
        bifrost = BifrostClient(ssh_connection, ssh_key_path=ssh_key)

        remote_log = (
            f"~/.bifrost/workspaces/rollouts-rl/rollouts/results/rl/{run_id}/logs_server.log"
        )
        result = bifrost.exec(f"cat {remote_log} 2>/dev/null || echo '[log file not found]'")

        print("\n--- logs_server.log from remote ---")
        print(result.stdout if result.stdout else "[empty]")
        print("--- end logs_server.log ---\n")

        print("--- tmux sessions ---")
        tmux_result = bifrost.exec("tmux ls 2>/dev/null || echo '[no tmux sessions]'")
        print(tmux_result.stdout if tmux_result.stdout else "[empty]")

        print("--- logs-server tmux pane content ---")
        pane_result = bifrost.exec(
            f"tmux capture-pane -t bifrost-job-logs-{run_id} -p 2>/dev/null || echo '[no pane]'"
        )
        print(pane_result.stdout if pane_result.stdout else "[empty]")

        print("--- port 9100 status ---")
        port_result = bifrost.exec("ss -tlnp | grep 9100 || echo '[not listening]'")
        print(port_result.stdout if port_result.stdout else "[empty]")
    except Exception as e:
        print(f"Failed to fetch remote logs: {e}")


def _run_attached(
    run_id: str | None,
    *,
    tail: bool = False,
    tail_lines: int | None = None,
    keep_alive: bool = False,
    terminate: bool = False,
    sync_only: bool = False,
) -> int:
    """Attach to a remote training run via LogsServer."""
    import time as _time

    from miniray import RemoteWorker

    if not tail and not sync_only and tail_lines is None:
        print(
            "Attached monitor TUI has been removed. Use `--tail`, `--tail-lines`, "
            "or `--sync-only`.",
            file=sys.stderr,
        )
        return 2

    run = _resolve_job_connection(run_id)
    resolved_run_id = run["run_id"]
    provider = run.get("provider")
    logs_host = run.get("logs_host")
    logs_port = run.get("logs_port")
    node_id = run.get("node_id")
    log_path = run.get("log_path")

    tunnel_cleanup: Callable[[], None] | None = None
    modal_sandbox = None
    modal_output_dir = None
    modal_files: tuple[str, ...] = ()

    if provider == "modal" and node_id:
        print(f"Attaching to run: {resolved_run_id}")
        print(f"Opening Modal sandbox {node_id}...")
        _, sandbox_id = node_id.split(":", 1)
        try:
            modal_sandbox = _get_modal_sandbox(sandbox_id)
            modal_output_dir = _resolve_modal_output_dir(modal_sandbox, resolved_run_id, log_path)
        except FileNotFoundError as exc:
            print(
                f"Run {resolved_run_id} is not attachable via Modal sandbox: {exc}",
                file=sys.stderr,
            )
            print(
                "The Modal sandbox may have already finished or been terminated.", file=sys.stderr
            )
            return 1
        print(f"Modal output dir: {modal_output_dir}")
    elif logs_host and logs_port:
        print(f"Attaching to run: {resolved_run_id}")
        print(f"LogsServer: {logs_host}:{logs_port} (direct TCP)")
    elif node_id:
        print(f"Attaching to run: {resolved_run_id}")
        print(f"Opening SSH tunnel to {node_id}...")
        local_port, tunnel_cleanup = _open_ssh_tunnel(node_id, remote_port=9100)
        logs_host = "127.0.0.1"
        logs_port = local_port
        print(f"LogsServer: localhost:{logs_port} (SSH tunnel)")
    else:
        print(f"Run {resolved_run_id} has no live instance — cannot attach")
        return 1

    worker = None
    available = {"files": []}
    if modal_sandbox is None:
        worker = RemoteWorker(logs_host, logs_port)
        max_retries = 10
        retry_delay = 1.0
        for attempt in range(max_retries):
            try:
                worker.connect()
                break
            except ConnectionRefusedError:
                if attempt == max_retries - 1:
                    _fetch_and_print_logs_server_log(node_id, resolved_run_id)
                    raise
                print(
                    f"LogsServer not ready, retrying in {retry_delay}s... ({attempt + 1}/{max_retries})"
                )
                _time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 5.0)

        worker.send({"cmd": "list"})
        available = worker.recv()

    local_sync_dir = Path("results/rl") / resolved_run_id
    local_sync_dir.mkdir(parents=True, exist_ok=True)

    global _MONITOR_LOG
    _MONITOR_LOG = local_sync_dir / "monitor.jsonl"
    _log("attach_start", run_id=resolved_run_id, node_id=node_id, files=available["files"])

    offsets_file = local_sync_dir / ".sync_offsets.json"
    offsets: dict[str, int] = {}
    if offsets_file.exists():
        try:
            offsets = json.loads(offsets_file.read_text())
            _log("offsets_loaded", files=offsets)
        except (json.JSONDecodeError, OSError):
            pass
    if modal_sandbox is not None:
        modal_files, _ = _sync_modal_files_once(
            sandbox=modal_sandbox,
            remote_output_dir=modal_output_dir,
            local_sync_dir=local_sync_dir,
            line_offsets=offsets,
        )
        available = {"files": list(modal_files)}
    print(f"Files: {', '.join(available['files'])}", file=sys.stderr)

    if tail_lines is not None:
        if modal_sandbox is not None:
            for filename in available.get("files", []):
                text = _read_modal_text_file(modal_sandbox, f"{modal_output_dir}/{filename}")
                if not text:
                    continue
                lines = text.splitlines()
                if not lines:
                    continue
                print(f"\n=== {filename} (last {tail_lines} lines) ===")
                for line in lines[-tail_lines:]:
                    print(line)
        else:
            assert worker is not None
            for filename in available.get("files", []):
                worker.send({"cmd": "tail", "file": filename, "offset": 0})
                result = worker.recv()
                lines = result.get("lines", [])
                if not lines:
                    continue
                print(f"\n=== {filename} (last {tail_lines} lines) ===")
                for line in lines[-tail_lines:]:
                    print(line)
            worker.close()
        if tunnel_cleanup:
            tunnel_cleanup()
        return 0

    sync_count = 0
    stop_sync = threading.Event()
    connection_lost = threading.Event()

    def sync_loop() -> None:
        nonlocal sync_count
        while not stop_sync.is_set():
            try:
                if modal_sandbox is not None:
                    files, total_new_lines = _sync_modal_files_once(
                        sandbox=modal_sandbox,
                        remote_output_dir=modal_output_dir,
                        local_sync_dir=local_sync_dir,
                        line_offsets=offsets,
                    )
                else:
                    assert worker is not None
                    worker.send({"cmd": "list"})
                    resp = worker.recv()
                    files = tuple(resp.get("files", []))

                    total_new_lines = 0
                    for filename in files:
                        offset = offsets.get(filename, 0)
                        worker.send({"cmd": "tail", "file": filename, "offset": offset})
                        result = worker.recv()
                        if result.get("error"):
                            _log("sync_error", file=filename, error=result.get("error"))
                            continue

                        new_lines = result.get("lines", [])
                        new_offset = result.get("offset", offset)
                        if new_lines:
                            local_path = local_sync_dir / filename
                            with open(local_path, "a", encoding="utf-8") as f:
                                for line in new_lines:
                                    f.write(line + "\n")
                            total_new_lines += len(new_lines)
                        offsets[filename] = new_offset

                sync_count += 1
                if sync_count <= 3 or total_new_lines > 0 or sync_count % 30 == 0:
                    _log("sync", count=sync_count, files=len(files), new_lines=total_new_lines)

                try:
                    offsets_file.write_text(json.dumps(offsets))
                except OSError:
                    pass
            except (EOFError, BrokenPipeError, ConnectionResetError, FileNotFoundError) as e:
                _log("sync_connection_lost", error=str(e))
                connection_lost.set()
                try:
                    with open(local_sync_dir / "training.log", "a", encoding="utf-8") as f:
                        f.write("[monitor] LogsServer connection lost\n")
                except OSError:
                    pass
                break

            stop_sync.wait(2.0)

    sync_thread = threading.Thread(target=sync_loop, daemon=True)
    sync_thread.start()
    _time.sleep(1.5)

    if sync_only:
        try:
            while not connection_lost.is_set():
                _time.sleep(2.0)
        except KeyboardInterrupt:
            pass
    elif tail:
        print(f"Tailing: {local_sync_dir}", file=sys.stderr)
        tail_offsets: dict[str, int] = {}
        for filename in available.get("files", []):
            local_path = local_sync_dir / filename
            if local_path.exists():
                with open(local_path, encoding="utf-8") as f:
                    for line in f:
                        sys.stdout.write(line)
                tail_offsets[filename] = local_path.stat().st_size
        try:
            while not connection_lost.is_set():
                for filename in available.get("files", []):
                    local_path = local_sync_dir / filename
                    if not local_path.exists():
                        continue
                    prev = tail_offsets.get(filename, 0)
                    cur = local_path.stat().st_size
                    if cur > prev:
                        with open(local_path, encoding="utf-8") as f:
                            f.seek(prev)
                            sys.stdout.write(f.read())
                            sys.stdout.flush()
                        tail_offsets[filename] = cur
                _time.sleep(1.0)
        except KeyboardInterrupt:
            pass

    stop_sync.set()
    sync_thread.join(timeout=5.0)
    if connection_lost.is_set():
        print(
            "\n[monitor] LogsServer connection lost (see training.log for details)", file=sys.stderr
        )

    print("\nFinal sync...")
    try:
        if modal_sandbox is not None:
            files, _ = _sync_modal_files_once(
                sandbox=modal_sandbox,
                remote_output_dir=modal_output_dir,
                local_sync_dir=local_sync_dir,
                line_offsets=offsets,
            )
            for filename in files:
                print(f"  Synced: {resolved_run_id}/{filename}")
        else:
            assert worker is not None
            worker.send({"cmd": "list"})
            resp = worker.recv()
            for filename in resp.get("files", []):
                offset = offsets.get(filename, 0)
                worker.send({"cmd": "tail", "file": filename, "offset": offset})
                result = worker.recv()
                new_lines = result.get("lines", [])
                new_offset = result.get("offset", offset)
                if new_lines:
                    local_path = local_sync_dir / filename
                    with open(local_path, "a", encoding="utf-8") as f:
                        for line in new_lines:
                            f.write(line + "\n")
                    print(f"  Synced: {resolved_run_id}/{filename} (+{len(new_lines)} lines)")
                offsets[filename] = new_offset
        try:
            offsets_file.write_text(json.dumps(offsets))
        except OSError:
            pass
    except (EOFError, BrokenPipeError, ConnectionResetError, FileNotFoundError):
        print("  LogsServer disconnected, skipping final sync")

    if worker is not None:
        worker.close()
    if tunnel_cleanup is not None:
        tunnel_cleanup()

    if node_id:
        should_terminate = False
        if terminate:
            should_terminate = True
        elif keep_alive:
            should_terminate = False
            print(f"\nInstance kept alive: {node_id}")
            print(f"Reattach: python -m argus monitor --attach {resolved_run_id}")
        else:
            answer = input(f"\nTerminate instance {node_id}? [y/n] ").strip().lower()
            should_terminate = answer == "y"
            if not should_terminate:
                print(f"Instance kept alive: {node_id}")
                print(f"Reattach: python -m argus monitor --attach {resolved_run_id}")

        if should_terminate:
            provider_name, instance_id = node_id.split(":", 1)
            print(f"Terminating {node_id}...")
            if provider_name == "modal":
                _get_modal_sandbox(instance_id).terminate()
            else:
                import trio

                from broker.client import GPUClient

                credentials = _broker_credentials()

                async def _do_terminate() -> None:
                    client = GPUClient(credentials=credentials)
                    inst = await client.get_instance(instance_id, provider_name)
                    assert inst is not None, f"Instance not found: {node_id}"
                    await inst.terminate()

                trio.run(_do_terminate)
            print("Terminated.")

    return 0


def _find_latest_run_dir(base_dir: str) -> Path | None:
    """Resolve the most recent run directory under a candidate base dir."""
    base = Path(base_dir)
    if not base.is_dir():
        return None

    candidates: list[Path] = []
    for child in base.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            candidates.append(child)
            for grandchild in child.iterdir():
                if grandchild.is_dir() and not grandchild.name.startswith("."):
                    candidates.append(grandchild)

    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_local_view_run_dir(args: argparse.Namespace) -> Path:
    """Resolve the local run directory for non-attach viewing/tailing."""
    if args.attach is not None:
        raise ValueError("_resolve_local_view_run_dir only supports local runs")

    if args.latest:
        base_dir = args.output_dir or "results"
        resolved = _find_latest_run_dir(base_dir)
        if resolved is None:
            raise ValueError(f"No run directories found under {base_dir!r}")
        return resolved

    if args.output_dir is None:
        raise ValueError("A run directory or --latest is required")

    return Path(args.output_dir)


def _resolve_wait_run_dir(args: argparse.Namespace) -> Path:
    """Resolve a local run directory for wait-mode polling."""
    if args.attach is not None:
        raise ValueError("--wait-for-event does not support --attach yet")

    if args.latest:
        if args.output_dir is not None:
            resolved = _find_latest_run_dir(args.output_dir)
            if resolved is None:
                raise ValueError(f"No run directories found under {args.output_dir!r}")
            return resolved

        candidate_runs = [
            resolved
            for resolved in (
                _find_latest_run_dir("results"),
                _find_latest_run_dir("rollouts/results"),
            )
            if resolved is not None
        ]
        if not candidate_runs:
            raise ValueError("No run directories found under 'results' or 'rollouts/results'")
        candidate_runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidate_runs[0]

    if args.output_dir is None:
        raise ValueError("--wait-for-event requires a run directory or --latest")

    return Path(args.output_dir)


def _event_is_failure(entry: dict[str, Any], fail_on_events: set[str]) -> bool:
    """Return whether a structured journal entry denotes terminal failure."""
    event = entry.get("event")
    if not isinstance(event, str):
        return False

    if event in fail_on_events:
        return True

    if event == "remote_exit_observed":
        exit_code = entry.get("exit_code")
        return isinstance(exit_code, int) and exit_code != 0

    return False


def _wait_for_run_event(
    run_dir: Path,
    *,
    target_event: str,
    timeout_seconds: float | None,
    poll_interval: float,
    fail_on_events: set[str],
) -> int:
    """Poll a local run journal until a target event or terminal failure occurs."""
    run_jsonl = run_dir / "control.jsonl"
    if not run_jsonl.exists():
        run_jsonl = run_dir / "run.jsonl"
    start = time.monotonic()
    offset = 0

    print(
        f"Waiting for event {target_event!r} in {run_jsonl}"
        + (f" (timeout {timeout_seconds:.1f}s)" if timeout_seconds is not None else "")
    )

    while True:
        if timeout_seconds is not None and (time.monotonic() - start) > timeout_seconds:
            print(
                f"Timed out waiting for {target_event!r} in {run_jsonl} after {timeout_seconds:.1f}s",
            )
            return 124

        if run_jsonl.exists():
            with open(run_jsonl, encoding="utf-8") as handle:
                handle.seek(offset)
                while True:
                    line = handle.readline()
                    if not line:
                        break
                    offset = handle.tell()
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event = entry.get("event")
                    if event == target_event:
                        print(json.dumps(entry, sort_keys=True))
                        return 0

                    if _event_is_failure(entry, fail_on_events):
                        print(json.dumps(entry, sort_keys=True))
                        return 1

        time.sleep(poll_interval)


def monitor_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="argus monitor",
        description="Inspect local journals and attach to remote Argus-managed runs",
    )
    parser.add_argument("output_dir", nargs="?", help="Run directory to inspect")
    parser.add_argument("--latest", action="store_true", help="Use the most recent run directory")
    parser.add_argument(
        "--attach",
        nargs="?",
        const="__latest__",
        metavar="RUN_ID",
        help="Resolve/sync a remote run from ~/.rollouts/jobs.json",
    )
    parser.add_argument("--runs", action="store_true", help="List known runs from local registry")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="With --runs, check broker liveness and LogsServer reachability",
    )
    parser.add_argument(
        "--launches",
        action="store_true",
        help="List active local Argus launcher records",
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help="Tail remote/local logs to stdout",
    )
    parser.add_argument(
        "--format",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format when tailing: pretty (human-readable, default) or json (raw JSONL)",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=None,
        metavar="N",
        help="With --tail or --attach, print the last N lines and exit",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep instance running after an attached remote run completes",
    )
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="Auto-terminate instance after an attached remote run completes",
    )
    parser.add_argument(
        "--cancel",
        metavar="RUN_ID",
        help="Cancel a running remote job by killing its tmux session",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="With --attach: sync logs locally without launching a viewer",
    )
    parser.add_argument(
        "--wait-for-event",
        metavar="EVENT",
        help="Poll run.jsonl until this event appears, then exit",
    )
    parser.add_argument(
        "--fail-on-event",
        action="append",
        default=None,
        metavar="EVENT",
        help="Treat this event as terminal failure while waiting (repeatable)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Timeout for --wait-for-event",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Polling interval for --wait-for-event",
    )
    args = parser.parse_args(argv)

    import logging

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(logging.NullHandler())

    if args.launches:
        return _print_launches()

    if args.cancel:
        return _cancel_job(args.cancel)

    if args.runs:
        return _print_runs(probe=args.probe)

    if args.wait_for_event is not None:
        fail_on_events = set(_DEFAULT_WAIT_FAILURE_EVENTS)
        if args.fail_on_event is not None:
            fail_on_events.update(args.fail_on_event)
        run_dir = _resolve_wait_run_dir(args)
        return _wait_for_run_event(
            run_dir,
            target_event=args.wait_for_event,
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
            fail_on_events=fail_on_events,
        )

    if args.attach is not None:
        attach_run_id = None if args.attach == "__latest__" or args.latest else args.attach
        return _run_attached(
            run_id=attach_run_id,
            tail=args.tail,
            tail_lines=args.tail_lines,
            keep_alive=args.keep_alive,
            terminate=args.terminate,
            sync_only=args.sync_only,
        )

    if args.attach is None and args.tail:
        run_dir = _resolve_local_view_run_dir(args)
        if not run_dir.is_dir():
            print(f"Error: {run_dir} is not a directory")
            return 1
        return tail_run(run_dir, fmt=args.format)

    if args.attach is None and (args.latest or args.output_dir is not None):
        print(
            "Local monitor TUI has been removed. Use `--tail` for streaming output "
            "or `--wait-for-event` for polling.",
        )
        return 2

    parser.error(
        "Specify a local run directory, `--latest`, `--attach`, `--runs`, or `--launches`."
    )
