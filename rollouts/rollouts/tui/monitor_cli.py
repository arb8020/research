"""Monitor subcommand for rollouts CLI.

Usage:
    rollouts monitor results/rl/run_20250127/       # Watch specific run
    rollouts monitor --latest                        # Watch most recent run in results/
    rollouts monitor --latest results/sft/           # Most recent in a custom dir
    rollouts monitor --attach run_20250127-143052    # Attach to remote run by ID
    rollouts monitor --attach --latest               # Attach to most recent active run
    rollouts monitor --runs                          # List jobs from ~/.rollouts/jobs.json
    rollouts monitor --runs --probe                  # + check broker liveness & LogsServer
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
from collections.abc import Callable
from pathlib import Path


def _broker_credentials() -> dict[str, str]:
    """Load broker credentials from environment."""
    credentials: dict[str, str] = {}
    for env_key, provider in [
        ("RUNPOD_API_KEY", "runpod"),
        ("WAFER_RUNPOD_API_KEY", "runpod"),
    ]:
        val = os.environ.get(env_key)
        if val and provider not in credentials:
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


def _get_live_instance_ids() -> set[str]:
    """Query broker for all live instance IDs."""
    import trio

    from broker.client import GPUClient

    credentials = _broker_credentials()
    if not credentials:
        return set()

    async def _fetch() -> set[str]:
        client = GPUClient(credentials=credentials)
        instances = await client.list_instances()
        return {inst.id for inst in instances}

    try:
        return trio.run(_fetch)
    except Exception as e:
        print(f"Warning: could not query broker: {e}", file=sys.stderr)
        return set()


def _resolve_logs_endpoint(instance: object) -> tuple[str | None, int | None]:
    """Extract LogsServer host:port from instance runtime port mapping.

    RunPod maps container ports to random public ports.
    Returns (host, public_port) or (None, None) if not found.

    NOTE: Don't fall back to public_ip:9100 - RunPod doesn't expose arbitrary
    container ports. If port 9100 isn't in runtime_ports, use SSH tunnel instead.
    """
    raw = instance.raw_data or {}
    runtime = raw.get("runtime") or {}
    runtime_ports = runtime.get("ports") or []

    for p in runtime_ports:
        if p.get("privatePort") == 9100 and p.get("isIpPublic"):
            return p["ip"], p["publicPort"]

    # No direct port mapping - caller should use SSH tunnel
    return None, None


def find_latest_run(base_dir: str = "results") -> Path | None:
    """Find the most recent run directory across all experiment types.

    Searches results/, results/rl/, results/sft/, results/eval/, etc.
    """
    base = Path(base_dir)
    if not base.is_dir():
        return None

    # Collect candidate dirs: immediate children + one level deeper
    candidates: list[Path] = []
    for child in base.iterdir():
        if child.is_dir():
            if child.name.startswith("."):
                continue
            candidates.append(child)
            # Also check subdirs (e.g. results/rl/run_20250127/)
            for grandchild in child.iterdir():
                if grandchild.is_dir() and not grandchild.name.startswith("."):
                    candidates.append(grandchild)

    if not candidates:
        return None

    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_job_connection(job_id: str | None) -> dict:
    """Resolve a job to its LogsServer connection info.

    Reads job→node mapping from ~/.rollouts/jobs.json,
    then queries broker for live port data.

    Returns dict with: run_id, node_id, logs_host, logs_port.
    """
    from dotenv import load_dotenv

    from rollouts.jobs import get_job, get_latest_job

    load_dotenv()

    job = get_latest_job() if job_id is None else get_job(job_id)

    # For now, connect to the first node (single-node jobs).
    # Multi-node: would pick the rank-0 / training node.
    assert job.nodes, f"Job {job.job_id} has no nodes"
    node = job.nodes[0]

    # Query broker for live instance data (ports, IPs)
    instance = _get_instance(node.provider, node.node_id)
    node_id_str = f"{node.provider}:{node.node_id}"

    if instance is None:
        return {
            "run_id": job.job_id,
            "node_id": node_id_str,
            "logs_host": None,
            "logs_port": None,
        }

    logs_host, logs_port = _resolve_logs_endpoint(instance)
    return {
        "run_id": job.job_id,
        "node_id": node_id_str,
        "logs_host": logs_host,
        "logs_port": logs_port,
    }


def _open_ssh_tunnel(
    node_id: str,
    remote_port: int = 9100,
) -> tuple[int, Callable[[], None]]:
    """Open an SSH tunnel to a remote LogsServer.

    For instances provisioned without exposed_ports — we tunnel through SSH
    to reach LogsServer on the remote's localhost.

    Returns (local_port, cleanup_fn). Connect RemoteWorker to localhost:local_port.
    """
    import paramiko

    from broker.client import GPUClient

    provider, instance_id = node_id.split(":", 1)

    from dotenv import load_dotenv

    load_dotenv()

    credentials = _broker_credentials()
    client = GPUClient(credentials=credentials)

    import trio

    instance = trio.run(client.get_instance, instance_id, provider)
    assert instance is not None, f"Instance not found: {node_id}"

    ssh_key = client.get_ssh_key_path(provider) or os.path.expanduser("~/.ssh/id_ed25519")

    # Connect paramiko
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

    # Bind local socket on ephemeral port
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
            except OSError:
                pass
            try:
                dst.close()
            except OSError:
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

    tunnel_thread = threading.Thread(target=_tunnel_accept_loop, daemon=True)
    tunnel_thread.start()

    def cleanup() -> None:
        try:
            server.close()
        except OSError:
            pass
        ssh_client.close()

    return local_port, cleanup


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

        # Try to read the logs_server.log
        remote_log = f"~/.bifrost/workspaces/rollouts-rl/rollouts/results/rl/{run_id}/logs_server.log"
        result = bifrost.exec(f"cat {remote_log} 2>/dev/null || echo '[log file not found]'")

        print("\n--- logs_server.log from remote ---")
        print(result.stdout if result.stdout else "[empty]")
        print("--- end logs_server.log ---\n")

        # Also check tmux sessions and try to diagnose
        print("--- tmux sessions ---")
        tmux_result = bifrost.exec("tmux ls 2>/dev/null || echo '[no tmux sessions]'")
        print(tmux_result.stdout if tmux_result.stdout else "[empty]")

        # Capture the tmux pane content to see what's happening
        print("--- logs-server tmux pane content ---")
        pane_result = bifrost.exec(
            "tmux capture-pane -t bifrost-job-logs-server -p 2>/dev/null || echo '[no pane]'"
        )
        print(pane_result.stdout if pane_result.stdout else "[empty]")

        # Check if port 9100 is listening
        print("--- port 9100 status ---")
        port_result = bifrost.exec("ss -tlnp | grep 9100 || echo '[not listening]'")
        print(port_result.stdout if port_result.stdout else "[empty]")

    except Exception as e:
        print(f"Failed to fetch remote logs: {e}")


def _run_attached(run_id: str | None) -> int:
    """Attach to a remote training run via LogsServer.

    Two transport modes:
      - Direct TCP: If logs_host + logs_port resolved from broker runtime ports.
      - SSH tunnel: If only node_id (instances without exposed_ports).
        Opens paramiko tunnel to remote localhost:9100.

    Flow:
    1. Resolve job→node from ~/.rollouts/jobs.json
    2. Query broker for live port data
    3. Connect to LogsServer (direct or via SSH tunnel)
    4. Start background sync thread (tail commands every 2s)
    5. Launch monitor TUI watching the local sync dir
    6. On quit: final sync, optionally terminate instance
    """
    import time

    from miniray import RemoteWorker

    from .rlmon import make_app

    run = _resolve_job_connection(run_id)
    resolved_run_id = run["run_id"]
    logs_host = run.get("logs_host")
    logs_port = run.get("logs_port")
    node_id = run.get("node_id")

    tunnel_cleanup: Callable[[], None] | None = None

    if logs_host and logs_port:
        # Direct TCP — instance was provisioned with exposed_ports
        print(f"Attaching to run: {resolved_run_id}")
        print(f"LogsServer: {logs_host}:{logs_port} (direct TCP)")
    elif node_id:
        # SSH tunnel — instance without exposed_ports
        print(f"Attaching to run: {resolved_run_id}")
        print(f"Opening SSH tunnel to {node_id}...")
        local_port, tunnel_cleanup = _open_ssh_tunnel(node_id, remote_port=9100)
        logs_host = "127.0.0.1"
        logs_port = local_port
        print(f"LogsServer: localhost:{logs_port} (SSH tunnel)")
    else:
        print(f"Run {resolved_run_id} has no live instance — cannot attach")
        return 1

    worker = RemoteWorker(logs_host, logs_port)

    # Retry connection - LogsServer may still be starting
    max_retries = 10
    retry_delay = 1.0
    for attempt in range(max_retries):
        try:
            worker.connect()
            break
        except ConnectionRefusedError:
            if attempt == max_retries - 1:
                # Try to fetch logs_server.log to see what went wrong
                _fetch_and_print_logs_server_log(node_id, resolved_run_id)
                raise
            print(
                f"LogsServer not ready, retrying in {retry_delay}s... ({attempt + 1}/{max_retries})"
            )
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 1.5, 5.0)

    # Discover available files
    worker.send({"cmd": "list"})
    available = worker.recv()
    print(f"Files: {', '.join(available['files'])}")

    local_sync_dir = Path("results/rl") / resolved_run_id
    local_sync_dir.mkdir(parents=True, exist_ok=True)

    # Track byte offsets per file for incremental tail
    offsets: dict[str, int] = {}

    stop_sync = threading.Event()

    def sync_loop() -> None:
        while not stop_sync.is_set():
            try:
                worker.send({"cmd": "list"})
                resp = worker.recv()
                files = resp.get("files", [])

                for filename in files:
                    offset = offsets.get(filename, 0)
                    worker.send({"cmd": "tail", "file": filename, "offset": offset})
                    result = worker.recv()

                    if result.get("error"):
                        continue

                    new_lines = result.get("lines", [])
                    new_offset = result.get("offset", offset)

                    if new_lines:
                        local_path = local_sync_dir / filename
                        with open(local_path, "a") as f:
                            for line in new_lines:
                                f.write(line + "\n")

                    offsets[filename] = new_offset

            except (EOFError, BrokenPipeError, ConnectionResetError):
                print("[monitor] LogsServer connection lost")
                break

            stop_sync.wait(2.0)

    sync_thread = threading.Thread(target=sync_loop, daemon=True)
    sync_thread.start()

    # Give first sync a moment to populate files
    time.sleep(1.5)

    print(f"Watching: {local_sync_dir}")
    app = make_app(str(local_sync_dir))
    app.run()

    # TUI exited — final sync
    stop_sync.set()
    sync_thread.join(timeout=5.0)

    print("\nFinal sync...")
    try:
        worker.send({"cmd": "list"})
        resp = worker.recv()
        for filename in resp.get("files", []):
            offset = offsets.get(filename, 0)
            worker.send({"cmd": "tail", "file": filename, "offset": offset})
            result = worker.recv()
            new_lines = result.get("lines", [])
            if new_lines:
                local_path = local_sync_dir / filename
                with open(local_path, "a") as f:
                    for line in new_lines:
                        f.write(line + "\n")
                print(f"  Synced: {resolved_run_id}/{filename} (+{len(new_lines)} lines)")
    except (EOFError, BrokenPipeError, ConnectionResetError):
        print("  LogsServer disconnected, skipping final sync")

    worker.close()

    if tunnel_cleanup is not None:
        tunnel_cleanup()

    # Optional: terminate instance
    if node_id:
        answer = input(f"\nTerminate instance {node_id}? [y/n] ").strip().lower()
        if answer == "y":
            import trio

            from broker.client import GPUClient

            credentials = _broker_credentials()
            provider, instance_id = node_id.split(":", 1)

            async def _terminate() -> None:
                client = GPUClient(credentials=credentials)
                inst = await client.get_instance(instance_id, provider)
                assert inst is not None, f"Instance not found: {node_id}"
                await inst.terminate()

            print(f"Terminating {node_id}...")
            trio.run(_terminate)
            print("Terminated.")
        else:
            print(f"Instance kept alive: {node_id}")
            print(f"Reattach: rollouts monitor --attach {resolved_run_id}")

    return 0


def monitor_main(argv: list[str] | None = None) -> int:
    """Entry point for `rollouts monitor` subcommand."""
    parser = argparse.ArgumentParser(
        prog="rollouts monitor",
        description="btop-style experiment monitor (RL, SFT, eval, generic)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Path to training/eval output directory",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Watch the most recent run directory",
    )
    parser.add_argument(
        "--attach",
        nargs="?",
        const="__latest__",
        metavar="RUN_ID",
        help="Attach to remote run by ID (e.g. run_20250127-143052). No value = latest.",
    )
    parser.add_argument(
        "--runs",
        action="store_true",
        help="List jobs from ~/.rollouts/jobs.json",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="With --runs, check broker liveness and probe LogsServer",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dump frame layout snapshots to /tmp/rlmon-debug.jsonl every ~5s",
    )
    parser.add_argument(
        "--debug-interval",
        type=int,
        default=100,
        metavar="N",
        help="Dump debug snapshot every N frames (default: 100, ~5s at 20fps)",
    )
    args = parser.parse_args(argv)

    # ── List mode ──
    if args.runs:
        from dotenv import load_dotenv

        from rollouts.jobs import list_jobs, prune_jobs

        load_dotenv()

        jobs = list_jobs()
        if not jobs:
            print("No jobs found. Run a training job first.")
            return 0

        # When probing, query broker for live instances and prune dead jobs
        live_ids: set[str] | None = None
        if args.probe:
            live_ids = _get_live_instance_ids()

        header = f"{'JOB ID':<30} {'NODE':<25} {'SCRIPT':<35} {'STARTED':<20}"
        if args.probe:
            header += f" {'STATUS':<10}"
        print(header)
        print("-" * len(header))

        for job in jobs:
            node_str = ", ".join(job.node_ids) if job.nodes else "?"
            script = job.script
            # Truncate long paths
            if len(script) > 33:
                script = "..." + script[-30:]
            started = job.started_at[:19] if len(job.started_at) >= 19 else job.started_at

            row = f"{job.job_id:<30} {node_str:<25} {script:<35} {started:<20}"

            if args.probe and live_ids is not None:
                # Check if any node is still alive
                has_live = any(n.node_id in live_ids for n in job.nodes)
                if not has_live:
                    continue  # skip dead jobs (will be pruned below)

                # Probe LogsServer on first node
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

        # Prune dead jobs
        if args.probe and live_ids is not None:
            pruned = prune_jobs(live_ids)
            if pruned > 0:
                print(f"\nPruned {pruned} dead job(s)")

        return 0

    # ── Attach mode ──
    if args.attach is not None:
        if args.attach == "__latest__" or args.latest:
            return _run_attached(run_id=None)
        else:
            return _run_attached(run_id=args.attach)

    # ── Local mode ──
    from .rlmon import make_app

    if args.latest:
        base = args.output_dir or "results"
        latest = find_latest_run(base)
        if latest is None:
            print(f"No run directories found in {base}", file=sys.stderr)
            return 1
        output_dir = latest
        print(f"Watching: {output_dir}")
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        parser.print_help()
        return 1

    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory", file=sys.stderr)
        return 1

    app = make_app(str(output_dir), debug=args.debug, debug_frame_interval=args.debug_interval)
    app.run()
    return 0
