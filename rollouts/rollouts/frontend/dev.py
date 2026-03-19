from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SESSION_NAME = "rollouts-webui"


def _port_in_use(port: int) -> bool:
    result = subprocess.run(
        ["lsof", "-ti", f":{port}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def _pids_on_port(port: int) -> list[str]:
    result = subprocess.run(
        ["lsof", "-ti", f":{port}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _resolve_port(port: int, port_range: str | None, kill_port: bool) -> int:
    if port_range:
        start_s, sep, end_s = port_range.partition("-")
        if not sep:
            raise ValueError(f"Invalid --port-range: {port_range!r}")
        start = int(start_s)
        end = int(end_s)
        for candidate in range(start, end + 1):
            if not _port_in_use(candidate):
                return candidate
        raise ValueError(f"No free port found in range {port_range}")

    if _port_in_use(port):
        if not kill_port:
            raise ValueError(
                f"Port {port} is already in use. Pass --kill-port or --port-range START-END."
            )
        pids = _pids_on_port(port)
        if pids:
            print(f"Killing process(es) on port {port}: {' '.join(pids)}")
            subprocess.run(["kill", "-9", *pids], check=False)
            time.sleep(0.3)
    return port


def _server_command(
    *,
    repo_root: Path,
    project: Path,
    port: int,
    results_dirs: list[Path],
    log_dir: Path,
) -> str:
    cmd = [
        sys.executable,
        "-m",
        "rollouts.frontend.server",
        "--project",
        str(project),
        "--port",
        str(port),
        "--no-browser",
    ]
    if results_dirs:
        cmd.append("--results-dirs")
        cmd.extend(str(path) for path in results_dirs)
    log_path = log_dir / "server.log"
    return "; ".join([
        f"cd {shlex.quote(str(repo_root))}",
        f"mkdir -p {shlex.quote(str(log_dir))}",
        f"{shlex.join(cmd)} 2>&1 | tee {shlex.quote(str(log_path))}",
        "echo '[server exited]'",
        "read",
    ])


def _watcher_command(*, ui_dir: Path, log_dir: Path) -> str:
    log_path = log_dir / "ui-build.log"
    return "; ".join([
        f"cd {shlex.quote(str(ui_dir))}",
        f"mkdir -p {shlex.quote(str(log_dir))}",
        f"npm run build:watch 2>&1 | tee {shlex.quote(str(log_path))}",
        "echo '[watcher exited]'",
        "read",
    ])


def _tail_command(log_dir: Path) -> str:
    server_log = log_dir / "server.log"
    ui_log = log_dir / "ui-build.log"
    return "; ".join([
        f"mkdir -p {shlex.quote(str(log_dir))}",
        f"touch {shlex.quote(str(server_log))} {shlex.quote(str(ui_log))}",
        f"tail -F {shlex.quote(str(server_log))} {shlex.quote(str(ui_log))}",
        "read",
    ])


def _tmux_available() -> bool:
    return shutil.which("tmux") is not None


def _start_tmux(
    *,
    repo_root: Path,
    ui_dir: Path,
    project: Path,
    port: int,
    results_dirs: list[Path],
    log_dir: Path,
    attach: bool,
) -> int:
    has_session = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if has_session.returncode == 0:
        print(f"Session {SESSION_NAME!r} already running.")
        print(f"  Attach: tmux attach -t {SESSION_NAME}")
        print(f"  Kill:   tmux kill-session -t {SESSION_NAME}")
        return 0

    print("Building UI once before starting watcher...")
    subprocess.run(["npm", "run", "build"], cwd=ui_dir, check=True, stdout=subprocess.DEVNULL)

    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            SESSION_NAME,
            "-n",
            "webui",
            "-c",
            str(repo_root),
            _server_command(
                repo_root=repo_root,
                project=project,
                port=port,
                results_dirs=results_dirs,
                log_dir=log_dir,
            ),
        ],
        check=True,
    )
    subprocess.run(
        [
            "tmux",
            "split-window",
            "-t",
            f"{SESSION_NAME}:0",
            "-h",
            "-c",
            str(repo_root),
            _watcher_command(ui_dir=ui_dir, log_dir=log_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            "tmux",
            "split-window",
            "-t",
            f"{SESSION_NAME}:0.1",
            "-v",
            "-c",
            str(repo_root),
            _tail_command(log_dir),
        ],
        check=True,
    )
    subprocess.run(
        ["tmux", "select-layout", "-t", f"{SESSION_NAME}:0", "main-vertical"], check=True
    )
    subprocess.run(["tmux", "resize-pane", "-t", f"{SESSION_NAME}:0.0", "-x", "60%"], check=True)
    subprocess.run(["tmux", "select-pane", "-t", f"{SESSION_NAME}:0.0"], check=True)

    print(f"Session {SESSION_NAME!r} started.")
    print(f"  URL:    http://localhost:{port}")
    print(f"  Attach: tmux attach -t {SESSION_NAME}")
    print(f"  Kill:   tmux kill-session -t {SESSION_NAME}")

    if attach and "TMUX" not in os.environ:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME], check=False)
    return 0


def _start_foreground(
    *,
    repo_root: Path,
    ui_dir: Path,
    project: Path,
    port: int,
    results_dirs: list[Path],
) -> int:
    print("tmux not found; falling back to foreground mode.", file=sys.stderr)
    server_cmd = [
        sys.executable,
        "-m",
        "rollouts.frontend.server",
        "--project",
        str(project),
        "--port",
        str(port),
        "--no-browser",
    ]
    if results_dirs:
        server_cmd.append("--results-dirs")
        server_cmd.extend(str(path) for path in results_dirs)

    server_proc = subprocess.Popen(server_cmd, cwd=repo_root)
    watcher_proc = subprocess.Popen(["npm", "run", "build:watch"], cwd=ui_dir)
    try:
        return server_proc.wait()
    finally:
        for proc in (server_proc, watcher_proc):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rollouts webui tmux dev launcher")
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--port-range")
    parser.add_argument("--kill-port", action="store_true")
    parser.add_argument("--results-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--attach", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    ui_dir = repo_root / "rollouts" / "frontend" / "ui"
    log_dir = repo_root / "logs" / "webui"
    project = args.project.expanduser().resolve()
    results_dirs = [path.expanduser().resolve() for path in args.results_dirs]
    port = _resolve_port(args.port, args.port_range, args.kill_port)

    if _tmux_available():
        return _start_tmux(
            repo_root=repo_root,
            ui_dir=ui_dir,
            project=project,
            port=port,
            results_dirs=results_dirs,
            log_dir=log_dir,
            attach=args.attach,
        )
    return _start_foreground(
        repo_root=repo_root,
        ui_dir=ui_dir,
        project=project,
        port=port,
        results_dirs=results_dirs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
