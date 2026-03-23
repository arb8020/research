"""Helpers for launching detached remote services over SSH."""

from __future__ import annotations

import json
import shlex


def build_detached_service_launch_command(
    *,
    full_cmd: str,
    stdout_log_file: str,
    stderr_log_file: str,
    pid_file: str,
) -> str:
    """Build a remote command that launches a detached service and returns."""

    launcher = (
        "import pathlib, subprocess; "
        f"stdout_path = pathlib.Path({json.dumps(stdout_log_file)}); "
        f"stderr_path = pathlib.Path({json.dumps(stderr_log_file)}); "
        f"pid_path = pathlib.Path({json.dumps(pid_file)}); "
        "stdout_path.parent.mkdir(parents=True, exist_ok=True); "
        "stdout = stdout_path.open('ab'); "
        "stderr = stderr_path.open('ab'); "
        f"proc = subprocess.Popen(['bash', '-lc', {json.dumps(full_cmd)}], "
        "stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, "
        "start_new_session=True, close_fds=True); "
        "stdout.close(); "
        "stderr.close(); "
        "pid_path.write_text(str(proc.pid))"
    )
    return f"python3 -c {shlex.quote(launcher)}"
