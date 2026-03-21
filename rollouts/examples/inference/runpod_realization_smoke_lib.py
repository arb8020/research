from __future__ import annotations

import json
import os
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_UV_BIN = "~/.local/bin/uv"
_REMOTE_UV_BIN = "/root/.local/bin/uv"
_JSON_SENTINEL = "__RUNPOD_SMOKE_JSON__"


@dataclass(frozen=True)
class RunPodSmokeSpec:
    name: str
    launch_module: str
    model_flag: str
    default_model: str
    default_gpu_type: str
    port: int
    bootstrap_steps: tuple[tuple[str, str], ...]
    env: dict[str, str]
    probe_weight_update_schema: bool = False
    probe_generation: bool = True
    launch_with_uv: bool = True
    remote_python_bin: str | None = None


def _stdout(result: Any) -> str:
    if hasattr(result, "stdout"):
        return str(result.stdout)
    return str(result)


def _stderr(result: Any) -> str:
    if hasattr(result, "stderr"):
        return str(result.stderr)
    return ""


def _exit_code(result: Any) -> int | None:
    if hasattr(result, "exit_code"):
        value = getattr(result, "exit_code")
        if isinstance(value, int):
            return value
    return None


def _run_remote_checked(
    bifrost: Any,
    command: str,
    *,
    working_dir: str,
    timeout: float | None = None,
) -> str:
    result = bifrost.exec(command, working_dir=working_dir, timeout=timeout)
    exit_code = _exit_code(result)
    if exit_code not in (None, 0):
        raise RuntimeError(
            f"Remote command failed with exit_code={exit_code}: {command}\n"
            f"stdout:\n{_stdout(result)}\n"
            f"stderr:\n{_stderr(result)}"
        )
    return _stdout(result)


def _run_remote_json(
    bifrost: Any,
    script: str,
    *,
    working_dir: str,
    python_bin: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    payload = dedent(script).strip()
    if python_bin is None:
        command = (
            f"{_UV_BIN} run python - <<'PY'\n"
            f"{payload}\n"
            "PY"
        )
    else:
        command = (
            f"{python_bin} - <<'PY'\n"
            f"{payload}\n"
            "PY"
        )
    stdout = _run_remote_checked(
        bifrost,
        command,
        working_dir=working_dir,
        timeout=timeout,
    )
    for line in reversed(stdout.splitlines()):
        if line.startswith(_JSON_SENTINEL):
            return json.loads(line.removeprefix(_JSON_SENTINEL))
    raise RuntimeError(f"Remote JSON sentinel missing from stdout:\n{stdout}")


def _wait_for_http_ready(
    bifrost: Any,
    *,
    url: str,
    working_dir: str,
    timeout_seconds: int,
    log_path: str,
    tmux_session: str | None = None,
) -> str:
    last_stdout = ""
    for attempt in range(timeout_seconds):
        result = bifrost.exec(f"curl -fsS {shlex.quote(url)}", working_dir=working_dir, timeout=10)
        if _exit_code(result) == 0:
            return _stdout(result)
        last_stdout = _stdout(result)
        time.sleep(1)
        if attempt > 0 and attempt % 20 == 0:
            tail = bifrost.exec(
                f"tail -n 40 {shlex.quote(log_path)} 2>/dev/null || true",
                working_dir=working_dir,
                timeout=10,
            )
            pane = None
            if tmux_session is not None:
                pane = bifrost.exec(
                    "tmux capture-pane "
                    f"-pt {shlex.quote(tmux_session)} -S -80 2>/dev/null || true",
                    working_dir=working_dir,
                    timeout=10,
                )
            print(f"[wait:{attempt}s] still waiting for {url}")
            print(_stdout(tail).strip())
            if pane is not None:
                print(_stdout(pane).strip())
    tail = bifrost.exec(
        f"tail -n 80 {shlex.quote(log_path)} 2>/dev/null || true",
        working_dir=working_dir,
        timeout=10,
    )
    pane = None
    if tmux_session is not None:
        pane = bifrost.exec(
            "tmux capture-pane "
            f"-pt {shlex.quote(tmux_session)} -S -120 2>/dev/null || true",
            working_dir=working_dir,
            timeout=10,
        )
    raise TimeoutError(
        f"Timed out waiting for {url}\n"
        f"last stdout:\n{last_stdout}\n"
        f"log tail:\n{_stdout(tail)}"
        f"\ntmux pane:\n{_stdout(pane) if pane is not None else ''}"
    )


def _probe_generation(
    bifrost: Any,
    *,
    working_dir: str,
    base_url: str,
    model: str,
    python_bin: str | None = None,
) -> dict[str, Any]:
    script = f"""
import json
import urllib.request

payload = json.dumps({{
    "model": {model!r},
    "messages": [
        {{
            "role": "user",
            "content": "Reply with exactly: smoke",
        }}
    ],
    "temperature": 0.0,
    "max_tokens": 8,
}}).encode("utf-8")

request = urllib.request.Request(
    {f"{base_url}/v1/chat/completions"!r},
    data=payload,
    headers={{"Content-Type": "application/json"}},
    method="POST",
)
with urllib.request.urlopen(request, timeout=60) as response:
    body = json.loads(response.read().decode("utf-8"))

message = body["choices"][0]["message"]["content"]
print("{_JSON_SENTINEL}" + json.dumps({{
    "status": "ok",
    "content": message,
    "id": body.get("id"),
}}))
"""
    return _run_remote_json(
        bifrost,
        script,
        working_dir=working_dir,
        python_bin=python_bin,
        timeout=120.0,
    )


def _probe_weight_update_schema(
    bifrost: Any,
    *,
    working_dir: str,
    base_url: str,
    python_bin: str | None = None,
) -> dict[str, Any]:
    script = f"""
import json
import urllib.request

with urllib.request.urlopen({f"{base_url}/weight_update_schema?limit=1"!r}, timeout=60) as response:
    body = json.loads(response.read().decode("utf-8"))

parameters = body.get("parameters", [])
print("{_JSON_SENTINEL}" + json.dumps({{
    "status": body.get("status"),
    "num_parameters": len(parameters),
    "first_parameter": parameters[0] if parameters else None,
}}))
"""
    return _run_remote_json(
        bifrost,
        script,
        working_dir=working_dir,
        python_bin=python_bin,
        timeout=120.0,
    )


def _build_launch_args(spec: RunPodSmokeSpec, model: str, port: int) -> tuple[str, ...]:
    args = (
        "-m",
        spec.launch_module,
        spec.model_flag,
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--trust-remote-code",
    )
    if spec.launch_with_uv:
        return ("run", "python", *args)
    return args


async def run_realization_smoke_on_runpod(
    spec: RunPodSmokeSpec,
    *,
    model: str | None = None,
    gpu_type: str | None = None,
    keep_alive: bool = False,
) -> dict[str, Any]:
    from bifrost import GPUQuery, ProcessSpec, acquire_node

    os.chdir(REPO_ROOT)

    chosen_model = model or spec.default_model
    chosen_gpu_type = gpu_type or spec.default_gpu_type
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"{spec.name}-{timestamp}"

    print(f"Provisioning {chosen_gpu_type} on RunPod for {spec.name}...")
    bifrost, instance = await acquire_node(
        provision=GPUQuery(
            type=chosen_gpu_type,
            count=1,
            min_cuda="12.0",
            name=f"rollouts/{run_name}",
            provider="runpod",
        )
    )
    node_id = f"{instance.provider}:{instance.id}"
    print(f"Provisioned {node_id}")
    job_tmux_session: str | None = None

    try:
        workspace = bifrost.push(
            "~/.bifrost/workspaces/rollouts-inference-smokes",
            allow_dirty=True,
        )
        print(f"Workspace pushed to {workspace}")

        for label, command in spec.bootstrap_steps:
            print(f"Bootstrapping: {label}")
            _run_remote_checked(bifrost, command, working_dir=workspace, timeout=1800.0)

        smoke_dir = f"{workspace}/.runpod-smokes"
        _run_remote_checked(
            bifrost,
            f"mkdir -p {shlex.quote(smoke_dir)}",
            working_dir=workspace,
            timeout=30.0,
        )
        log_path = f"{smoke_dir}/{spec.name}.log"

        print(f"Launching {spec.launch_module} with model {chosen_model}")
        job = bifrost.submit(
            ProcessSpec(
                command=spec.remote_python_bin or _REMOTE_UV_BIN,
                args=_build_launch_args(spec, chosen_model, spec.port),
                cwd=f"{workspace}/rollouts",
                env=spec.env,
                cuda_device_ids=(0,),
            ),
            name=run_name,
            log_file=log_path,
        )
        job_tmux_session = job.tmux_session
        print(f"Process started in tmux session {job.tmux_session}")

        base_url = f"http://localhost:{spec.port}"
        health_body = _wait_for_http_ready(
            bifrost,
            url=f"{base_url}/health",
            working_dir=workspace,
            timeout_seconds=600,
            log_path=log_path,
            tmux_session=job_tmux_session,
        )
        print(f"/health ok: {health_body.strip()}")

        schema_result: dict[str, Any] | None = None
        if spec.probe_weight_update_schema:
            schema_result = _probe_weight_update_schema(
                bifrost,
                working_dir=f"{workspace}/rollouts",
                base_url=base_url,
                python_bin=spec.remote_python_bin,
            )
            print(f"/weight_update_schema ok: {json.dumps(schema_result)}")

        generation_result: dict[str, Any] | None = None
        if spec.probe_generation:
            generation_result = _probe_generation(
                bifrost,
                working_dir=f"{workspace}/rollouts",
                base_url=base_url,
                model=chosen_model,
                python_bin=spec.remote_python_bin,
            )
            print(f"generation ok: {json.dumps(generation_result)}")

        return {
            "status": "ok",
            "node_id": node_id,
            "workspace": workspace,
            "tmux_session": job_tmux_session,
            "base_url": base_url,
            "model": chosen_model,
            "schema_result": schema_result,
            "generation_result": generation_result,
        }
    finally:
        if job_tmux_session is not None:
            bifrost.exec(
                f"tmux kill-session -t {shlex.quote(job_tmux_session)} 2>/dev/null || true",
                working_dir="~",
                timeout=10,
            )
        if not keep_alive:
            print(f"Terminating {node_id}")
            await instance.terminate()
        else:
            print(f"Keeping {node_id} alive")
