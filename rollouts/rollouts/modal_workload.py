"""Rollouts-owned Modal workload helpers.

These helpers are workload semantics, not execution substrate. They remain in
`rollouts` while `bifrost` owns sandbox/session/process lifecycle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import trio

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
MODAL_APP_NAME = "rollouts-training"
MODEL_CACHE_DICT_NAME = "rollouts-model-cache"
HF_CACHE_DIR = "/root/.cache/huggingface"
IMAGE_VENV_DIR = "/opt/venvs/rollouts"
IMAGE_VENV_PYTHON = f"{IMAGE_VENV_DIR}/bin/python"
WORKLOAD_ENTRYPOINT_SENTINEL = "__ARGUS_WORKLOAD_ENTRYPOINT_STARTED__"
# TODO(workload-staging): This sentinel currently marks the inner
# `argus.run --local` trampoline, not the real workload denotation. Replace it
# with a resolved workload launch spec and explicit child semantic events so the
# parent stream does not lose meaning at the control-plane handoff.
ARGUS_DIAG_EVENT_SENTINEL = "__ARGUS_DIAG__"
MODAL_FAILURE_DIAGNOSTICS_TIMEOUT_S = 30
MODAL_FAILURE_DIAGNOSTICS_OUTPUT_CHAR_LIMIT = 4000
MODAL_CLEANUP_SCOPES = ("app", "tag", "run", "none")


def sandbox_runtime_diag_python() -> str:
    """Return a small sibling-process monitor for hard-kill debugging."""
    return r"""
import json
import os
import pathlib
import subprocess
import sys
import time

SENTINEL = "__ARGUS_DIAG__"


def emit(event: str, **data: object) -> None:
    payload = {"event": event, **data}
    sys.stderr.write(f"{SENTINEL}{json.dumps(payload, sort_keys=True)}\n")
    sys.stderr.flush()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_text(path: str) -> str | None:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return None


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    text = read_text("/proc/meminfo")
    if not text:
        return values
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            try:
                values[key] = int(parts[0])
            except ValueError:
                pass
    return values


def read_proc_status(pid: int) -> dict[str, str]:
    wanted = {"VmRSS", "VmHWM", "VmSize", "Threads", "State"}
    text = read_text(f"/proc/{pid}/status")
    if not text:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key in wanted:
            values[key] = value.strip()
    return values


def parse_cgroup_events(text: str | None) -> dict[str, int] | None:
    if not text:
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, raw = line.partition(" ")
        try:
            values[key] = int(raw.strip())
        except ValueError:
            continue
    return values


def query_nvidia_smi() -> list[dict[str, object]]:
    try:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]

    rows: list[dict[str, object]] = []
    for raw_line in gpu_result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 5:
            continue
        index, name, total_mb, used_mb, free_mb = parts
        try:
            total = int(total_mb)
            used = int(used_mb)
            free = int(free_mb)
        except ValueError:
            continue
        rows.append(
            {
                "device": int(index),
                "name": name,
                "total_gb": round(total / 1024, 3),
                "used_gb": round(used / 1024, 3),
                "free_gb": round(free / 1024, 3),
                "used_frac": round(used / total, 4) if total else 0.0,
            }
        )
    return rows


def build_sample(main_pid: int, seq: int) -> dict[str, object]:
    meminfo = read_meminfo()
    mem_total_kb = meminfo.get("MemTotal", 0)
    mem_available_kb = meminfo.get("MemAvailable", 0)
    host_mem_used_frac = None
    if mem_total_kb and mem_available_kb:
        host_mem_used_frac = round(1.0 - (mem_available_kb / mem_total_kb), 4)

    return {
        "sample_seq": seq,
        "time_unix_s": round(time.time(), 3),
        "main_pid": main_pid,
        "main_alive": pid_alive(main_pid),
        "main_proc_status": read_proc_status(main_pid),
        "host_mem_total_gb": round(mem_total_kb / (1024**2), 3) if mem_total_kb else None,
        "host_mem_available_gb": round(mem_available_kb / (1024**2), 3) if mem_available_kb else None,
        "host_mem_used_frac": host_mem_used_frac,
        "cgroup_memory_current": read_text("/sys/fs/cgroup/memory.current"),
        "cgroup_memory_max": read_text("/sys/fs/cgroup/memory.max"),
        "cgroup_memory_events": parse_cgroup_events(read_text("/sys/fs/cgroup/memory.events")),
        "nvidia_smi": query_nvidia_smi(),
    }


def main() -> int:
    main_pid = int(sys.argv[1])
    interval_s = float(sys.argv[2])
    emit(
        "remote_runtime_diag_started",
        main_pid=main_pid,
        monitor_pid=os.getpid(),
        interval_s=interval_s,
    )
    seq = 0
    while True:
        emit("remote_runtime_diag_sample", **build_sample(main_pid, seq))
        if not pid_alive(main_pid):
            emit("remote_runtime_diag_target_gone", **build_sample(main_pid, seq + 1))
            return 0
        seq += 1
        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
"""


def sandbox_runtime_artifact_tail_python() -> str:
    """Return a small sibling-process monitor for sandbox-local JSONL artifacts.

    This is a control-plane bridge, not a new source of truth. The workload still
    owns its local `training.jsonl` / `metrics.jsonl`; this helper only projects
    selected structured progress back to the parent run journal while remote
    stdout/stderr remain sparse.
    """
    return r"""
import json
import os
import pathlib
import sys
import time

SENTINEL = "__ARGUS_DIAG__"
WATCH_FILES = ("training.jsonl", "metrics.jsonl")


def emit(event: str, **data: object) -> None:
    payload = {"event": event, **data}
    sys.stderr.write(f"{SENTINEL}{json.dumps(payload, sort_keys=True)}\n")
    sys.stderr.flush()


def _project_training_event(data: dict[str, object]) -> None:
    event_name = data.get("event")
    if event_name != "step_complete":
        return
    emit(
        "step_complete",
        projected_from_artifact=True,
        projection_source="training.jsonl",
        step=data.get("step"),
        mean_reward=data.get("mean_reward"),
        pg_loss=data.get("pg_loss"),
        entropy=data.get("entropy"),
        num_samples=data.get("num_samples"),
        num_groups=data.get("num_groups"),
        step_total_ms=data.get("step_total_ms"),
        rollout_step_count=data.get("rollout_step_count"),
        gpu_allocated_gb=data.get("gpu_allocated_gb"),
        gpu_reserved_gb=data.get("gpu_reserved_gb"),
        ram_gb=data.get("ram_gb"),
    )


def _project_metrics_event(data: dict[str, object]) -> None:
    emit(
        "metrics_update",
        projected_from_artifact=True,
        projection_source="metrics.jsonl",
        step=data.get("step"),
        mean_reward=data.get("mean_reward"),
        loss=data.get("loss"),
        grad_norm=data.get("grad_norm"),
        pg_loss=data.get("pg_loss"),
        entropy=data.get("entropy"),
        rollout_step_count=data.get("rollout_step_count"),
        rollout_samples_generated=data.get("rollout_samples_generated"),
        timestamp=data.get("timestamp"),
    )


def _consume_file(path: pathlib.Path, state: dict[str, object]) -> None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(int(state["offset"]))
            chunk = f.read()
            state["offset"] = f.tell()
    except FileNotFoundError:
        return

    if not chunk:
        return

    buffer = f"{state['buffer']}{chunk}"
    lines = buffer.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        state["buffer"] = lines.pop()
    else:
        state["buffer"] = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception as exc:
            emit(
                "remote_artifact_parse_failed",
                file=path.name,
                error=f"{type(exc).__name__}: {exc}",
                line_preview=line[:400],
            )
            continue
        if not isinstance(parsed, dict):
            continue
        if path.name == "training.jsonl":
            _project_training_event(parsed)
        elif path.name == "metrics.jsonl":
            _project_metrics_event(parsed)


def main() -> int:
    output_dir = pathlib.Path(sys.argv[1])
    interval_s = float(sys.argv[2])
    state = {
        name: {"offset": 0, "buffer": "", "ready_announced": False}
        for name in WATCH_FILES
    }
    emit(
        "remote_artifact_tail_started",
        output_dir=str(output_dir),
        files=list(WATCH_FILES),
        interval_s=interval_s,
    )
    while True:
        for name in WATCH_FILES:
            path = output_dir / name
            file_state = state[name]
            if path.exists() and not bool(file_state["ready_announced"]):
                file_state["ready_announced"] = True
                emit(
                    "remote_artifact_file_ready",
                    file=name,
                    output_dir=str(output_dir),
                )
            _consume_file(path, file_state)
        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
"""


def sandbox_runtime_supervisor_python() -> str:
    """Return the supervisor for the remote workload process tree."""
    return r"""
import json
import os
import signal
import subprocess
import sys
import time

ARGUS_DIAG_EVENT_SENTINEL = "__ARGUS_DIAG__"


def emit(event: str, **data: object) -> None:
    payload = {"event": event, **data}
    sys.stderr.write(f"{ARGUS_DIAG_EVENT_SENTINEL}{json.dumps(payload, sort_keys=True)}\n")
    sys.stderr.flush()


def write_status(status_path: str | None, payload: dict[str, object]) -> None:
    if not status_path:
        return
    parent = os.path.dirname(status_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{status_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, status_path)


def read_proc_status(pid: int) -> dict[str, str]:
    wanted = {"Name", "State", "VmRSS", "VmHWM", "VmSize", "Threads"}
    path = f"/proc/{pid}/status"
    try:
        text = open(path, encoding="utf-8").read()
    except Exception:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key in wanted:
            values[key] = value.strip()
    return values


def read_children(pid: int) -> list[int]:
    path = f"/proc/{pid}/task/{pid}/children"
    try:
        raw = open(path, encoding="utf-8").read().strip()
    except Exception:
        return []
    if not raw:
        return []
    out: list[int] = []
    for item in raw.split():
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def snapshot_tree(root_pid: int) -> list[dict[str, object]]:
    seen: set[int] = set()
    queue = [root_pid]
    rows: list[dict[str, object]] = []
    while queue:
        pid = queue.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        status = read_proc_status(pid)
        if not status:
            continue
        children = read_children(pid)
        rows.append({"pid": pid, "status": status, "children": children})
        queue.extend(children)
    return rows


def terminate_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass


def main() -> int:
    workspace = sys.argv[1]
    image_python = sys.argv[2]
    diag_python = sys.argv[3]
    artifact_tail_python = sys.argv[4]
    config_rel = sys.argv[5]
    child = None
    diag = None
    artifact_tail = None
    started_at = time.monotonic()
    env = os.environ.copy()
    status_path = env.get("ARGUS_SUPERVISOR_STATUS_FILE")
    output_dir = env.get("ROLLOUTS_OUTPUT_DIR")

    def _handle_signal(signum, _frame):
        emit("remote_supervisor_signal", signum=signum)
        write_status(status_path, {"event": "remote_supervisor_signal", "signum": signum})
        terminate_process(child)
        terminate_process(diag)
        terminate_process(artifact_tail)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        child = subprocess.Popen(
            [image_python, "-m", "argus.run", "--local", "--config", config_rel],
            cwd=workspace,
            env=env,
        )
        emit(
            "remote_supervisor_child_started",
            child_pid=child.pid,
            process_tree=snapshot_tree(child.pid),
        )
        write_status(
            status_path,
            {"event": "remote_supervisor_child_started", "child_pid": child.pid},
        )
        diag = subprocess.Popen(
            [image_python, "-u", "-c", diag_python, str(child.pid), "1.0"],
            cwd=workspace,
            env=env,
        )
        emit(
            "remote_supervisor_diag_started",
            child_pid=child.pid,
            diag_pid=diag.pid,
        )
        if output_dir:
            output_path = output_dir
            if not os.path.isabs(output_path):
                output_path = os.path.join(workspace, output_path)
            artifact_tail = subprocess.Popen(
                [image_python, "-u", "-c", artifact_tail_python, output_path, "0.5"],
                cwd=workspace,
                env=env,
            )
            emit(
                "remote_supervisor_artifact_tail_started",
                child_pid=child.pid,
                artifact_pid=artifact_tail.pid,
                output_dir=output_path,
            )
        rc = child.wait()
        emit(
            "remote_supervisor_child_exit",
            child_pid=child.pid,
            child_returncode=rc,
            child_was_signaled=(rc < 0),
            child_signal=(-rc if rc < 0 else None),
            elapsed_sec=round(time.monotonic() - started_at, 3),
            process_tree=snapshot_tree(child.pid),
        )
        write_status(
            status_path,
            {
                "event": "remote_supervisor_child_exit",
                "child_pid": child.pid,
                "child_returncode": rc,
                "child_was_signaled": (rc < 0),
                "child_signal": (-rc if rc < 0 else None),
                "elapsed_sec": round(time.monotonic() - started_at, 3),
            },
        )
        if diag.poll() is None:
            try:
                diag.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                terminate_process(diag)
        if artifact_tail is not None and artifact_tail.poll() is None:
            try:
                artifact_tail.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                terminate_process(artifact_tail)
        if rc < 0:
            return 128 + (-rc)
        return rc
    finally:
        terminate_process(diag)
        terminate_process(artifact_tail)


if __name__ == "__main__":
    raise SystemExit(main())
"""


def trim_failure_diagnostics_output(text: str) -> tuple[str, bool]:
    trimmed = text.rstrip()
    if len(trimmed) <= MODAL_FAILURE_DIAGNOSTICS_OUTPUT_CHAR_LIMIT:
        return trimmed, False
    return trimmed[: MODAL_FAILURE_DIAGNOSTICS_OUTPUT_CHAR_LIMIT - 3] + "...", True


def modal_failure_diagnostic_probes() -> tuple[tuple[str, str], ...]:
    return (
        (
            "cgroup_memory_events",
            "if [ -f /sys/fs/cgroup/memory.events ]; then cat /sys/fs/cgroup/memory.events; "
            "elif [ -f /sys/fs/cgroup/memory/memory.oom_control ]; then cat /sys/fs/cgroup/memory/memory.oom_control; "
            "else echo unavailable; fi",
        ),
        (
            "cgroup_memory_state",
            "for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.max "
            "/sys/fs/cgroup/memory.swap.current /sys/fs/cgroup/memory.swap.max; do "
            'if [ -f "$f" ]; then printf \'%s=\' "$f"; cat "$f"; fi; done',
        ),
        (
            "memory_pressure",
            "if [ -f /proc/pressure/memory ]; then cat /proc/pressure/memory; else echo unavailable; fi",
        ),
        ("proc_meminfo", "cat /proc/meminfo"),
        ("nvidia_smi", "nvidia-smi"),
        (
            "nvidia_smi_compute_apps",
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits",
        ),
        (
            "process_table",
            "ps -eo pid,ppid,pgid,stat,%mem,%cpu,rss,vsz,etimes,cmd --sort=-rss | head -n 40",
        ),
        ("kernel_messages", "dmesg | tail -n 200"),
    )


def parse_counter_map(raw: str) -> dict[str, int]:
    counters: dict[str, int] = {}
    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        name, value = parts
        try:
            counters[name] = int(value)
        except ValueError:
            continue
    return counters


def summarize_failure_diagnostics(
    probes: dict[str, dict[str, Any]],
    *,
    exit_code: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"exit_code": exit_code}

    memory_events = probes.get("cgroup_memory_events")
    if memory_events is not None:
        counters = parse_counter_map(str(memory_events.get("stdout", "")))
        if counters:
            summary["cgroup_memory_events"] = counters
            if counters.get("oom_kill", 0) > 0 or counters.get("oom_group_kill", 0) > 0:
                summary["suspected_cause"] = "cgroup_oom_kill"
            elif counters.get("oom", 0) > 0:
                summary["suspected_cause"] = "cgroup_oom"

    kernel_messages = probes.get("kernel_messages")
    if kernel_messages is not None:
        stderr = str(kernel_messages.get("stderr", ""))
        stdout = str(kernel_messages.get("stdout", ""))
        if "Operation not permitted" in stderr or "permission denied" in stderr.lower():
            summary["kernel_messages_access"] = "denied"
        elif stdout:
            summary["kernel_messages_access"] = "available"

    return summary


async def collect_modal_failure_diagnostics(
    sandbox: Any,
    *,
    exit_code: int,
    emit: Callable[..., None],
) -> dict[str, Any]:
    """Snapshot remote runtime state after a hard workload failure."""
    from bifrost.modal_backend import exec_modal_command_sync

    emit("modal_failure_diagnostics_start", exit_code=exit_code)
    probes: dict[str, dict[str, Any]] = {}

    for probe_name, command in modal_failure_diagnostic_probes():
        emit("modal_failure_diagnostics_probe_start", probe=probe_name)

        def _run_probe() -> tuple[str, str, int]:
            return exec_modal_command_sync(
                sandbox,
                command,
                timeout=MODAL_FAILURE_DIAGNOSTICS_TIMEOUT_S,
                stream_output=False,
            )

        try:
            stdout, stderr, probe_exit_code = await trio.to_thread.run_sync(_run_probe)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            probes[probe_name] = {"error": error}
            emit("modal_failure_diagnostics_probe_failed", probe=probe_name, error=error)
            continue

        stdout_excerpt, stdout_truncated = trim_failure_diagnostics_output(stdout)
        stderr_excerpt, stderr_truncated = trim_failure_diagnostics_output(stderr)
        probe_result = {
            "exit_code": probe_exit_code,
            "stdout": stdout_excerpt,
            "stderr": stderr_excerpt,
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
        probes[probe_name] = probe_result
        emit("modal_failure_diagnostics_probe_finished", probe=probe_name, **probe_result)

    summary = summarize_failure_diagnostics(probes, exit_code=exit_code)
    emit("modal_failure_diagnostics_finished", summary=summary)
    return {"summary": summary, "probes": probes}


def get_model_cache_key(model_name: str, pruning_recipe: str | None = None) -> str:
    key = model_name.replace("/", "--").replace(":", "-")
    if pruning_recipe:
        import hashlib

        recipe_hash = hashlib.md5(pruning_recipe.encode()).hexdigest()[:8]
        key = f"{key}--pruned-{recipe_hash}"
    return key


async def get_cached_snapshot(model_name: str, pruning_recipe: str | None = None) -> Any | None:
    import modal

    cache_key = get_model_cache_key(model_name, pruning_recipe)

    try:

        def _lookup() -> str | None:
            cache_dict = modal.Dict.from_name(MODEL_CACHE_DICT_NAME, create_if_missing=True)
            return cache_dict.get(cache_key)

        snapshot_id = await trio.to_thread.run_sync(_lookup)
        if snapshot_id:
            logger.info("Found cached weights snapshot for %s: %s", model_name, snapshot_id)
            return modal.Image.from_id(snapshot_id)
    except Exception as exc:
        logger.warning("Failed to look up model cache: %s", exc)

    return None


async def save_snapshot_to_cache(
    model_name: str,
    snapshot: Any,
    pruning_recipe: str | None = None,
) -> None:
    import modal

    cache_key = get_model_cache_key(model_name, pruning_recipe)
    snapshot_id = snapshot.object_id

    try:

        def _save() -> None:
            cache_dict = modal.Dict.from_name(MODEL_CACHE_DICT_NAME, create_if_missing=True)
            cache_dict[cache_key] = snapshot_id

        await trio.to_thread.run_sync(_save)
        logger.info("Cached weights snapshot for %s: %s", model_name, snapshot_id)
    except Exception as exc:
        logger.warning("Failed to save model cache: %s", exc)


async def download_and_snapshot_model(sandbox: Any, model_name: str) -> Any | None:
    import threading

    import trio_asyncio

    logger.info("Downloading model weights for %s...", model_name)
    download_script = f"""
import sys
from huggingface_hub import snapshot_download
print(f"Starting download of {model_name!r}...", flush=True)
path = snapshot_download("{model_name}")
print(f"Downloaded to: {{path}}", flush=True)
"""
    proc = await trio_asyncio.aio_as_trio(
        sandbox.exec.aio("python", "-c", download_script.strip(), timeout=1800)
    )

    def _read_stdout() -> None:
        for line in proc.stdout:
            logger.info("[download] %s", line.rstrip())

    def _read_stderr() -> None:
        for line in proc.stderr:
            logger.info("[download] %s", line.rstrip())

    def _stream_both() -> None:
        t1 = threading.Thread(target=_read_stdout)
        t2 = threading.Thread(target=_read_stderr)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    await trio.to_thread.run_sync(_stream_both)

    exit_code = await trio_asyncio.aio_as_trio(proc.wait.aio())
    if exit_code != 0:
        logger.error("Model download failed with exit code %s", exit_code)
        return None

    logger.info("Model downloaded, creating directory snapshot...")

    try:
        snapshot = await trio_asyncio.aio_as_trio(
            sandbox._experimental_snapshot_directory.aio(HF_CACHE_DIR)
        )
        logger.info("Created snapshot: %s", snapshot.object_id)
        return snapshot
    except Exception:
        logger.exception("Failed to create snapshot")
        return None


async def download_prune_and_snapshot_model(
    sandbox: Any,
    model_name: str,
    pruning_recipe: str,
) -> Any | None:
    import threading

    import trio_asyncio

    logger.info("Downloading and pruning model: %s", model_name)
    prune_script = f'''
import json
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "{model_name}"
recipe_json = {pruning_recipe!r}

recipe = json.loads(recipe_json)
experts_to_keep = {{int(k): v for k, v in recipe["experts_to_keep"].items()}}

logger.info(f"Downloading {{model_name}}...")
cache_path = snapshot_download(model_name)
logger.info(f"Downloaded to: {{cache_path}}")

logger.info("Loading model for pruning...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

config = model.config
original_num = getattr(config, "n_routed_experts", None) or getattr(config, "num_experts", 0)
logger.info(f"Original experts: {{original_num}}")

for layer_idx, keep_indices in experts_to_keep.items():
    layer = model.model.layers[layer_idx]
    moe_block = getattr(layer, "mlp", None)
    if moe_block is None or not hasattr(moe_block, "experts"):
        continue

    experts = moe_block.experts
    num_experts = len(experts)
    moe_block.experts = nn.ModuleList([experts[i] for i in keep_indices])

    router = getattr(moe_block, "gate", None)
    if router is not None and hasattr(router, "weight"):
        weight = router.weight.data
        if weight.shape[0] == num_experts:
            router.weight = nn.Parameter(weight[keep_indices])
        elif weight.shape[1] == num_experts:
            router.weight = nn.Parameter(weight[:, keep_indices])
        if hasattr(router, "bias") and router.bias is not None:
            if router.bias.shape[0] == num_experts:
                router.bias = nn.Parameter(router.bias.data[keep_indices])

    logger.info(f"Layer {{layer_idx}}: {{num_experts}} -> {{len(keep_indices)}} experts")

new_num = len(next(iter(experts_to_keep.values())))
if hasattr(config, "n_routed_experts"):
    config.n_routed_experts = new_num
if hasattr(config, "num_experts"):
    config.num_experts = new_num

logger.info(f"Pruning complete: {{original_num}} -> {{new_num}} experts")
logger.info(f"Saving pruned model to {{cache_path}}...")
model.save_pretrained(cache_path, safe_serialization=True)
tokenizer.save_pretrained(cache_path)

with open(f"{{cache_path}}/pruning_recipe.json", "w") as f:
    json.dump(recipe, f, indent=2)

logger.info("Pruned model saved!")
'''

    proc = await trio_asyncio.aio_as_trio(
        sandbox.exec.aio("python", "-c", prune_script, timeout=3600)
    )

    def _read_stdout() -> None:
        for line in proc.stdout:
            logger.info("[prune] %s", line.rstrip())

    def _read_stderr() -> None:
        for line in proc.stderr:
            logger.info("[prune] %s", line.rstrip())

    def _stream_both() -> None:
        t1 = threading.Thread(target=_read_stdout)
        t2 = threading.Thread(target=_read_stderr)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    await trio.to_thread.run_sync(_stream_both)

    exit_code = await trio_asyncio.aio_as_trio(proc.wait.aio())
    if exit_code != 0:
        logger.error("Pruning failed with exit code %s", exit_code)
        return None

    logger.info("Pruning complete, creating directory snapshot...")

    try:
        snapshot = await trio_asyncio.aio_as_trio(
            sandbox._experimental_snapshot_directory.aio(HF_CACHE_DIR)
        )
        logger.info("Created pruned model snapshot: %s", snapshot.object_id)
        return snapshot
    except Exception:
        logger.exception("Failed to create snapshot")
        return None


async def prune_mounted_model_and_snapshot(
    sandbox: Any,
    model_name: str,
    pruning_recipe: str,
) -> Any | None:
    import threading

    import trio_asyncio

    logger.info("Pruning mounted model: %s", model_name)
    prune_script = f'''
import json
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "{model_name}"
recipe_json = {pruning_recipe!r}

recipe = json.loads(recipe_json)
experts_to_keep = {{int(k): v for k, v in recipe["experts_to_keep"].items()}}

cache_path = snapshot_download(model_name, local_files_only=True)
logger.info(f"Using cached model at: {{cache_path}}")

logger.info("Loading model for pruning...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

config = model.config
original_num = getattr(config, "n_routed_experts", None) or getattr(config, "num_experts", 0)
logger.info(f"Original experts: {{original_num}}")

for layer_idx, keep_indices in experts_to_keep.items():
    layer = model.model.layers[layer_idx]
    moe_block = getattr(layer, "mlp", None)
    if moe_block is None or not hasattr(moe_block, "experts"):
        continue

    experts = moe_block.experts
    num_experts = len(experts)
    moe_block.experts = nn.ModuleList([experts[i] for i in keep_indices])

    router = getattr(moe_block, "gate", None)
    if router is not None and hasattr(router, "weight"):
        weight = router.weight.data
        if weight.shape[0] == num_experts:
            router.weight = nn.Parameter(weight[keep_indices])
        elif weight.shape[1] == num_experts:
            router.weight = nn.Parameter(weight[:, keep_indices])
        if hasattr(router, "bias") and router.bias is not None:
            if router.bias.shape[0] == num_experts:
                router.bias = nn.Parameter(router.bias.data[keep_indices])

    logger.info(f"Layer {{layer_idx}}: {{num_experts}} -> {{len(keep_indices)}} experts")

new_num = len(next(iter(experts_to_keep.values())))
if hasattr(config, "n_routed_experts"):
    config.n_routed_experts = new_num
if hasattr(config, "num_experts"):
    config.num_experts = new_num

logger.info(f"Pruning complete: {{original_num}} -> {{new_num}} experts")
logger.info(f"Saving pruned model to {{cache_path}}...")
model.save_pretrained(cache_path, safe_serialization=True)
tokenizer.save_pretrained(cache_path)

with open(f"{{cache_path}}/pruning_recipe.json", "w") as f:
    json.dump(recipe, f, indent=2)

logger.info("Pruned model saved!")
'''

    proc = await trio_asyncio.aio_as_trio(
        sandbox.exec.aio("python", "-c", prune_script, timeout=1800)
    )

    def _read_stdout() -> None:
        for line in proc.stdout:
            logger.info("[prune] %s", line.rstrip())

    def _read_stderr() -> None:
        for line in proc.stderr:
            logger.info("[prune] %s", line.rstrip())

    def _stream_both() -> None:
        t1 = threading.Thread(target=_read_stdout)
        t2 = threading.Thread(target=_read_stderr)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    await trio.to_thread.run_sync(_stream_both)

    exit_code = await trio_asyncio.aio_as_trio(proc.wait.aio())
    if exit_code != 0:
        logger.error("Pruning failed with exit code %s", exit_code)
        return None

    logger.info("Pruning complete, creating directory snapshot...")

    try:
        snapshot = await trio_asyncio.aio_as_trio(
            sandbox._experimental_snapshot_directory.aio(HF_CACHE_DIR)
        )
        logger.info("Created pruned model snapshot: %s", snapshot.object_id)
        return snapshot
    except Exception:
        logger.exception("Failed to create snapshot")
        return None


async def mount_cached_weights(sandbox: Any, snapshot: Any) -> None:
    import trio_asyncio

    logger.info("Mounting cached weights from snapshot %s...", snapshot.object_id)
    await trio_asyncio.aio_as_trio(sandbox._experimental_mount_image.aio(HF_CACHE_DIR, snapshot))
    logger.info("Cached weights mounted")
