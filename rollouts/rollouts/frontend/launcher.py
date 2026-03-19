from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .live_runs import (
    acquire_run_slot,
    next_run_id,
    register_run,
    release_run_slot,
)

logger = logging.getLogger(__name__)


def launch_config_run(project_root: Path, config_name: str) -> dict[str, Any]:
    if not config_name:
        raise ValueError("Missing configName")

    config_path = project_root / "configs" / f"{config_name}.py"
    if not config_path.exists():
        raise FileNotFoundError(config_name)

    cuda_device_ids = _extract_cuda_device_ids(config_path.read_text())
    gpu_busy_error = _gpu_preflight_error(cuda_device_ids)
    if gpu_busy_error is not None:
        return {
            "success": False,
            "error": gpu_busy_error,
            "preflight_failed": True,
        }

    if not acquire_run_slot():
        from .live_runs import max_concurrent_runs

        return {
            "success": False,
            "error": (
                f"Maximum concurrent runs ({max_concurrent_runs()}) reached. "
                "Please wait for a run to complete."
            ),
            "queue_full": True,
        }

    run_id = next_run_id()
    command = ["python", "entrypoint.py", str(config_path)]

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,
            start_new_session=True,
            env=env,
        )
    except Exception as e:
        logger.exception("Failed to start process: %s", e)
        release_run_slot()
        return {
            "success": False,
            "error": f"Failed to start process: {str(e)}",
        }

    register_run(
        run_id,
        {
            "process": process,
            "config_name": config_name,
            "start_time": time.time(),
            "status": "running",
            "output_lines": [],
            "exit_code": None,
            "cuda_device_ids": cuda_device_ids,
        },
    )

    return {
        "success": True,
        "run_id": run_id,
        "command": " ".join(command),
        "config_name": config_name,
    }


def _extract_cuda_device_ids(config_source: str) -> list[int]:
    gpu_list_match = re.search(r'["\']cuda_device_ids["\']\s*:\s*\[([^\]]+)\]', config_source)
    if not gpu_list_match:
        gpu_list_match = re.search(r"cuda_device_ids\s*[:=]\s*\[([^\]]+)\]", config_source)

    if gpu_list_match:
        return [int(x.strip()) for x in gpu_list_match.group(1).split(",") if x.strip().isdigit()]

    gpu_match = re.search(r'["\']gpu_id["\']\s*:\s*(\d+)', config_source)
    if not gpu_match:
        gpu_match = re.search(r"gpu_id\s*[:=]\s*(\d+)", config_source)
    if gpu_match:
        return [int(gpu_match.group(1))]
    return []


def _gpu_preflight_error(cuda_device_ids: list[int]) -> str | None:
    if not cuda_device_ids:
        return None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        return None

    gpu_stats: dict[int, dict[str, int]] = {}
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpu_id = int(parts[0])
            gpu_stats[gpu_id] = {
                "memory_mb": int(parts[1]),
                "util_pct": int(parts[2]),
            }
        except ValueError:
            continue

    for gpu_id in cuda_device_ids:
        if gpu_id not in gpu_stats:
            continue
        stats = gpu_stats[gpu_id]
        if stats["memory_mb"] > 1000 or stats["util_pct"] > 5:
            return (
                f"GPU {gpu_id} is busy "
                f"({stats['memory_mb']}MB used, {stats['util_pct']}% util)"
            )
    return None
