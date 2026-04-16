from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.event_log import RunEventSinks, emit_run_event


@dataclass(frozen=True)
class ResourceWatchdogConfig:
    enabled: bool = False
    sample_interval_s: float = 1.0
    heartbeat_interval_s: float = 10.0
    warn_gpu_reserved_frac: float = 0.9
    warn_host_mem_used_frac: float = 0.9


def _read_proc_status_value_kb(name: str) -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(f"{name}:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except Exception:
        return None
    return None


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, rest = line.split(":", 1)
            parts = rest.split()
            if parts:
                values[key] = int(parts[0])
    except Exception:
        return {}
    return values


def _sample_nvidia_smi() -> list[dict[str, Any]]:
    """Return system-level GPU/process usage from nvidia-smi.

    Torch allocator stats only describe the current Python process. For our
    training runs that is often the wrong process entirely: SGLang runs in tmux
    and Megatron workers are separate children. Use nvidia-smi as the source of
    truth for cross-process GPU pressure.
    """
    try:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as e:
        return [{"error": f"nvidia_smi_gpu_query_failed:{type(e).__name__}:{e}"}]

    gpus: dict[str, dict[str, Any]] = {}
    for raw_line in gpu_result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 6:
            continue
        index, uuid, name, total_mb, used_mb, free_mb = parts
        try:
            total = int(total_mb)
            used = int(used_mb)
            free = int(free_mb)
        except ValueError:
            continue
        gpus[uuid] = {
            "device": int(index),
            "uuid": uuid,
            "name": name,
            "total_gb": round(total / 1024, 3),
            "used_gb": round(used / 1024, 3),
            "free_gb": round(free / 1024, 3),
            "used_frac": round(used / total, 4) if total else 0.0,
            "processes": [],
        }

    try:
        proc_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        proc_result = None

    if proc_result is not None:
        for raw_line in proc_result.stdout.splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) != 4:
                continue
            gpu_uuid, pid, process_name, used_mb = parts
            gpu = gpus.get(gpu_uuid)
            if gpu is None:
                continue
            try:
                used = int(used_mb)
            except ValueError:
                continue
            gpu["processes"].append({
                "pid": int(pid),
                "name": process_name,
                "used_gb": round(used / 1024, 3),
            })

    return [gpus[key] for key in sorted(gpus, key=lambda uuid: gpus[uuid]["device"])]


class ResourceWatchdog:
    def __init__(
        self,
        *,
        config: ResourceWatchdogConfig,
        run_logger: RunEventSinks,
        run_context: dict[str, Any],
    ) -> None:
        self.config = config
        self.run_logger = run_logger
        self.run_context = dict(run_context)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._phase = "startup"
        self._phase_data: dict[str, Any] = {}
        self._last_heartbeat = 0.0
        self._warned_gpu: set[int] = set()
        self._warned_host = False

    def start(self) -> None:
        if not self.config.enabled:
            return
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="resource-watchdog", daemon=True)
        self._thread.start()
        emit_run_event(
            self.run_logger,
            "resource_watchdog_started",
            **self.run_context,
            sample_interval_s=self.config.sample_interval_s,
            heartbeat_interval_s=self.config.heartbeat_interval_s,
            warn_gpu_reserved_frac=self.config.warn_gpu_reserved_frac,
            warn_host_mem_used_frac=self.config.warn_host_mem_used_frac,
        )

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        emit_run_event(
            self.run_logger,
            "resource_watchdog_stopped",
            **self.run_context,
            phase=self.phase,
        )

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def set_phase(self, phase: str, **phase_data: Any) -> None:
        if not self.config.enabled:
            return
        with self._lock:
            self._phase = phase
            self._phase_data = dict(phase_data)
        # TODO(watchdog-noise): The run journal currently gets flooded with
        # repeated resource_watchdog_phase events during long startup/warmup
        # regimes. Compress unchanged phases or rate-limit phase emission so
        # the journal stays queryable and the important control-flow edges
        # remain visually dominant.
        emit_run_event(
            self.run_logger,
            "resource_watchdog_phase",
            **self.run_context,
            phase=phase,
            phase_data=phase_data,
        )

    def _sample_gpu(self) -> list[dict[str, Any]]:
        torch = sys.modules.get("torch")
        if torch is None or not hasattr(torch, "cuda"):
            return []
        try:
            if not torch.cuda.is_available():
                return []
            device_count = torch.cuda.device_count()
            samples: list[dict[str, Any]] = []
            for idx in range(device_count):
                props = torch.cuda.get_device_properties(idx)
                total = int(props.total_memory)
                allocated = int(torch.cuda.memory_allocated(idx))
                reserved = int(torch.cuda.memory_reserved(idx))
                max_allocated = int(torch.cuda.max_memory_allocated(idx))
                samples.append({
                    "device": idx,
                    "name": props.name,
                    "total_gb": round(total / (1024**3), 3),
                    "allocated_gb": round(allocated / (1024**3), 3),
                    "reserved_gb": round(reserved / (1024**3), 3),
                    "max_allocated_gb": round(max_allocated / (1024**3), 3),
                    "reserved_frac": round(reserved / total, 4) if total else 0.0,
                })
            return samples
        except Exception as e:
            return [{"error": f"{type(e).__name__}: {e}"}]

    def _sample(self) -> dict[str, Any]:
        meminfo = _read_meminfo()
        mem_total_kb = meminfo.get("MemTotal", 0)
        mem_available_kb = meminfo.get("MemAvailable", 0)
        mem_used_frac = (
            1.0 - (mem_available_kb / mem_total_kb) if mem_total_kb and mem_available_kb else None
        )
        vmrss_kb = _read_proc_status_value_kb("VmRSS")
        with self._lock:
            phase = self._phase
            phase_data = dict(self._phase_data)
        return {
            "phase": phase,
            "phase_data": phase_data,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "process_rss_gb": round(vmrss_kb / (1024**2), 3) if vmrss_kb is not None else None,
            "host_mem_total_gb": round(mem_total_kb / (1024**2), 3) if mem_total_kb else None,
            "host_mem_available_gb": (
                round(mem_available_kb / (1024**2), 3) if mem_available_kb else None
            ),
            "host_mem_used_frac": round(mem_used_frac, 4) if mem_used_frac is not None else None,
            "gpus": self._sample_gpu(),
            "system_gpus": _sample_nvidia_smi(),
        }

    def _emit_sample(self, event: str, sample: dict[str, Any]) -> None:
        emit_run_event(self.run_logger, event, **self.run_context, **sample)

    def _emit_thresholds(self, sample: dict[str, Any]) -> None:
        host_used_frac = sample.get("host_mem_used_frac")
        if host_used_frac is not None:
            above = host_used_frac >= self.config.warn_host_mem_used_frac
            if above and not self._warned_host:
                self._warned_host = True
                self._emit_sample("resource_watchdog_warning", sample)
            elif not above:
                self._warned_host = False

        current_gpu_warns: set[int] = set()
        gpu_samples = sample.get("system_gpus") or sample.get("gpus", [])
        for gpu in gpu_samples:
            frac = gpu.get("used_frac", gpu.get("reserved_frac"))
            if "device" not in gpu or frac is None:
                continue
            if frac >= self.config.warn_gpu_reserved_frac:
                current_gpu_warns.add(int(gpu["device"]))
                if int(gpu["device"]) not in self._warned_gpu:
                    self._emit_sample("resource_watchdog_warning", sample)
        self._warned_gpu = current_gpu_warns

    def _run(self) -> None:
        self._last_heartbeat = time.monotonic()
        while not self._stop.wait(self.config.sample_interval_s):
            sample = self._sample()
            self._emit_thresholds(sample)
            now = time.monotonic()
            if now - self._last_heartbeat >= self.config.heartbeat_interval_s:
                self._last_heartbeat = now
                self._emit_sample("resource_watchdog_heartbeat", sample)
