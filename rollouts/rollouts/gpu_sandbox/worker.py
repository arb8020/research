"""SandboxWorker: interface for kernel scoring workers.

Workers can be:
- LocalSandboxWorker: runs scoring in local subprocess
- RemoteSandboxWorker: connects to remote WorkerServer via miniray

All workers implement the same scoring protocol:
    request: {"kernel_code": str, "ref_code": str}
    response: {"compiled": float, "correct": float, "speedup": float, ...}
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trio

logger = logging.getLogger(__name__)


class SandboxWorker(ABC):
    """Abstract base for sandbox workers."""

    @abstractmethod
    async def score(
        self,
        kernel_code: str,
        ref_code: str,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Score a kernel.

        Args:
            kernel_code: Generated kernel code containing ModelNew class
            ref_code: Reference code with Model, get_inputs, get_init_inputs
            timeout: Scoring timeout in seconds

        Returns:
            Dict with: compiled, correct, speedup, reward, pass_rate, error
        """

    @abstractmethod
    async def close(self) -> None:
        """Cleanup worker resources."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if worker is healthy."""

    @abstractmethod
    async def describe_runtime(self) -> dict[str, Any]:
        """Describe the runtime environment used for scoring."""


@dataclass
class LocalSandboxWorker(SandboxWorker):
    """Runs kernel scoring in local subprocess.

    Uses the same scoring logic as scoring.py but in an isolated subprocess
    to avoid polluting the main process with compiled kernels.

    Uses trio.to_thread.run_sync() to run blocking subprocess in a thread,
    keeping the trio event loop responsive.
    """

    async def score(
        self,
        kernel_code: str,
        ref_code: str,
        timeout: float = 120.0,
        *,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        num_correct_seeds: int = 5,
        warmup_runs: int = 5,
        timed_runs: int = 30,
        check_determinism: bool = True,
        adaptive_baseline: bool = True,
        excessive_speedup_threshold: float = 10.0,
        forbidden_ops: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Score kernel in subprocess using trio threading."""

        def _run_subprocess() -> tuple[str, str, int]:
            """Run scoring in subprocess (blocking, runs in thread)."""
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as kf:
                kf.write(kernel_code)
                kernel_path = kf.name

            script = _build_scoring_script(
                kernel_code,
                ref_code,
                kernel_path,
                atol=atol,
                rtol=rtol,
                num_correct_seeds=num_correct_seeds,
                warmup_runs=warmup_runs,
                timed_runs=timed_runs,
                check_determinism=check_determinism,
                adaptive_baseline=adaptive_baseline,
                excessive_speedup_threshold=excessive_speedup_threshold,
                forbidden_ops=forbidden_ops,
            )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script)
                script_path = f.name

            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return result.stdout, result.stderr, result.returncode
            except subprocess.TimeoutExpired:
                return "", "", -1  # Timeout sentinel
            finally:
                Path(script_path).unlink(missing_ok=True)
                Path(kernel_path).unlink(missing_ok=True)

        # Run blocking subprocess in thread
        stdout, stderr, returncode = await trio.to_thread.run_sync(_run_subprocess)

        logger.info(f"[Scoring] returncode={returncode}")
        logger.info(f"[Scoring] stdout={stdout[:500] if stdout else 'empty'}...")
        logger.info(f"[Scoring] stderr={stderr[:500] if stderr else 'empty'}...")

        if returncode == -1:
            return {
                "compiled": 0.0,
                "correct": 0.0,
                "speedup": 0.0,
                "reward": 0.0,
                "pass_rate": 0.0,
                "error": "Scoring timeout",
            }

        return _parse_scoring_output(stdout, stderr, returncode)

    async def close(self) -> None:
        """No cleanup needed for local worker."""
        pass

    async def health_check(self) -> bool:
        """Local worker is always healthy."""
        return True

    async def describe_runtime(self) -> dict[str, Any]:
        """Describe the local subprocess runtime used for scoring."""

        def _run_probe() -> tuple[str, str, int]:
            result = subprocess.run(
                [sys.executable, "-c", _build_runtime_probe_script()],
                capture_output=True,
                text=True,
            )
            return result.stdout, result.stderr, result.returncode

        stdout, stderr, returncode = await trio.to_thread.run_sync(_run_probe)
        return _parse_runtime_probe(stdout, stderr, returncode, worker_kind="local")


@dataclass
class BrokerSandboxWorker(SandboxWorker):
    """Runs kernel scoring on a broker-provisioned GPU instance.

    Uses broker's GPUInstance.aexec() for remote command execution.
    Scoring runs as a subprocess on the remote instance (same pattern
    as LocalSandboxWorker but over SSH).
    """

    instance: Any  # broker.types.GPUInstance
    keep_alive: bool = False
    ssh_key_path: str | None = None

    async def score(
        self,
        kernel_code: str,
        ref_code: str,
        timeout: float = 120.0,
        *,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        num_correct_seeds: int = 5,
        warmup_runs: int = 5,
        timed_runs: int = 30,
        check_determinism: bool = True,
        adaptive_baseline: bool = True,
        excessive_speedup_threshold: float = 10.0,
        forbidden_ops: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Score kernel on remote instance via SSH."""
        import base64

        kernel_b64 = base64.b64encode(kernel_code.encode()).decode()
        script = _build_scoring_script_for_remote(
            ref_code,
            kernel_b64,
            atol=atol,
            rtol=rtol,
            num_correct_seeds=num_correct_seeds,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            check_determinism=check_determinism,
            adaptive_baseline=adaptive_baseline,
            excessive_speedup_threshold=excessive_speedup_threshold,
            forbidden_ops=forbidden_ops,
        )

        # Escape for bash heredoc
        escaped_script = script.replace("'", "'\"'\"'")

        # Write script and execute on remote
        command = f"""
python3 << 'SCORING_SCRIPT_EOF'
{escaped_script}
SCORING_SCRIPT_EOF
"""

        try:
            result = await self.instance.aexec(
                command,
                ssh_key_path=self.ssh_key_path,
                timeout=int(timeout),
            )

            return _parse_scoring_output(result.stdout, result.stderr, 0 if result.success else 1)

        except Exception as e:
            return {
                "compiled": 0.0,
                "correct": 0.0,
                "speedup": 0.0,
                "reward": 0.0,
                "pass_rate": 0.0,
                "error": f"SSH error: {e}",
            }

    async def close(self) -> None:
        """Terminate the instance (unless keep_alive)."""
        if not self.keep_alive:
            try:
                await self.instance.terminate()
            except Exception as e:
                logger.warning(f"Failed to terminate instance {self.instance.id}: {e}")

    async def health_check(self) -> bool:
        """Check if instance is reachable."""
        try:
            result = await self.instance.aexec(
                "echo ok",
                ssh_key_path=self.ssh_key_path,
                timeout=10,
            )
            return result.success and "ok" in result.stdout
        except Exception:
            return False

    async def describe_runtime(self) -> dict[str, Any]:
        """Describe the remote runtime used for scoring."""
        escaped_script = _build_runtime_probe_script().replace("'", "'\"'\"'")
        command = f"""
python3 << 'RUNTIME_PROBE_EOF'
{escaped_script}
RUNTIME_PROBE_EOF
"""
        try:
            result = await self.instance.aexec(
                command,
                ssh_key_path=self.ssh_key_path,
                timeout=30,
            )
            return _parse_runtime_probe(
                result.stdout,
                result.stderr,
                0 if result.success else 1,
                worker_kind="broker",
            )
        except Exception as e:
            return {
                "worker_kind": "broker",
                "runtime_ok": False,
                "error": f"Runtime probe failed: {e}",
            }


def _build_runtime_provenance_snippet() -> str:
    """Build Python code that prints scorer runtime provenance as JSON."""
    return """
def _run_optional_command(command):
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None

    output = completed.stdout.strip()
    return output or None


def _collect_runtime_provenance():
    provenance = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "hip_version": getattr(torch.version, "hip", None),
            "cuda_available": torch.cuda.is_available(),
        },
    }

    torch_info = provenance["torch"]
    if torch.cuda.is_available():
        try:
            torch_info["device_count"] = torch.cuda.device_count()
        except Exception:
            pass

        try:
            torch_info["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

        try:
            capability = torch.cuda.get_device_capability(0)
        except Exception:
            capability = None
        if capability is not None:
            torch_info["device_capability"] = list(capability)

        driver_version = _run_optional_command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        if driver_version:
            torch_info["driver_version"] = driver_version.splitlines()[0]

    return provenance


try:
    print(f"PROVENANCE_RESULT:{json.dumps(_collect_runtime_provenance(), sort_keys=True)}")
except Exception as e:
    print(f"PROVENANCE_ERROR:{e}")
"""


def _build_runtime_probe_script() -> str:
    """Build a small Python script that reports sandbox runtime capabilities."""
    return r"""
import json
import platform
import sys

result = {
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "runtime_ok": True,
    "torch": {
        "available": False,
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "hip_version": None,
        "device_count": 0,
        "device_name": None,
    },
    "errors": [],
}

try:
    import torch
except Exception as e:
    result["runtime_ok"] = False
    result["errors"].append(f"import torch failed: {e!r}")
else:
    result["torch"]["available"] = True
    result["torch"]["version"] = torch.__version__
    result["torch"]["cuda_available"] = torch.cuda.is_available()
    result["torch"]["cuda_version"] = torch.version.cuda
    result["torch"]["hip_version"] = getattr(torch.version, "hip", None)
    if torch.cuda.is_available():
        try:
            result["torch"]["device_count"] = torch.cuda.device_count()
        except Exception as e:
            result["runtime_ok"] = False
            result["errors"].append(f"torch.cuda.device_count failed: {e!r}")
        try:
            result["torch"]["device_name"] = torch.cuda.get_device_name(0)
        except Exception as e:
            result["runtime_ok"] = False
            result["errors"].append(f"torch.cuda.get_device_name failed: {e!r}")

print(json.dumps(result, sort_keys=True))
"""


def _parse_runtime_probe(
    stdout: str,
    stderr: str,
    returncode: int,
    *,
    worker_kind: str,
) -> dict[str, Any]:
    """Parse runtime probe output into a stable description."""
    import json

    if returncode != 0:
        return {
            "worker_kind": worker_kind,
            "runtime_ok": False,
            "error": _tail_text(stderr, limit=500)
            or _tail_text(stdout, limit=500)
            or f"Runtime probe failed with return code {returncode}",
        }

    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "worker_kind": worker_kind,
            "runtime_ok": False,
            "error": "Runtime probe produced no output",
        }

    try:
        parsed = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {
            "worker_kind": worker_kind,
            "runtime_ok": False,
            "error": _tail_text(stdout, limit=500) or "Failed to parse runtime probe JSON",
        }

    parsed["worker_kind"] = worker_kind
    return parsed


def _build_scoring_script_body(
    kernel_loader_snippet: str,
    ref_code: str,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    num_correct_seeds: int = 5,
    warmup_runs: int = 5,
    timed_runs: int = 30,
    check_determinism: bool = True,
    adaptive_baseline: bool = True,
    excessive_speedup_threshold: float = 10.0,
    forbidden_ops: tuple[str, ...] = (),
) -> str:
    """Build the shared scoring script body.

    Pipeline stages (each gates the next on failure):
      1. static checks   - STATIC_CHECK_RESULT
      2. compile         - COMPILE_RESULT
      3. correctness     - CORRECTNESS_RESULT  (N seeds, NaN/Inf, shape, dtype tolerance)
      4. determinism     - DETERMINISM_RESULT  (2 runs, torch.equal)
      5. baseline        - BASELINE_RESULT     (adaptive: eager vs torch.compile)
      6. perf            - PERF_RESULT         (CUDA events, median/p10/p90)
      7. excessive speed - EXCESSIVE_SPEEDUP   (post-perf sanity flag, non-gating)

    kernel_loader_snippet: Python code that defines ModelNew and _kernel_source_code
      (differs between remote/base64 and local/file-path variants).
    """
    runtime_provenance = _build_runtime_provenance_snippet()
    forbidden_ops_json = repr(list(forbidden_ops))
    return f"""
import sys
import re
import os
import gc
import json
import base64
import tempfile
import platform
import subprocess
import importlib.util
import statistics

# Set CUDA_HOME if not set (common Modal/container issue)
if "CUDA_HOME" not in os.environ:
    for cuda_path in ["/usr/local/cuda", "/usr/cuda", "/opt/cuda"]:
        if os.path.exists(cuda_path):
            os.environ["CUDA_HOME"] = cuda_path
            break

# ── Reference code (defines Model, get_inputs, get_init_inputs) ──────────────

{ref_code}

# ── Kernel loader (defines ModelNew and _kernel_source_code) ─────────────────

{kernel_loader_snippet}

import torch
{runtime_provenance}

def _graceful_cleanup(device=None):
    try:
        gc.collect()
        if torch.cuda.is_available():
            if device is None:
                device = torch.device("cuda")
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=device)
                torch.cuda.synchronize(device=device)
    except Exception:
        pass

def _dtype_to_precision(dtype):
    s = str(dtype)
    if "float8" in s: return "fp8"
    if "bfloat16" in s: return "bf16"
    if "float16" in s: return "fp16"
    if "float32" in s: return "fp32"
    if "float64" in s: return "fp64"
    return s.replace("torch.", "")

_PRECISION_TOLERANCES = {{
    "fp8":  {{"atol": 0.1,   "rtol": 0.05}},
    "fp16": {{"atol": 0.01,  "rtol": 0.01}},
    "bf16": {{"atol": 0.01,  "rtol": 0.01}},
    "fp32": {{"atol": 0.001, "rtol": 0.001}},
    "fp64": {{"atol": 1e-5,  "rtol": 1e-5}},
}}

def _get_tolerance(precision):
    # Use config-provided values as floor, upstream precision table as guide.
    # We take the looser of the two so config overrides never tighten below
    # what the precision requires.
    base = _PRECISION_TOLERANCES.get(precision, {{"atol": 0.05, "rtol": 0.02}})
    return {{"atol": max({atol}, base["atol"]), "rtol": max({rtol}, base["rtol"])}}

def _cuda_event_time_ms(model, inputs, n):
    times = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model(*inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Stage 1: Static checks ────────────────────────────────────────────────
    _forbidden_ops = {forbidden_ops_json}
    for _pattern in _forbidden_ops:
        _m = re.search(_pattern, _kernel_source_code)
        if _m:
            print(json.dumps({{"stage": "static_check", "passed": False,
                               "error": f"Forbidden pattern: {{_m.group(0).strip()}}"}}))
            print("STATIC_CHECK_RESULT:FAIL")
            sys.exit(0)
    print("STATIC_CHECK_RESULT:PASS")

    # ── Stage 2: Compile ──────────────────────────────────────────────────────
    # (ModelNew already loaded by kernel_loader_snippet above)
    if "ModelNew" not in dir():
        print(json.dumps({{"stage": "compile", "passed": False, "error": "ModelNew not defined"}}))
        print("COMPILE_RESULT:FAIL")
        sys.exit(0)
    print("COMPILE_RESULT:PASS")

    # ── Stage 3: Correctness ──────────────────────────────────────────────────
    model_ref = Model(*get_init_inputs()).to(device).eval()
    model_new = ModelNew(*get_init_inputs()).to(device).eval()

    _CORRECTNESS_SEEDS = list(range({num_correct_seeds}))
    _CORRECTNESS_SEEDS = [42, 123, 456, 789, 1337][:{num_correct_seeds}]
    passed = 0
    precision = "fp32"
    tol = _get_tolerance(precision)
    worst_diff = 0.0

    for _seed in _CORRECTNESS_SEEDS:
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]

        for _v in inputs:
            if isinstance(_v, torch.Tensor):
                precision = _dtype_to_precision(_v.dtype)
                break
        tol = _get_tolerance(precision)

        with torch.no_grad():
            ref_out = model_ref(*inputs)
            new_out = model_new(*inputs)

        if not isinstance(ref_out, torch.Tensor) or not isinstance(new_out, torch.Tensor):
            print(json.dumps({{"stage": "correctness", "passed": False,
                               "error": "non-tensor output", "precision": precision}}))
            print("CORRECTNESS_RESULT:FAIL")
            sys.exit(0)

        if ref_out.shape != new_out.shape:
            print(json.dumps({{"stage": "correctness", "passed": False,
                               "error": f"shape mismatch: {{tuple(ref_out.shape)}} vs {{tuple(new_out.shape)}}",
                               "precision": precision}}))
            print("CORRECTNESS_RESULT:FAIL")
            sys.exit(0)

        if torch.isnan(new_out).any() or torch.isinf(new_out).any():
            print(json.dumps({{"stage": "correctness", "passed": False,
                               "error": "NaN or Inf in solution output", "precision": precision}}))
            print("CORRECTNESS_RESULT:FAIL")
            sys.exit(0)

        ref_f, new_f = ref_out.float(), new_out.float()
        diff = (ref_f - new_f).abs().max().item()
        worst_diff = max(worst_diff, diff)
        if torch.allclose(ref_f, new_f, atol=tol["atol"], rtol=tol["rtol"]):
            passed += 1
        else:
            print(json.dumps({{"stage": "correctness", "passed": False,
                               "error": f"seed={{_seed}} max_diff={{diff:.6f}}",
                               "atol": tol["atol"], "rtol": tol["rtol"],
                               "precision": precision}}))
            print("CORRECTNESS_RESULT:FAIL")
            sys.exit(0)

    print(json.dumps({{"stage": "correctness", "passed": True, "seeds": len(_CORRECTNESS_SEEDS),
                       "worst_diff": worst_diff, "atol": tol["atol"], "rtol": tol["rtol"],
                       "precision": precision}}))
    print(f"CORRECTNESS_RESULT:{{passed}}/{{len(_CORRECTNESS_SEEDS)}}")

    # ── Stage 4: Determinism ──────────────────────────────────────────────────
    _check_determinism = {str(check_determinism).lower() == "true"}
    if _check_determinism:
        torch.manual_seed(2026)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(2026)
        _det_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
        with torch.no_grad():
            _out_a = model_new(*_det_inputs).clone()
            _out_b = model_new(*_det_inputs).clone()
        if not torch.equal(_out_a, _out_b):
            print(json.dumps({{"stage": "determinism", "passed": False,
                               "error": "non-deterministic output (possible race condition)"}}))
            print("DETERMINISM_RESULT:FAIL")
            sys.exit(0)
    print("DETERMINISM_RESULT:PASS")

    # ── Stage 5: Baseline ─────────────────────────────────────────────────────
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    bench_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]

    baseline_type = "pytorch_eager"
    _adaptive = {str(adaptive_baseline).lower() == "true"}
    if _adaptive:
        try:
            _compiled = torch.compile(model_ref, mode="reduce-overhead")
            for _ in range(3):
                with torch.no_grad():
                    _compiled(*bench_inputs)
            torch.cuda.synchronize()
            _eager_ms = statistics.median(_cuda_event_time_ms(model_ref, bench_inputs, 10))
            _compile_ms = statistics.median(_cuda_event_time_ms(_compiled, bench_inputs, 10))
            if _compile_ms < _eager_ms * 0.95:
                model_ref = _compiled
                baseline_type = "torch_compile"
        except Exception:
            pass
    print(json.dumps({{"stage": "baseline", "baseline_type": baseline_type}}))
    print(f"BASELINE_RESULT:{{baseline_type}}")

    # ── Stage 6: Perf ─────────────────────────────────────────────────────────
    NUM_WARMUP = {warmup_runs}
    NUM_RUNS = {timed_runs}

    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            model_ref(*bench_inputs)
            model_new(*bench_inputs)
    torch.cuda.synchronize()

    ref_times = sorted(_cuda_event_time_ms(model_ref, bench_inputs, NUM_RUNS))
    new_times = sorted(_cuda_event_time_ms(model_new, bench_inputs, NUM_RUNS))

    n = len(ref_times)
    p10 = int(0.10 * (n - 1))
    p90 = int(0.90 * (n - 1))
    ref_ms = statistics.median(ref_times)
    new_ms = statistics.median(new_times)
    speedup = ref_ms / new_ms if new_ms > 0 else 0.0

    perf = {{
        "stage": "perf", "speedup": speedup,
        "ref_ms": ref_ms, "sol_ms": new_ms,
        "ref_p10_ms": ref_times[p10], "ref_p90_ms": ref_times[p90],
        "sol_p10_ms": new_times[p10], "sol_p90_ms": new_times[p90],
        "ref_std_ms": statistics.pstdev(ref_times), "sol_std_ms": statistics.pstdev(new_times),
        "baseline_type": baseline_type, "warmup_runs": NUM_WARMUP, "timed_runs": NUM_RUNS,
    }}
    print(json.dumps(perf))
    print(f"SPEEDUP_RESULT:{{speedup:.4f}}")

    # ── Stage 7: Excessive speedup ────────────────────────────────────────────
    _threshold = {excessive_speedup_threshold}
    if speedup > _threshold:
        print(json.dumps({{"stage": "excessive_speedup", "speedup": speedup,
                           "threshold": _threshold,
                           "warning": f"speedup {{speedup:.2f}}x exceeds threshold {{_threshold}}x — verify kernel is not reward hacking"}}))
        print(f"EXCESSIVE_SPEEDUP:{{speedup:.4f}}")

except Exception as _exc:
    import traceback
    traceback.print_exc()
    print(json.dumps({{"stage": "error", "error": str(_exc)}}))
finally:
    _graceful_cleanup(locals().get("device"))
"""


def _build_scoring_script_for_remote(
    ref_code: str,
    kernel_b64: str,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    num_correct_seeds: int = 5,
    warmup_runs: int = 5,
    timed_runs: int = 30,
    check_determinism: bool = True,
    adaptive_baseline: bool = True,
    excessive_speedup_threshold: float = 10.0,
    forbidden_ops: tuple[str, ...] = (),
) -> str:
    """Build scoring script for remote execution with base64-encoded kernel."""
    kernel_loader = f'''
_kernel_b64 = "{kernel_b64}"
_kernel_source_code = base64.b64decode(_kernel_b64).decode()
_kernel_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
_kernel_path = _kernel_file.name
_kernel_file.write(_kernel_source_code)
_kernel_file.close()
try:
    _spec = importlib.util.spec_from_file_location("generated_candidate_module", _kernel_path)
    _module = importlib.util.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_module)
    ModelNew = _module.ModelNew
    print("COMPILE_RESULT:PASS")
except Exception as _e:
    print(json.dumps({{"stage": "compile", "passed": False, "error": str(_e)}}))
    print("COMPILE_RESULT:FAIL")
    try:
        os.remove(_kernel_path)
    except OSError:
        pass
    sys.exit(0)
'''
    return _build_scoring_script_body(
        kernel_loader,
        ref_code,
        atol=atol,
        rtol=rtol,
        num_correct_seeds=num_correct_seeds,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        check_determinism=check_determinism,
        adaptive_baseline=adaptive_baseline,
        excessive_speedup_threshold=excessive_speedup_threshold,
        forbidden_ops=forbidden_ops,
    )


def _build_scoring_script(
    kernel_code: str,
    ref_code: str,
    kernel_file_path: str,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    num_correct_seeds: int = 5,
    warmup_runs: int = 5,
    timed_runs: int = 30,
    check_determinism: bool = True,
    adaptive_baseline: bool = True,
    excessive_speedup_threshold: float = 10.0,
    forbidden_ops: tuple[str, ...] = (),
) -> str:
    """Build a standalone scoring script for local subprocess execution."""
    kernel_loader = f'''
_kernel_source_code = open("{kernel_file_path}").read()
try:
    _spec = importlib.util.spec_from_file_location("generated_candidate_module", "{kernel_file_path}")
    _module = importlib.util.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_module)
    ModelNew = _module.ModelNew
    print("COMPILE_RESULT:PASS")
except Exception as _e:
    print(json.dumps({{"stage": "compile", "passed": False, "error": str(_e)}}))
    print("COMPILE_RESULT:FAIL")
    sys.exit(0)
'''
    return _build_scoring_script_body(
        kernel_loader,
        ref_code,
        atol=atol,
        rtol=rtol,
        num_correct_seeds=num_correct_seeds,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        check_determinism=check_determinism,
        adaptive_baseline=adaptive_baseline,
        excessive_speedup_threshold=excessive_speedup_threshold,
        forbidden_ops=forbidden_ops,
    )


def _indent(code: str, prefix: str) -> str:
    """Indent all lines of code with prefix."""
    lines = code.split("\n")
    # Don't indent empty lines
    return "\n".join(prefix + line if line.strip() else line for line in lines)


def _tail_text(text: str, limit: int = 2000) -> str | None:
    """Keep a bounded tail of debug output for observability."""
    stripped = text.strip()
    if not stripped:
        return None
    if len(stripped) <= limit:
        return stripped
    return stripped[-limit:]


def _parse_scoring_output(stdout: str, stderr: str, returncode: int) -> dict[str, Any]:
    """Parse output from the staged scoring script.

    Reads structured stage tags emitted by _build_scoring_script_body:
      STATIC_CHECK_RESULT:PASS|FAIL
      COMPILE_RESULT:PASS|FAIL
      CORRECTNESS_RESULT:N/M  or  CORRECTNESS_RESULT:FAIL
      DETERMINISM_RESULT:PASS|FAIL
      BASELINE_RESULT:<baseline_type>
      SPEEDUP_RESULT:<float>
      EXCESSIVE_SPEEDUP:<float>
      PROVENANCE_RESULT:{...}

    JSON blobs on their own lines carry per-stage detail (precision, diffs, etc.).
    """
    import json
    import re

    result: dict[str, Any] = {
        "compiled": 0.0,
        "correct": 0.0,
        "speedup": 0.0,
        "pass_rate": 0.0,
        "error": None,
        "baseline_type": None,
        "precision": None,
        "worst_diff": None,
        "is_deterministic": None,
        "excessive_speedup": False,
        "perf_detail": None,
        "runtime_provenance": None,
        "debug_stdout_tail": _tail_text(stdout),
        "debug_stderr_tail": _tail_text(stderr),
        "returncode": returncode,
    }

    # Parse per-stage JSON detail lines
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                blob = json.loads(line)
            except json.JSONDecodeError:
                continue
            stage = blob.get("stage")
            if stage == "correctness":
                result["precision"] = blob.get("precision")
                result["worst_diff"] = blob.get("worst_diff")
            elif stage == "baseline":
                result["baseline_type"] = blob.get("baseline_type")
            elif stage == "perf":
                result["perf_detail"] = blob
                result["baseline_type"] = blob.get("baseline_type")
            elif stage == "excessive_speedup":
                result["excessive_speedup"] = True
                logger.warning(blob.get("warning", "Excessive speedup detected"))

    # Provenance
    m = re.search(r"PROVENANCE_RESULT:(\{.*\})", stdout)
    if m:
        try:
            result["runtime_provenance"] = json.loads(m.group(1))
        except json.JSONDecodeError:
            logger.warning("Failed to parse runtime provenance")

    # Stage 1: static checks
    if "STATIC_CHECK_RESULT:FAIL" in stdout:
        m = re.search(r'"error":\s*"([^"]+)"', stdout)
        result["error"] = m.group(1) if m else "Forbidden op pattern detected"
        return result

    # Stage 2: compile
    if "COMPILE_RESULT:PASS" in stdout:
        result["compiled"] = 1.0
    elif "COMPILE_RESULT:FAIL" in stdout:
        m = re.search(r'"error":\s*"([^"]+)"', stdout)
        result["error"] = m.group(1) if m else "Compilation failed"
        return result

    # Stage 3: correctness
    m = re.search(r"CORRECTNESS_RESULT:(\d+)/(\d+)", stdout)
    if m:
        passed, total = int(m.group(1)), int(m.group(2))
        result["pass_rate"] = passed / total
        if passed == total:
            result["correct"] = 1.0
    elif "CORRECTNESS_RESULT:FAIL" in stdout:
        m = re.search(r'"error":\s*"([^"]+)"', stdout)
        result["error"] = m.group(1) if m else "Correctness check failed"
        return result

    # Stage 4: determinism
    if "DETERMINISM_RESULT:PASS" in stdout:
        result["is_deterministic"] = True
    elif "DETERMINISM_RESULT:FAIL" in stdout:
        result["is_deterministic"] = False
        result["error"] = "Non-deterministic output"
        return result

    # Stage 5: baseline (already parsed from JSON blob above)
    if "BASELINE_RESULT:" in stdout and result["baseline_type"] is None:
        m = re.search(r"BASELINE_RESULT:(\S+)", stdout)
        if m:
            result["baseline_type"] = m.group(1)

    # Stage 6: perf
    m = re.search(r"SPEEDUP_RESULT:([\d.]+)", stdout)
    if m:
        result["speedup"] = float(m.group(1))

    if result["error"] is None and returncode != 0:
        result["error"] = (
            _tail_text(stderr, limit=500)
            or _tail_text(stdout, limit=500)
            or f"Scoring subprocess failed with return code {returncode}"
        )

    return result
