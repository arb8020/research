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
    ) -> dict[str, Any]:
        """Score kernel in subprocess using trio threading."""

        def _run_subprocess() -> tuple[str, str, int]:
            """Run scoring in subprocess (blocking, runs in thread)."""
            # Write kernel code to a separate file to avoid string escaping issues
            # (kernel code often contains triple quotes for CUDA sources)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as kf:
                kf.write(kernel_code)
                kernel_path = kf.name

            script = _build_scoring_script(kernel_code, ref_code, kernel_path)

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
    ) -> dict[str, Any]:
        """Score kernel on remote instance via SSH."""
        import base64

        # Encode kernel code as base64 to avoid escaping issues
        kernel_b64 = base64.b64encode(kernel_code.encode()).decode()

        # Build script that decodes kernel from base64 and writes to temp file
        script = _build_scoring_script_for_remote(ref_code, kernel_b64)

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
    return r'''
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
'''


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


def _build_scoring_script_for_remote(ref_code: str, kernel_b64: str) -> str:
    """Build scoring script for remote execution with base64-encoded kernel.

    Uses base64 encoding to safely transmit kernel code through SSH heredoc
    without escaping issues from triple quotes in CUDA source strings.
    """
    runtime_provenance = _build_runtime_provenance_snippet()
    return f'''
import sys
import time
import base64
import tempfile
import os
import json
import platform
import subprocess
import importlib.util

# Set CUDA_HOME if not set (common Modal/container issue)
if "CUDA_HOME" not in os.environ:
    for cuda_path in ["/usr/local/cuda", "/usr/cuda", "/opt/cuda"]:
        if os.path.exists(cuda_path):
            os.environ["CUDA_HOME"] = cuda_path
            break

# ─────────────────────────────────────────────────────────────────────────────
# Reference code (defines Model, get_inputs, get_init_inputs)
# ─────────────────────────────────────────────────────────────────────────────

{ref_code}

# ─────────────────────────────────────────────────────────────────────────────
# Generated kernel code (defines ModelNew) - decoded from base64
# ─────────────────────────────────────────────────────────────────────────────

_kernel_b64 = "{kernel_b64}"
_kernel_code = base64.b64decode(_kernel_b64).decode()
_kernel_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
_kernel_path = _kernel_file.name
_kernel_file.write(_kernel_code)
_kernel_file.close()

try:
    spec = importlib.util.spec_from_file_location("kernelbench_candidate", _kernel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    ModelNew = module.ModelNew
    print("COMPILE_SUCCESS")
except Exception as e:
    print(f"COMPILE_ERROR:{{e}}")
    try:
        os.remove(_kernel_path)
    except OSError:
        pass
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Verify ModelNew exists
# ─────────────────────────────────────────────────────────────────────────────

if "ModelNew" not in globals():
    print("COMPILE_ERROR:ModelNew class not defined")
    try:
        os.remove(_kernel_path)
    except OSError:
        pass
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Test correctness
# ─────────────────────────────────────────────────────────────────────────────

import torch
{runtime_provenance}

try:
    model_ref = Model(*get_init_inputs())
    model_new = ModelNew(*get_init_inputs())
    model_ref.eval()
    model_new.eval()

    if torch.cuda.is_available():
        model_ref = model_ref.cuda()
        model_new = model_new.cuda()

    NUM_TESTS = 3
    passed = 0
    for i in range(NUM_TESTS):
        inputs = get_inputs()
        if torch.cuda.is_available():
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

        with torch.no_grad():
            ref_out = model_ref(*inputs)
            new_out = model_new(*inputs)

        if torch.allclose(ref_out, new_out, rtol=1e-3, atol=1e-3):
            passed += 1

    print(f"CORRECTNESS_RESULT:{{passed}}/{{NUM_TESTS}}")

    if passed < NUM_TESTS:
        sys.exit(0)

except Exception as e:
    print(f"CORRECTNESS_ERROR:{{e}}")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark speedup (only if fully correct)
# ─────────────────────────────────────────────────────────────────────────────

try:
    NUM_WARMUP = 3
    NUM_RUNS = 10

    inputs = get_inputs()
    if torch.cuda.is_available():
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = model_ref(*inputs)
            _ = model_new(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NUM_RUNS):
        with torch.no_grad():
            _ = model_ref(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    ref_time = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NUM_RUNS):
        with torch.no_grad():
            _ = model_new(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    new_time = time.perf_counter() - t0

    speedup = ref_time / new_time if new_time > 0 else 0.0
    print(f"SPEEDUP_RESULT:{{speedup:.4f}}")

except Exception as e:
    print(f"BENCHMARK_ERROR:{{e}}")
finally:
    try:
        os.remove(_kernel_path)
    except OSError:
        pass
'''


def _build_scoring_script(kernel_code: str, ref_code: str, kernel_file_path: str) -> str:
    """Build a standalone Python script that scores a kernel.

    The script:
    1. Executes ref_code to define Model, get_inputs, get_init_inputs
    2. Loads and executes kernel_code from a separate file to define ModelNew
    3. Tests correctness (torch.allclose)
    4. Benchmarks speedup if correct
    5. Prints results in parseable format

    Args:
        kernel_code: Generated kernel code (unused here, written to kernel_file_path separately)
        ref_code: Reference code with Model, get_inputs, get_init_inputs
        kernel_file_path: Path to the file containing kernel_code
    """
    runtime_provenance = _build_runtime_provenance_snippet()
    return f'''
import sys
import time
import os
import json
import platform
import subprocess
import importlib.util

# Set CUDA_HOME if not set (common Modal/container issue)
if "CUDA_HOME" not in os.environ:
    for cuda_path in ["/usr/local/cuda", "/usr/cuda", "/opt/cuda"]:
        if os.path.exists(cuda_path):
            os.environ["CUDA_HOME"] = cuda_path
            break

# ─────────────────────────────────────────────────────────────────────────────
# Reference code (defines Model, get_inputs, get_init_inputs)
# ─────────────────────────────────────────────────────────────────────────────

{ref_code}

# ─────────────────────────────────────────────────────────────────────────────
# Generated kernel code (defines ModelNew)
# ─────────────────────────────────────────────────────────────────────────────

try:
    spec = importlib.util.spec_from_file_location("kernelbench_candidate", "{kernel_file_path}")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    ModelNew = module.ModelNew
    print("COMPILE_SUCCESS")
except Exception as e:
    print(f"COMPILE_ERROR:{{e}}")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Verify ModelNew exists
# ─────────────────────────────────────────────────────────────────────────────

if "ModelNew" not in globals():
    print("COMPILE_ERROR:ModelNew class not defined")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Test correctness
# ─────────────────────────────────────────────────────────────────────────────

import torch
{runtime_provenance}

try:
    model_ref = Model(*get_init_inputs())
    model_new = ModelNew(*get_init_inputs())
    model_ref.eval()
    model_new.eval()

    if torch.cuda.is_available():
        model_ref = model_ref.cuda()
        model_new = model_new.cuda()

    NUM_TESTS = 3
    passed = 0
    for i in range(NUM_TESTS):
        inputs = get_inputs()
        if torch.cuda.is_available():
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

        with torch.no_grad():
            ref_out = model_ref(*inputs)
            new_out = model_new(*inputs)

        if torch.allclose(ref_out, new_out, rtol=1e-3, atol=1e-3):
            passed += 1

    print(f"CORRECTNESS_RESULT:{{passed}}/{{NUM_TESTS}}")

    if passed < NUM_TESTS:
        # Not fully correct, don't benchmark
        sys.exit(0)

except Exception as e:
    print(f"CORRECTNESS_ERROR:{{e}}")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark speedup (only if fully correct)
# ─────────────────────────────────────────────────────────────────────────────

try:
    NUM_WARMUP = 3
    NUM_RUNS = 10

    inputs = get_inputs()
    if torch.cuda.is_available():
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

    # Warmup
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = model_ref(*inputs)
            _ = model_new(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark reference
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NUM_RUNS):
        with torch.no_grad():
            _ = model_ref(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    ref_time = time.perf_counter() - t0

    # Benchmark new
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NUM_RUNS):
        with torch.no_grad():
            _ = model_new(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    new_time = time.perf_counter() - t0

    speedup = ref_time / new_time if new_time > 0 else 0.0
    print(f"SPEEDUP_RESULT:{{speedup:.4f}}")

except Exception as e:
    print(f"BENCHMARK_ERROR:{{e}}")
'''


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
    """Parse output from scoring script."""
    import json
    import re

    result = {
        "compiled": 0.0,
        "correct": 0.0,
        "speedup": 0.0,
        "reward": 0.0,
        "pass_rate": 0.0,
        "error": None,
        "runtime_provenance": None,
        "debug_stdout_tail": _tail_text(stdout),
        "debug_stderr_tail": _tail_text(stderr),
        "returncode": returncode,
    }

    provenance_match = re.search(r"PROVENANCE_RESULT:(\{.*\})", stdout)
    if provenance_match:
        try:
            result["runtime_provenance"] = json.loads(provenance_match.group(1))
        except json.JSONDecodeError:
            logger.warning("Failed to parse runtime provenance from scoring output")

    # Check for compile success
    if "COMPILE_SUCCESS" in stdout:
        result["compiled"] = 1.0
    elif "COMPILE_ERROR" in stdout:
        match = re.search(r"COMPILE_ERROR:(.*)", stdout)
        result["error"] = match.group(1) if match else "Compilation failed"
        return result

    # Check correctness
    match = re.search(r"CORRECTNESS_RESULT:(\d+)/(\d+)", stdout)
    if match:
        passed, total = int(match.group(1)), int(match.group(2))
        result["pass_rate"] = passed / total
        if passed == total:
            result["correct"] = 1.0
    elif "CORRECTNESS_ERROR" in stdout:
        match = re.search(r"CORRECTNESS_ERROR:(.*)", stdout)
        result["error"] = match.group(1) if match else "Correctness test failed"

    # Check speedup
    match = re.search(r"SPEEDUP_RESULT:([\d.]+)", stdout)
    if match:
        result["speedup"] = float(match.group(1))
    elif "BENCHMARK_ERROR" in stdout:
        match = re.search(r"BENCHMARK_ERROR:(.*)", stdout)
        # Don't set error - kernel is still correct, just couldn't benchmark
        logger.warning(f"Benchmark error: {match.group(1) if match else 'unknown'}")

    if result["error"] is None and returncode != 0:
        result["error"] = (
            _tail_text(stderr, limit=500)
            or _tail_text(stdout, limit=500)
            or f"Scoring subprocess failed with return code {returncode}"
        )

    # Compute reward: 0.2 * compiled + 1.0 * correct + speedup (if correct)
    result["reward"] = (
        0.2 * result["compiled"]
        + 1.0 * result["correct"]
        + (result["speedup"] if result["correct"] > 0 else 0.0)
    )

    return result
