"""SandboxWorker: interface for kernel scoring workers.

Workers can be:
- LocalSandboxWorker: runs scoring in local subprocess
- RemoteSandboxWorker: connects to remote WorkerServer via miniray

All workers implement the same scoring protocol:
    request: {"kernel_code": str, "ref_code": str}
    response: {"compiled": float, "correct": float, "speedup": float, ...}
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass
class LocalSandboxWorker(SandboxWorker):
    """Runs kernel scoring in local subprocess.

    Uses the same scoring logic as scoring.py but in an isolated subprocess
    to avoid polluting the main process with compiled kernels.
    """

    async def score(
        self,
        kernel_code: str,
        ref_code: str,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Score kernel in subprocess."""
        # Build the scoring script
        script = _build_scoring_script(kernel_code, ref_code)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            # Run in subprocess
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {
                    "compiled": 0.0,
                    "correct": 0.0,
                    "speedup": 0.0,
                    "reward": 0.0,
                    "pass_rate": 0.0,
                    "error": "Scoring timeout",
                }

            # Parse output
            return _parse_scoring_output(
                stdout.decode(), stderr.decode(), proc.returncode
            )

        finally:
            Path(script_path).unlink(missing_ok=True)

    async def close(self) -> None:
        """No cleanup needed for local worker."""
        pass

    async def health_check(self) -> bool:
        """Local worker is always healthy."""
        return True


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
        # Build the scoring script
        script = _build_scoring_script(kernel_code, ref_code)

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

            return _parse_scoring_output(
                result.stdout, result.stderr, 0 if result.success else 1
            )

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


def _build_scoring_script(kernel_code: str, ref_code: str) -> str:
    """Build a standalone Python script that scores a kernel.

    The script:
    1. Executes ref_code to define Model, get_inputs, get_init_inputs
    2. Executes kernel_code to define ModelNew
    3. Tests correctness (torch.allclose)
    4. Benchmarks speedup if correct
    5. Prints results in parseable format
    """
    return f'''
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# Reference code (defines Model, get_inputs, get_init_inputs)
# ─────────────────────────────────────────────────────────────────────────────

{ref_code}

# ─────────────────────────────────────────────────────────────────────────────
# Generated kernel code (defines ModelNew)
# ─────────────────────────────────────────────────────────────────────────────

_kernel_code = """
{kernel_code}
"""

try:
    exec(_kernel_code, globals())
    print("COMPILE_SUCCESS")
except Exception as e:
    print(f"COMPILE_ERROR:{{e}}")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Verify ModelNew exists
# ─────────────────────────────────────────────────────────────────────────────

if "ModelNew" not in dir():
    print("COMPILE_ERROR:ModelNew class not defined")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Test correctness
# ─────────────────────────────────────────────────────────────────────────────

import torch

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


def _parse_scoring_output(
    stdout: str, stderr: str, returncode: int
) -> dict[str, Any]:
    """Parse output from scoring script."""
    import re

    result = {
        "compiled": 0.0,
        "correct": 0.0,
        "speedup": 0.0,
        "reward": 0.0,
        "pass_rate": 0.0,
        "error": None,
    }

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

    # Compute reward: 0.2 * compiled + 1.0 * correct + speedup (if correct)
    result["reward"] = (
        0.2 * result["compiled"]
        + 1.0 * result["correct"]
        + (result["speedup"] if result["correct"] > 0 else 0.0)
    )

    return result
