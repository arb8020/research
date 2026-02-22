"""Benchmark runner for Modal.

Runs inference benchmarks on Modal sandboxes.
Reuses the same DepsConfig/HardwareConfig infrastructure as training.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

if TYPE_CHECKING:
    from rollouts.training.configs import DepsConfig

from .config import BenchmarkConfig, BenchmarkResult

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.parent
MODAL_APP_NAME = "rollouts-benchmark"


def _build_modal_image(modal: Any, deps: DepsConfig, gpu_type: str) -> Any:
    """Build Modal image from DepsConfig specification.

    Uses nvidia/cuda base image instead of debian_slim because SGLang
    requires nvcc for JIT kernel compilation.
    """
    # GPU-specific torch index override
    # Use CUDA 12.6 by default for FlashInfer 0.3+ support (has backend= param)
    # Use CUDA 12.8 for Blackwell (B200/GB200)
    if gpu_type in ("B200", "GB200"):
        pip_index = "https://download.pytorch.org/whl/nightly/cu128"
        cuda_version = "12.8.0"
    elif deps.pip_index_url:
        pip_index = deps.pip_index_url
        # Infer CUDA version from pip index URL
        if "cu128" in pip_index:
            cuda_version = "12.8.0"
        elif "cu126" in pip_index:
            cuda_version = "12.6.0"
        else:
            cuda_version = "12.6.0"  # Default to 12.6
    else:
        pip_index = "https://download.pytorch.org/whl/cu126"
        cuda_version = "12.6.0"

    # Use CUDA devel image (includes nvcc) for SGLang JIT compilation
    cuda_image = f"nvidia/cuda:{cuda_version}-devel-ubuntu22.04"
    image = modal.Image.from_registry(cuda_image, add_python=deps.python_version)

    if deps.system_packages:
        image = image.apt_install(*deps.system_packages)

    if deps.pip_packages:
        pip_kwargs: dict[str, Any] = {"index_url": pip_index}
        if deps.pip_extra_index_url:
            pip_kwargs["extra_index_url"] = deps.pip_extra_index_url
        image = image.pip_install(*deps.pip_packages, **pip_kwargs)

    for cmd in deps.bootstrap_commands:
        image = image.run_commands(cmd)

    image = image.env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })

    return image


def _exec_sync(sandbox: Any, command: str, timeout: int = 300) -> tuple[str, str, int]:
    """Execute command in sandbox. Blocking."""
    proc = sandbox.exec("bash", "-c", command, timeout=timeout)

    stdout_lines = []
    for line in proc.stdout:
        stdout_lines.append(line)
        logger.info(f"[sandbox] {line.rstrip()}")

    stderr_lines = []
    for line in proc.stderr:
        stderr_lines.append(line)
        if "error" in line.lower() or "exception" in line.lower():
            logger.warning(f"[sandbox stderr] {line.rstrip()}")

    proc.wait()
    return "".join(stdout_lines), "".join(stderr_lines), proc.returncode


async def _sync_code_to_sandbox(sandbox: Any, local_root: Path) -> str:
    """Sync local code to sandbox via git bundle."""
    workspace = "/workspace/research/rollouts"

    def _sync() -> None:
        with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as f:
            bundle_path = f.name

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(local_root),
                capture_output=True,
                text=True,
                check=True,
            )
            commit = result.stdout.strip()
            logger.info(f"Bundling commit {commit[:8]}...")

            subprocess.run(
                ["git", "bundle", "create", bundle_path, "HEAD"],
                cwd=str(local_root),
                check=True,
                capture_output=True,
            )

            with open(bundle_path, "rb") as f:
                bundle_data = f.read()

            _exec_sync(sandbox, "mkdir -p /workspace", timeout=30)

            logger.info("Uploading bundle...")
            remote_file = sandbox.open("/tmp/repo.bundle", "wb")
            remote_file.write(bundle_data)
            remote_file.close()

            _exec_sync(
                sandbox,
                "cd /workspace && git clone /tmp/repo.bundle research && "
                "cd research && git checkout HEAD",
                timeout=120,
            )
            logger.info(f"Code synced to {workspace}")

        finally:
            os.unlink(bundle_path)

    await trio.to_thread.run_sync(_sync)
    return workspace


async def _run_benchmark_in_sandbox(
    sandbox: Any,
    workspace: str,
    config: BenchmarkConfig,
    run_name: str,
) -> BenchmarkResult:
    """Run benchmark inside Modal sandbox."""

    # Install deps
    logger.info("Installing dependencies...")

    def _install() -> None:
        _exec_sync(
            sandbox,
            f"cd {workspace} && pip install httpx aiohttp numpy --quiet",
            timeout=300,
        )

    await trio.to_thread.run_sync(_install)

    # Generate benchmark script
    benchmark_script = f'''
import asyncio
import json
import time
import random
import string
import numpy as np
import httpx

# Config
MODEL = "{config.model}"
BACKEND = "{config.backend}"
NUM_PROMPTS = {config.workload.num_prompts}
INPUT_LEN = {config.workload.input_len}
OUTPUT_LEN = {config.workload.output_len}
CONCURRENCY = {config.workload.concurrency}
MAX_BATCH_SIZE = {config.max_batch_size}
MEM_FRACTION = {config.mem_fraction}

def generate_random_prompt(length: int) -> str:
    """Generate random text of approximately the given token length."""
    # ~4 chars per token
    chars = length * 4
    return "".join(random.choices(string.ascii_letters + " ", k=chars))

async def start_server():
    """Start the inference server."""
    import subprocess
    import sys

    if BACKEND == "engine_v2":
        cmd = [
            sys.executable, "-m", "rollouts.inference.server",
            "--model", MODEL,
            "--port", "30000",
            "--max-batch-size", str(MAX_BATCH_SIZE),
            "--max-seq-len", "2048",  # Limit to fit in GPU memory
        ]
    elif BACKEND == "sglang":
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", MODEL,
            "--port", "30000",
            "--mem-fraction-static", str(MEM_FRACTION),
            "--attention-backend", "triton",  # Required: Modal doesn't have nvcc for FlashInfer JIT
            "--disable-cuda-graph",  # Also disable CUDA graphs which need nvcc
        ]
    else:
        raise ValueError(f"Unknown backend: {{BACKEND}}")

    print(f"Starting server: {{' '.join(cmd)}}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for server to be ready
    for i in range(120):  # 2 min timeout
        # Check if process died
        poll = proc.poll()
        if poll is not None:
            stdout, stderr = proc.communicate()
            print(f"Server process exited with code {{poll}}")
            print(f"stdout: {{stdout.decode()}}")
            print(f"stderr: {{stderr.decode()}}")
            raise RuntimeError(f"Server process died with exit code {{poll}}")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:30000/health", timeout=2.0)
                if resp.status_code == 200:
                    print("Server ready")
                    return proc
        except Exception as e:
            if i % 10 == 0:
                print(f"Waiting for server... ({{i}}s, {{e.__class__.__name__}})")
        await asyncio.sleep(1)

    # Server didn't respond but process is still running - kill and report
    proc.terminate()
    stdout, stderr = proc.communicate(timeout=5)
    print(f"Server timeout. stdout: {{stdout.decode()[:2000]}}")
    print(f"Server timeout. stderr: {{stderr.decode()[:2000]}}")
    raise RuntimeError("Server failed to start within timeout")

async def benchmark_request(client: httpx.AsyncClient, prompt: str, request_id: int) -> dict:
    """Send a single benchmark request and measure timing."""
    start = time.perf_counter()
    first_token_time = None
    tokens_received = 0

    async with client.stream(
        "POST",
        "http://localhost:30000/v1/chat/completions",
        json={{
            "model": MODEL,
            "messages": [{{"role": "user", "content": prompt}}],
            "max_tokens": OUTPUT_LEN,
            "temperature": 0.0,
            "stream": True,
        }},
        timeout=120.0,
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                tokens_received += 1

    end = time.perf_counter()

    return {{
        "request_id": request_id,
        "ttft_ms": (first_token_time - start) * 1000 if first_token_time else None,
        "e2e_ms": (end - start) * 1000,
        "tokens": tokens_received,
    }}

async def run_benchmark():
    """Run the benchmark."""
    proc = await start_server()

    try:
        # Generate prompts
        prompts = [generate_random_prompt(INPUT_LEN) for _ in range(NUM_PROMPTS)]

        # Run requests with concurrency limit
        semaphore = asyncio.Semaphore(CONCURRENCY)
        results = []

        async def bounded_request(client, prompt, idx):
            async with semaphore:
                return await benchmark_request(client, prompt, idx)

        start_time = time.perf_counter()

        async with httpx.AsyncClient() as client:
            tasks = [bounded_request(client, p, i) for i, p in enumerate(prompts)]
            results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

        # Calculate metrics
        ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
        e2es = [r["e2e_ms"] for r in results]
        total_tokens = sum(r["tokens"] for r in results)

        # TPOT = (e2e - ttft) / (tokens - 1)
        tpots = []
        for r in results:
            if r["ttft_ms"] and r["tokens"] > 1:
                tpots.append((r["e2e_ms"] - r["ttft_ms"]) / (r["tokens"] - 1))

        def percentile(data, p):
            return float(np.percentile(data, p)) if data else 0.0

        metrics = {{
            "requests_per_second": NUM_PROMPTS / total_time,
            "tokens_per_second": total_tokens / total_time,
            "output_tokens_per_second": total_tokens / total_time,
            "ttft_mean": float(np.mean(ttfts)) if ttfts else 0.0,
            "ttft_p50": percentile(ttfts, 50),
            "ttft_p95": percentile(ttfts, 95),
            "ttft_p99": percentile(ttfts, 99),
            "tpot_mean": float(np.mean(tpots)) if tpots else 0.0,
            "tpot_p50": percentile(tpots, 50),
            "tpot_p95": percentile(tpots, 95),
            "tpot_p99": percentile(tpots, 99),
            "e2e_mean": float(np.mean(e2es)),
            "e2e_p50": percentile(e2es, 50),
            "e2e_p95": percentile(e2es, 95),
            "e2e_p99": percentile(e2es, 99),
        }}

        print("BENCHMARK_RESULTS_START")
        print(json.dumps(metrics))
        print("BENCHMARK_RESULTS_END")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(run_benchmark())
'''

    # Write and run benchmark script
    logger.info("Running benchmark...")

    def _run() -> tuple[str, str, int]:
        # Write script
        _exec_sync(
            sandbox,
            f"cat > /tmp/benchmark.py << 'SCRIPT_EOF'\n{benchmark_script}\nSCRIPT_EOF",
            timeout=30,
        )

        # Run benchmark
        return _exec_sync(
            sandbox,
            f"cd {workspace} && PYTHONPATH={workspace} python /tmp/benchmark.py",
            timeout=3600,  # 1 hour timeout
        )

    stdout, stderr, exit_code = await trio.to_thread.run_sync(_run)

    if exit_code != 0:
        raise RuntimeError(f"Benchmark failed:\nstdout:\n{stdout}\n\nstderr:\n{stderr}")

    # Parse results
    if "BENCHMARK_RESULTS_START" not in stdout:
        raise RuntimeError(f"No benchmark results found in output: {stdout[-1000:]}")

    results_json = (
        stdout.split("BENCHMARK_RESULTS_START")[1].split("BENCHMARK_RESULTS_END")[0].strip()
    )
    metrics = json.loads(results_json)

    # Get GPU info
    def _get_gpu_info() -> str:
        out, _, _ = _exec_sync(
            sandbox, "nvidia-smi --query-gpu=name --format=csv,noheader", timeout=30
        )
        return out.strip().split("\n")[0]

    gpu_name = await trio.to_thread.run_sync(_get_gpu_info)

    return BenchmarkResult(
        backend=config.backend,
        model=config.model,
        gpu=gpu_name,
        workload=asdict(config.workload),
        requests_per_second=metrics["requests_per_second"],
        tokens_per_second=metrics["tokens_per_second"],
        output_tokens_per_second=metrics["output_tokens_per_second"],
        ttft_mean=metrics["ttft_mean"],
        ttft_p50=metrics["ttft_p50"],
        ttft_p95=metrics["ttft_p95"],
        ttft_p99=metrics["ttft_p99"],
        tpot_mean=metrics["tpot_mean"],
        tpot_p50=metrics["tpot_p50"],
        tpot_p95=metrics["tpot_p95"],
        tpot_p99=metrics["tpot_p99"],
        e2e_mean=metrics["e2e_mean"],
        e2e_p50=metrics["e2e_p50"],
        e2e_p95=metrics["e2e_p95"],
        e2e_p99=metrics["e2e_p99"],
        gpu_memory_peak_mb=0.0,  # TODO: collect from nvidia-smi
        gpu_utilization_mean=0.0,  # TODO: collect from nvidia-smi
    )


async def run_benchmark_local(
    config: BenchmarkConfig,
    gpu_type: str = "A100",
    gpu_count: int = 1,
) -> BenchmarkResult:
    """Run benchmark locally (when already on GPU machine).

    This is used when running on RunPod/SSH where we're already on the GPU.
    """
    import asyncio

    # Get actual GPU name from nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        gpu_name = result.stdout.strip().split("\n")[0]
    except Exception:
        gpu_name = gpu_type

    # Use the run_local_benchmark function
    return await trio.to_thread.run_sync(
        lambda: asyncio.run(_run_benchmark_impl(config, REPO_ROOT, gpu_name))
    )


async def _run_benchmark_impl(
    config: BenchmarkConfig, workspace: Path, gpu_name: str
) -> BenchmarkResult:
    """Run benchmark implementation (can be called from local or sandbox)."""
    import asyncio

    import httpx
    import numpy as np

    from ..engine_v2 import EngineConfig, InferenceEngineV2
    from ..server import create_app

    # Start the server
    engine_config = EngineConfig(
        model_path=config.model,
        max_batch_size=config.max_batch_size,
    )
    engine = InferenceEngineV2(engine_config)
    app = create_app(engine)

    import uvicorn

    server_config = uvicorn.Config(app, host="127.0.0.1", port=8080, log_level="warning")
    server = uvicorn.Server(server_config)

    # Start server in background
    server_task = asyncio.create_task(server.serve())

    # Wait for server to be ready
    import time

    start_wait = time.time()
    while time.time() - start_wait < 300:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://127.0.0.1:8080/health", timeout=5)
                if resp.status_code == 200:
                    break
        except Exception:
            pass
        await asyncio.sleep(0.5)

    try:
        # Generate random prompts
        def generate_random_prompt(length: int) -> list[int]:
            import random

            return [random.randint(1000, 30000) for _ in range(length)]

        prompts = [
            generate_random_prompt(config.workload.input_len)
            for _ in range(config.workload.num_prompts)
        ]

        # Run requests
        semaphore = asyncio.Semaphore(config.workload.concurrency)
        results = []

        async def benchmark_request(client: httpx.AsyncClient, prompt: list[int], idx: int) -> dict:
            async with semaphore:
                start = time.perf_counter()
                first_token_time = None
                tokens_received = 0

                request_body = {
                    "input_ids": prompt,
                    "sampling_params": {
                        "max_new_tokens": config.workload.output_len,
                        "temperature": 0.0,
                    },
                }

                async with client.stream(
                    "POST",
                    "http://127.0.0.1:8080/generate",
                    json=request_body,
                    timeout=300,
                ) as response:
                    async for chunk in response.aiter_bytes():
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        tokens_received += 1

                end = time.perf_counter()
                return {
                    "request_id": idx,
                    "ttft_ms": (first_token_time - start) * 1000 if first_token_time else None,
                    "e2e_ms": (end - start) * 1000,
                    "tokens": config.workload.output_len,  # Use expected output len
                }

        start_time = time.perf_counter()

        async with httpx.AsyncClient() as client:
            tasks = [benchmark_request(client, p, i) for i, p in enumerate(prompts)]
            results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

        # Calculate metrics
        ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
        e2es = [r["e2e_ms"] for r in results]
        total_tokens = sum(r["tokens"] for r in results)

        # TPOT = (e2e - ttft) / (tokens - 1)
        tpots = []
        for r in results:
            if r["ttft_ms"] and r["tokens"] > 1:
                tpots.append((r["e2e_ms"] - r["ttft_ms"]) / (r["tokens"] - 1))

        def percentile(data: list, p: int) -> float:
            return float(np.percentile(data, p)) if data else 0.0

        return BenchmarkResult(
            backend=config.backend,
            model=config.model,
            gpu=gpu_name,
            workload=asdict(config.workload),
            requests_per_second=config.workload.num_prompts / total_time,
            tokens_per_second=total_tokens / total_time,
            output_tokens_per_second=total_tokens / total_time,
            ttft_mean=float(np.mean(ttfts)) if ttfts else 0.0,
            ttft_p50=percentile(ttfts, 50),
            ttft_p95=percentile(ttfts, 95),
            ttft_p99=percentile(ttfts, 99),
            tpot_mean=float(np.mean(tpots)) if tpots else 0.0,
            tpot_p50=percentile(tpots, 50),
            tpot_p95=percentile(tpots, 95),
            tpot_p99=percentile(tpots, 99),
            e2e_mean=float(np.mean(e2es)),
            e2e_p50=percentile(e2es, 50),
            e2e_p95=percentile(e2es, 95),
            e2e_p99=percentile(e2es, 99),
            gpu_memory_peak_mb=0.0,
            gpu_utilization_mean=0.0,
        )
    finally:
        server.should_exit = True
        await server_task


async def run_benchmark(
    config: BenchmarkConfig,
    deps: DepsConfig,
    gpu_type: str = "A100",
    gpu_count: int = 1,
    timeout_hours: int = 2,
) -> BenchmarkResult:
    """Run benchmark on Modal.

    Args:
        config: Benchmark configuration
        deps: Environment dependencies (from HardwareConfig.deps)
        gpu_type: GPU type to use
        gpu_count: Number of GPUs
        timeout_hours: Sandbox timeout

    Returns:
        BenchmarkResult with metrics
    """
    import modal
    import trio_asyncio

    modal.enable_output()  # Show build logs

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"bench_{timestamp}"

    logger.info("=" * 60)
    logger.info(f"Benchmark: {run_name}")
    logger.info("=" * 60)
    logger.info(f"Backend: {config.backend}")
    logger.info(f"Model: {config.model}")
    logger.info(f"GPU: {gpu_count}x {gpu_type}")
    logger.info(f"Workload: {config.workload.type} ({config.workload.num_prompts} prompts)")

    async with trio_asyncio.open_loop():
        # Create sandbox
        logger.info("Creating Modal sandbox...")

        app = await trio_asyncio.aio_as_trio(
            modal.App.lookup.aio(MODAL_APP_NAME, create_if_missing=True)
        )

        image = _build_modal_image(modal, deps, gpu_type)

        gpu_spec = f"{gpu_type}:{gpu_count}" if gpu_count > 1 else gpu_type
        ts = int(datetime.now(timezone.utc).timestamp())
        sandbox_name = f"bench-{config.backend}-{ts}"

        sandbox = await trio_asyncio.aio_as_trio(
            modal.Sandbox.create.aio(
                app=app,
                image=image,
                gpu=gpu_spec,
                timeout=timeout_hours * 3600,
                name=sandbox_name,
            )
        )

        logger.info(f"Sandbox created: {sandbox.object_id}")

        try:
            # Verify GPU
            def _check_gpu() -> None:
                _, _, code = _exec_sync(sandbox, "nvidia-smi", timeout=30)
                assert code == 0, "nvidia-smi failed"

            await trio.to_thread.run_sync(_check_gpu)

            # Sync code
            logger.info("Syncing code...")
            workspace = await _sync_code_to_sandbox(sandbox, REPO_ROOT)

            # Run benchmark
            logger.info("Starting benchmark...")
            result = await _run_benchmark_in_sandbox(sandbox, workspace, config, run_name)

            logger.info("Benchmark complete!")
            logger.info(f"  Throughput: {result.requests_per_second:.1f} req/s")
            logger.info(f"  TTFT p50: {result.ttft_p50:.1f} ms")
            logger.info(f"  E2E p50: {result.e2e_p50:.1f} ms")

            return result

        finally:
            logger.info("Terminating sandbox...")

            def _terminate() -> None:
                sandbox.terminate()

            await trio.to_thread.run_sync(_terminate)
