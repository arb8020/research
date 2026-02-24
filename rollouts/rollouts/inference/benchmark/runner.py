"""Benchmark runner for Modal.

Runs inference benchmarks on Modal sandboxes.
Images are pre-built in images.py with pinned versions.
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
from typing import Any

import trio

from .config import BenchmarkConfig, BenchmarkResult

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.parent
MODAL_APP_NAME = "rollouts-benchmark"


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
    *,
    start_server: bool = True,
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

    # Generate benchmark script with JSONL wide event logging
    benchmark_script = f'''
import asyncio
import json
import sys
import time
import random
import string
import numpy as np
import httpx
from datetime import datetime, timezone

# Config
MODEL = "{config.model}"
BACKEND = "{config.backend}"
NUM_PROMPTS = {config.workload.num_prompts}
INPUT_LEN = {config.workload.input_len}
OUTPUT_LEN = {config.workload.output_len}
CONCURRENCY = {config.workload.concurrency}
MAX_BATCH_SIZE = {config.max_batch_size}
MEM_FRACTION = {config.mem_fraction}
SERVER_URL = "http://localhost:30000"
START_SERVER = {start_server}

# Wide event logging - each event is a complete JSON line
def log_event(event: str, **data):
    """Emit a JSONL wide event to stdout."""
    record = {{
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **data,
    }}
    print(json.dumps(record), flush=True)

def generate_random_prompt(length: int) -> str:
    """Generate random text of approximately the given token length."""
    chars = length * 4
    return "".join(random.choices(string.ascii_letters + " ", k=chars))

async def start_server():
    """Start the inference server."""
    import subprocess

    if BACKEND == "engine_v2":
        cmd = [
            sys.executable, "-m", "rollouts.inference.server",
            "--model", MODEL,
            "--port", "30000",
            "--max-batch-size", str(MAX_BATCH_SIZE),
            "--max-seq-len", "2048",
        ]
    elif BACKEND == "sglang":
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", MODEL,
            "--port", "30000",
            "--mem-fraction-static", str(MEM_FRACTION),
            "--attention-backend", "triton",
            "--disable-cuda-graph",
        ]
    else:
        raise ValueError(f"Unknown backend: {{BACKEND}}")

    log_event("server_start", backend=BACKEND, model=MODEL, cmd=" ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout for easier debugging
        env={{**dict(__import__("os").environ), "PYTHONUNBUFFERED": "1"}},
    )

    # Stream server output while waiting for ready
    import selectors
    import os
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)

    server_logs = []
    start_wait = time.time()
    timeout_sec = 180  # 3 min timeout

    while time.time() - start_wait < timeout_sec:
        # Check for output (non-blocking)
        ready = sel.select(timeout=0.1)
        for key, _ in ready:
            line = key.fileobj.readline()
            if line:
                line_str = line.decode("utf-8", errors="replace").rstrip()
                server_logs.append(line_str)
                log_event("server_log", line=line_str)

        # Check if process died
        poll = proc.poll()
        if poll is not None:
            # Drain remaining output
            remaining = proc.stdout.read().decode("utf-8", errors="replace")
            for line in remaining.splitlines():
                server_logs.append(line)
                log_event("server_log", line=line)
            log_event("server_died", exit_code=poll, logs_tail=server_logs[-20:])
            raise RuntimeError(f"Server died with exit code {{poll}}")

        # Check if server is ready
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{{SERVER_URL}}/health", timeout=1.0)
                if resp.status_code == 200:
                    elapsed = time.time() - start_wait
                    log_event("server_ready", elapsed_sec=round(elapsed, 1))
                    sel.close()
                    return proc
        except Exception:
            pass

    # Timeout
    proc.terminate()
    log_event("server_timeout", logs_tail=server_logs[-30:])
    raise RuntimeError("Server failed to start within timeout")


async def wait_for_existing_server() -> None:
    """Wait for an already running benchmark server."""
    start_wait = time.time()
    timeout_sec = 180

    while time.time() - start_wait < timeout_sec:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{{SERVER_URL}}/health", timeout=1.0)
                if resp.status_code == 200:
                    elapsed = time.time() - start_wait
                    log_event("server_ready", elapsed_sec=round(elapsed, 1))
                    return
        except Exception:
            await asyncio.sleep(0.1)

    raise RuntimeError("Existing server failed to become ready within timeout")

async def benchmark_request(client: httpx.AsyncClient, prompt: str, request_id: int) -> dict:
    """Send a single benchmark request and measure timing."""
    start = time.perf_counter()
    first_token_time = None
    tokens_received = 0
    error = None

    try:
        async with client.stream(
            "POST",
            f"{{SERVER_URL}}/v1/chat/completions",
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
    except Exception as e:
        error = f"{{type(e).__name__}}: {{e}}"

    end = time.perf_counter()

    return {{
        "request_id": request_id,
        "ttft_ms": (first_token_time - start) * 1000 if first_token_time else None,
        "e2e_ms": (end - start) * 1000,
        "tokens": tokens_received,
        "error": error,
    }}

async def run_benchmark():
    """Run the benchmark."""
    log_event("benchmark_init",
              backend=BACKEND,
              model=MODEL,
              num_prompts=NUM_PROMPTS,
              input_len=INPUT_LEN,
              output_len=OUTPUT_LEN,
              concurrency=CONCURRENCY)

    proc = None
    if START_SERVER:
        proc = await start_server()
    else:
        log_event("server_reuse", url=SERVER_URL)
        await wait_for_existing_server()

    try:
        # Generate prompts
        prompts = [generate_random_prompt(INPUT_LEN) for _ in range(NUM_PROMPTS)]
        log_event("prompts_generated", count=len(prompts))

        # Run requests with concurrency limit
        semaphore = asyncio.Semaphore(CONCURRENCY)
        results = []
        completed = 0
        errors = 0

        async def bounded_request(client, prompt, idx):
            nonlocal completed, errors
            async with semaphore:
                result = await benchmark_request(client, prompt, idx)
                completed += 1
                if result["error"]:
                    errors += 1
                # Log progress every 10% or on errors
                if completed % max(1, NUM_PROMPTS // 10) == 0 or result["error"]:
                    log_event("progress",
                              completed=completed,
                              total=NUM_PROMPTS,
                              errors=errors,
                              pct=round(100 * completed / NUM_PROMPTS, 1),
                              error=result.get("error"))
                return result

        log_event("benchmark_start", num_prompts=NUM_PROMPTS, concurrency=CONCURRENCY)
        start_time = time.perf_counter()

        async with httpx.AsyncClient() as client:
            tasks = [bounded_request(client, p, i) for i, p in enumerate(prompts)]
            results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time
        log_event("requests_done",
                  total_time_sec=round(total_time, 2),
                  completed=completed,
                  errors=errors)

        # Calculate metrics
        successful = [r for r in results if not r["error"]]
        ttfts = [r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]
        e2es = [r["e2e_ms"] for r in successful]
        total_tokens = sum(r["tokens"] for r in successful)

        tpots = []
        for r in successful:
            if r["ttft_ms"] and r["tokens"] > 1:
                tpots.append((r["e2e_ms"] - r["ttft_ms"]) / (r["tokens"] - 1))

        def percentile(data, p):
            return float(np.percentile(data, p)) if data else 0.0

        metrics = {{
            "requests_per_second": len(successful) / total_time if total_time > 0 else 0,
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "output_tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "ttft_mean": float(np.mean(ttfts)) if ttfts else 0.0,
            "ttft_p50": percentile(ttfts, 50),
            "ttft_p95": percentile(ttfts, 95),
            "ttft_p99": percentile(ttfts, 99),
            "tpot_mean": float(np.mean(tpots)) if tpots else 0.0,
            "tpot_p50": percentile(tpots, 50),
            "tpot_p95": percentile(tpots, 95),
            "tpot_p99": percentile(tpots, 99),
            "e2e_mean": float(np.mean(e2es)) if e2es else 0.0,
            "e2e_p50": percentile(e2es, 50),
            "e2e_p95": percentile(e2es, 95),
            "e2e_p99": percentile(e2es, 99),
            "total_requests": NUM_PROMPTS,
            "successful_requests": len(successful),
            "failed_requests": errors,
        }}

        log_event("benchmark_done", **metrics)

        # Also print the old format for backwards compat with result parsing
        print("BENCHMARK_RESULTS_START", flush=True)
        print(json.dumps(metrics), flush=True)
        print("BENCHMARK_RESULTS_END", flush=True)

    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()
            log_event("server_stopped")

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

        # Run benchmark with unbuffered output for streaming
        return _exec_sync(
            sandbox,
            f"cd {workspace} && PYTHONUNBUFFERED=1 PYTHONPATH={workspace} python /tmp/benchmark.py",
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
    gpu_count: int = 1,  # noqa: ARG001
) -> BenchmarkResult:
    """Run benchmark locally (when already on GPU machine).

    This is used when running on RunPod/SSH where we're already on the GPU.
    Uses subprocess to start server, similar to Modal version.
    """
    import random
    import time

    import httpx
    import numpy as np

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

    # Start server as subprocess
    workspace = REPO_ROOT
    server_cmd = [
        "python",
        "-m",
        "rollouts.inference.server",
        "--model",
        config.model,
        "--port",
        "8080",
        "--max-batch-size",
        str(config.max_batch_size),
    ]

    logger.info(f"Starting server: {' '.join(server_cmd)}")
    proc = subprocess.Popen(
        server_cmd,
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": str(workspace)},
    )

    # Wait for server to be ready
    logger.info("Waiting for server to be ready...")
    start_wait = time.time()
    while time.time() - start_wait < 300:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://127.0.0.1:8080/health", timeout=5)
                if resp.status_code == 200:
                    logger.info("Server is ready")
                    break
        except Exception:
            pass
        await trio.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError("Server failed to start within 5 minutes")

    try:
        # Generate random prompts
        def generate_random_prompt(length: int) -> list[int]:
            return [random.randint(1000, 30000) for _ in range(length)]

        prompts = [
            generate_random_prompt(config.workload.input_len)
            for _ in range(config.workload.num_prompts)
        ]

        logger.info(
            f"Running {len(prompts)} requests with concurrency {config.workload.concurrency}"
        )

        # Run requests using trio
        results: list[dict] = []
        semaphore = trio.Semaphore(config.workload.concurrency)

        async def benchmark_request(prompt: list[int], idx: int) -> dict:
            async with semaphore:
                start = time.perf_counter()

                request_body = {
                    "input_ids": prompt,
                    "sampling_params": {
                        "max_new_tokens": config.workload.output_len,
                        "temperature": 0.0,
                    },
                }

                # Make request
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "http://127.0.0.1:8080/generate",
                        json=request_body,
                        timeout=300,
                    )
                    resp.raise_for_status()

                end = time.perf_counter()
                return {
                    "request_id": idx,
                    "ttft_ms": None,  # Non-streaming doesn't give TTFT
                    "e2e_ms": (end - start) * 1000,
                    "tokens": config.workload.output_len,
                }

        start_time = time.perf_counter()

        # Run all requests concurrently with trio
        async with trio.open_nursery() as nursery:
            for i, prompt in enumerate(prompts):

                async def run_request(p: list[int], idx: int) -> None:
                    result = await benchmark_request(p, idx)
                    results.append(result)

                nursery.start_soon(run_request, prompt, i)

        total_time = time.perf_counter() - start_time

        logger.info(f"Completed {len(results)} requests in {total_time:.2f}s")

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
        proc.terminate()
        proc.wait()


async def run_benchmark(
    config: BenchmarkConfig,
    gpu_type: str = "A100",
    gpu_count: int = 1,
    timeout_hours: int = 2,
    sandbox_id: str | None = None,
    keep_sandbox_alive: bool = False,
    warm_sandbox: bool = False,
) -> BenchmarkResult:
    """Run benchmark on Modal.

    Images are pre-built in images.py with pinned versions.

    Args:
        config: Benchmark configuration
        gpu_type: GPU type to use
        gpu_count: Number of GPUs
        sandbox_id: Reuse existing Modal sandbox by ID
        keep_sandbox_alive: Keep sandbox alive after benchmark completes
        warm_sandbox: Skip starting server in benchmark script (assume server already running)
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
    if warm_sandbox and sandbox_id is None:
        logger.info(
            "Warm benchmark mode is enabled; this assumes a server is already running on port 30000."
        )

    async with trio_asyncio.open_loop():
        # Reuse existing sandbox or create a new one
        if sandbox_id:
            logger.info(f"Reusing sandbox: {sandbox_id}")
            sandbox = await trio.to_thread.run_sync(lambda: modal.Sandbox.from_id(sandbox_id))
        else:
            logger.info("Creating Modal sandbox...")

            app = await trio_asyncio.aio_as_trio(
                modal.App.lookup.aio(MODAL_APP_NAME, create_if_missing=True)
            )

            # Use pre-built images (defined at module level for proper caching)
            from .images import ENGINE_V2_IMAGE, SGLANG_IMAGE

            if config.backend == "sglang":
                image = SGLANG_IMAGE
            elif config.backend == "engine_v2":
                image = ENGINE_V2_IMAGE
            else:
                raise ValueError(f"Unknown backend: {config.backend}")

            gpu_spec = f"{gpu_type}:{gpu_count}" if gpu_count > 1 else gpu_type
            ts = int(datetime.now(timezone.utc).timestamp())
            sandbox_name = f"bench-{config.backend}-{ts}"

            sandbox = await trio.to_thread.run_sync(
                lambda: modal.Sandbox.create(
                    app=app,
                    image=image,
                    gpu=gpu_spec,
                    timeout=timeout_hours * 3600,
                    name=sandbox_name,
                )
            )

        logger.info(f"Sandbox in use: {sandbox.object_id}")
        if keep_sandbox_alive:
            logger.info(f"Keeping sandbox alive: --sandbox-id {sandbox.object_id}")

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
            result = await _run_benchmark_in_sandbox(
                sandbox=sandbox,
                workspace=workspace,
                config=config,
                run_name=run_name,
                start_server=not warm_sandbox,
            )

            logger.info("Benchmark complete!")
            logger.info(f"  Throughput: {result.requests_per_second:.1f} req/s")
            logger.info(f"  TTFT p50: {result.ttft_p50:.1f} ms")
            logger.info(f"  E2E p50: {result.e2e_p50:.1f} ms")

            return result

        finally:
            if keep_sandbox_alive:
                logger.info("Keeping sandbox alive after benchmark.")
            else:
                logger.info("Terminating sandbox...")

                def _terminate() -> None:
                    sandbox.terminate()

                await trio.to_thread.run_sync(_terminate)
