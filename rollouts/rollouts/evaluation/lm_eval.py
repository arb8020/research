"""Wrapper for lm-evaluation-harness.

Provides a simple interface to run standard benchmarks (MMLU, HellaSwag, etc.)
against a local or remote model server.

Usage:
    from rollouts.evaluation.lm_eval import run_lm_eval

    # Against a running server
    results = run_lm_eval(
        base_url="http://localhost:30000",
        tokenizer="Qwen/Qwen3-30B-A3B",
        tasks=["mmlu", "hellaswag"],
    )

    # Start SGLang server and run evals
    results = run_lm_eval(
        model_path="/path/to/model",
        tasks=["mmlu", "hellaswag"],
        backend="sglang",
    )
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_lm_eval(
    *,
    # Model source (one of these required)
    base_url: str | None = None,
    model_path: str | Path | None = None,
    # Config
    tokenizer: str | None = None,
    tasks: list[str] | None = None,
    batch_size: int = 8,
    # Server config (if model_path provided)
    backend: str = "sglang",
    port: int = 30000,
    server_timeout: int = 180,
) -> dict[str, Any]:
    """Run lm-eval benchmarks.

    Args:
        base_url: URL of running inference server (e.g. "http://localhost:30000")
        model_path: Path to model to serve (will start server automatically)
        tokenizer: HuggingFace tokenizer name (required for API models)
        tasks: List of lm-eval task names (default: ["hellaswag"])
        batch_size: Batch size for evaluation
        backend: Server backend if model_path provided ("sglang" or "vllm")
        port: Port for server if model_path provided
        server_timeout: Seconds to wait for server startup

    Returns:
        Dict mapping task names to metric dicts
    """
    import lm_eval

    if tasks is None:
        tasks = ["hellaswag"]

    server_proc = None
    try:
        # Start server if model_path provided
        if model_path is not None:
            if base_url is not None:
                raise ValueError("Cannot specify both base_url and model_path")

            base_url = f"http://localhost:{port}"
            server_proc = _start_server(
                model_path=str(model_path),
                backend=backend,
                port=port,
                timeout=server_timeout,
            )

            # Use model_path as tokenizer if not specified
            if tokenizer is None:
                # Try to infer from model config
                config_path = Path(model_path) / "config.json"
                if config_path.exists():
                    import json

                    config = json.loads(config_path.read_text())
                    tokenizer = config.get("_name_or_path", str(model_path))
                else:
                    tokenizer = str(model_path)

        if base_url is None:
            raise ValueError("Must specify either base_url or model_path")

        if tokenizer is None:
            raise ValueError("tokenizer required for API models")

        logger.info(f"Running lm-eval on tasks: {tasks}")
        logger.info(f"Server: {base_url}")
        logger.info(f"Tokenizer: {tokenizer}")

        results = lm_eval.simple_evaluate(
            model="local-completions",
            model_args={
                "pretrained": tokenizer,
                "base_url": f"{base_url}/v1/completions",
                "tokenizer": tokenizer,
                "tokenized_requests": False,
            },
            tasks=tasks,
            batch_size=batch_size,
        )

        # Extract metrics
        summary: dict[str, Any] = {}
        if "results" in results:
            for task, metrics in results["results"].items():
                summary[task] = {
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and not k.startswith("_")
                }

        logger.info("Evaluation complete")
        for task, metrics in summary.items():
            logger.info(f"  {task}:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    logger.info(f"    {k}: {v:.4f}")
                else:
                    logger.info(f"    {k}: {v}")

        return summary

    finally:
        if server_proc is not None:
            logger.info("Stopping server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


def _start_server(
    model_path: str,
    backend: str,
    port: int,
    timeout: int,
) -> subprocess.Popen:
    """Start inference server and wait for it to be ready."""
    import requests

    if backend == "sglang":
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--port",
            str(port),
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.85",
        ]
    elif backend == "vllm":
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--port",
            str(port),
            "--trust-remote-code",
        ]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    logger.info(f"Starting {backend} server on port {port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    base_url = f"http://localhost:{port}"
    logger.info(f"Waiting for server at {base_url}...")

    for attempt in range(timeout):
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                logger.info(f"Server ready after {attempt}s")
                return proc
        except requests.RequestException:
            pass

        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(f"Server died during startup. Output:\n{stdout[-2000:]}")

        time.sleep(1)

    proc.terminate()
    raise TimeoutError(f"Server failed to start within {timeout}s")
