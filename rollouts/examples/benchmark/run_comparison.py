"""Run engine_v2 benchmark on Modal (single backend test).

Usage:
    cd /Users/chiraagbalu/research/rollouts
    python examples/benchmark/run_comparison.py

    # Or with sglang:
    python examples/benchmark/run_comparison.py --backend sglang
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import trio

# Add rollouts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rollouts.inference.benchmark.config import BenchmarkConfig, WorkloadConfig
from rollouts.inference.benchmark.runner import run_benchmark
from rollouts.training.configs import DepsConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Small workload for testing
WORKLOAD = WorkloadConfig(
    type="random",
    num_prompts=20,  # Small for testing
    input_len=128,
    output_len=32,
    concurrency=8,
)

# Dependencies for engine_v2
ENGINE_V2_DEPS = DepsConfig(
    pip_packages=(
        "torch==2.6.0",
        "transformers>=5.0",
        "numpy",
        "httpx",
        "safetensors",
        "triton",
        "flashinfer-python>=0.3",
        "fastapi",
        "uvicorn",
    ),
    pip_index_url="https://download.pytorch.org/whl/cu126",
)

# Dependencies for sglang
SGLANG_DEPS = DepsConfig(
    pip_packages=(
        "torch==2.6.0",
        "transformers>=5.0",
        "numpy",
        "httpx",
        "sglang[srt]>=0.4",  # srt instead of all - faster install
    ),
    pip_index_url="https://download.pytorch.org/whl/cu126",
)


async def main(backend: str) -> None:
    model = "Qwen/Qwen3-0.6B"
    gpu_type = "A100"

    deps = ENGINE_V2_DEPS if backend == "engine_v2" else SGLANG_DEPS

    config = BenchmarkConfig(
        model=model,
        backend=backend,
        workload=WORKLOAD,
        max_batch_size=64,
    )

    logger.info("=" * 60)
    logger.info(f"Running {backend} benchmark")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(
        f"Workload: {WORKLOAD.num_prompts} prompts, {WORKLOAD.input_len} in, {WORKLOAD.output_len} out"
    )

    try:
        result = await run_benchmark(
            config=config,
            deps=deps,
            gpu_type=gpu_type,
        )

        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Backend: {backend}")
        print(f"Model: {model}")
        print(f"GPU: {gpu_type}")
        print()
        print(
            f"Throughput: {result.requests_per_second:.1f} req/s, {result.tokens_per_second:.0f} tok/s"
        )
        print(
            f"TTFT: mean={result.ttft_mean:.1f}ms, p50={result.ttft_p50:.1f}ms, p95={result.ttft_p95:.1f}ms"
        )
        print(
            f"E2E:  mean={result.e2e_mean:.1f}ms, p50={result.e2e_p50:.1f}ms, p95={result.e2e_p95:.1f}ms"
        )

        # Save results
        output_path = Path(f"results/benchmark/{backend}_result.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {output_path}")

    except Exception as e:
        logger.exception(f"{backend} benchmark failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["engine_v2", "sglang"], default="engine_v2")
    args = parser.parse_args()
    trio.run(main, args.backend)
