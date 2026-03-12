"""Benchmark engine_v2 on A100.

Run with:
    python -m argus run --config examples/benchmark/engine_v2_a100.py
"""

from rollouts.inference.benchmark.config import BenchmarkConfig, WorkloadConfig
from rollouts.training.configs import DepsConfig, HardwareConfig

# =============================================================================
# Hardware Configuration
# =============================================================================

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=1,
    provider="modal",
    deps=DepsConfig(
        pip_packages=(
            "torch>=2.6",
            "transformers>=5.0",
            "numpy",
            "httpx",
            "safetensors",
            "triton",
            "flashinfer-python>=0.3",  # 0.3+ has backend= param for decode wrapper
            "fastapi",
            "uvicorn",
        ),
        pip_index_url="https://download.pytorch.org/whl/cu126",
        pip_extra_index_url="https://pypi.org/simple",
    ),
)

# =============================================================================
# Benchmark Configuration
# =============================================================================

config = BenchmarkConfig(
    model="Qwen/Qwen3-0.6B",
    backend="engine_v2",
    workload=WorkloadConfig(
        type="random",
        num_prompts=100,  # Start small for testing
        input_len=256,
        output_len=64,
        concurrency=32,
    ),
    max_batch_size=128,
    mem_fraction=0.9,
    output_dir="results/benchmark",
    experiment_name="engine_v2_a100",
)
