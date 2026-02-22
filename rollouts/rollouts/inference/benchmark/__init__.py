"""Inference benchmarking infrastructure.

Compare inference backends (SGLang, vLLM, engine_v2) on Modal.

Usage:
    python -m rollouts.run --config examples/benchmark/engine_v2_a100.py
"""

from .config import BenchmarkConfig, WorkloadConfig
from .runner import run_benchmark

__all__ = ["BenchmarkConfig", "WorkloadConfig", "run_benchmark"]
