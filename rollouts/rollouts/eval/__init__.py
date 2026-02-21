"""Unified evaluation framework.

Run evals against API endpoints or self-hosted SGLang/vLLM servers.

Usage:
    python -m rollouts.eval --config examples/eval/reverse_text/smoke.py
    python -m rollouts.eval --config ... --provider modal  # Provision GPU
    python -m rollouts.eval --config ... --local  # Use local GPU
"""

from .configs import (
    EndpointConfig,
    EvalOutputConfig,
    EvalRunConfig,
    HardwareConfig,
    InferenceServerConfig,
)

__all__ = [
    "EndpointConfig",
    "EvalOutputConfig",
    "EvalRunConfig",
    "HardwareConfig",
    "InferenceServerConfig",
]
