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
from .lm_eval import run_lm_eval
from .native import (
    EvalReport,
    EvalRuntime,
    evaluate,
    evaluate_sample,
    group_by,
    simple_evaluate,
    summarize,
)

__all__ = [
    "EndpointConfig",
    "EvalReport",
    "EvalOutputConfig",
    "EvalRuntime",
    "EvalRunConfig",
    "HardwareConfig",
    "InferenceServerConfig",
    "evaluate",
    "evaluate_sample",
    "group_by",
    "run_lm_eval",
    "simple_evaluate",
    "summarize",
]
