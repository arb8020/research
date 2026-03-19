"""Unified evaluation framework.

Run evals against API endpoints or self-hosted SGLang/vLLM servers.

Usage:
    python -m rollouts.eval --config examples/eval/reverse_text/smoke.py
    python -m rollouts.eval --config ... --provider modal  # Provision GPU
    python -m rollouts.eval --config ... --local  # Use local GPU
"""

from .configs import (
    AgentRunSpec,
    AttemptExecutor,
    CostBudgetStop,
    EndpointConfig,
    EvalOutputConfig,
    EvalRunConfig,
    EvalTaskSpec,
    HardwareConfig,
    InferenceServerConfig,
    MaxTurnsStop,
    TokenBudgetStop,
    WallClockStop,
    resolve_eval_run_spec,
    resolve_eval_task_spec,
)
from .external_attempts import (
    ExternalAttemptArtifact,
    execute_external_attempt,
    trajectory_from_claude_code,
    trajectory_from_codex,
    trajectory_from_openhands,
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
    "AgentRunSpec",
    "AttemptExecutor",
    "CostBudgetStop",
    "ExternalAttemptArtifact",
    "EvalReport",
    "EvalOutputConfig",
    "EvalRuntime",
    "EvalRunConfig",
    "EvalTaskSpec",
    "HardwareConfig",
    "InferenceServerConfig",
    "MaxTurnsStop",
    "TokenBudgetStop",
    "WallClockStop",
    "resolve_eval_run_spec",
    "evaluate",
    "evaluate_sample",
    "execute_external_attempt",
    "group_by",
    "run_lm_eval",
    "simple_evaluate",
    "summarize",
    "trajectory_from_claude_code",
    "trajectory_from_codex",
    "trajectory_from_openhands",
    "resolve_eval_task_spec",
]
