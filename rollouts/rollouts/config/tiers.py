"""Shared config tiers for evals and training.

Three independent frozen dataclasses that separate concerns cleanly:
- EndpointConfig: LLM endpoint (model, provider, temperature)
- RunConfig: Execution params (max_turns, max_concurrent, limit)
- OutputConfig: Output dirs and experiment naming

These are the building blocks. EvalSpec + run_eval_from_spec() compose them
into a full eval run. Training configs (GRPOConfig) compose them similarly.

Design: Ported from wafer/research/evals/shared/config.py.
Style: Pythonic + hierarchical + serializable (see docs/code_style/experiment_config.md).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EndpointConfig:
    """LLM endpoint configuration.

    Just the model params. No API key handling, no URL construction.
    The runner converts this to a rollouts.dtypes.Endpoint.

    Example:
        endpoint = EndpointConfig(model="claude-sonnet-4-20250514")
        endpoint = EndpointConfig(model="gpt-4o", provider="openai", temperature=0.5)
    """

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096
    # OpenAI reasoning models (o1/o3) require max_completion_tokens
    max_completion_tokens: int | None = None
    # OpenAI reasoning effort: "none", "minimal", "low", "medium", "high", "xhigh"
    reasoning_effort: str | None = None
    # Anthropic extended thinking: {"type": "enabled", "budget_tokens": <int>}
    thinking: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunConfig:
    """Eval/training run configuration.

    Just execution params. Independent of what model or output you're using.

    Example:
        run = RunConfig(max_turns=10, max_concurrent=5)
        run = RunConfig(max_concurrent=1, verbose=False)  # sequential, quiet
    """

    max_turns: int = 50
    max_concurrent: int = 5
    max_api_concurrent: int | None = None
    max_tool_concurrent: int | None = None
    limit: int | None = None  # Max samples to evaluate
    verbose: bool = True
    show_progress: bool = True


@dataclass(frozen=True)
class OutputConfig:
    """Output configuration.

    Just where results go. Independent of model or execution params.

    Example:
        output = OutputConfig(experiment_name="insecure_code_smoke")
        output = OutputConfig(output_dir="/tmp/quick_test")
    """

    experiment_name: str = "eval"
    output_dir: str | None = None  # None = auto-generate from experiment_name + timestamp


# ── Serialization ─────────────────────────────────────────────────────────────


def save_config(config: Any, path: Path | str) -> None:
    """Save a frozen dataclass config as JSON for reproducibility.

    Args:
        config: Any frozen dataclass
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2, default=str))
