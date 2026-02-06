"""EvalSpec runner — high-level convenience for authoring evals.

This is a layer on top of evaluate(). Eval authors define what's unique
to their eval (EvalSpec), and the runner handles orchestration.

Granularity (Casey Muratori): EvalSpec is bool_button() to EvalConfig's
push_button(). If you need full control, use evaluate() directly.

Design: Ported from wafer/research/evals/shared/runner.py.

Usage:
    # 1. Define what's unique to your eval
    spec = EvalSpec(
        name="my_eval",
        prepare_messages=my_prep,
        score_fn=my_score,
        make_environment=lambda: CodingEnvironment(tools=["read", "write"]),
    )

    # 2. Run with config objects
    result = run_eval_from_spec(spec,
        endpoint=EndpointConfig(model="claude-sonnet-4-20250514"),
        run=RunConfig(max_concurrent=5),
    )

    # 3. Or with kwarg overrides
    result = run_eval_from_spec(spec, model="claude-opus-4", max_concurrent=10)
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dtypes import Environment, Message, Score

logger = logging.getLogger(__name__)


# ──────────────────────── EvalSpec ────────────────────────────────────────────


@dataclass
class EvalSpec:
    """Specification for an eval — the eval-specific parts only.

    This is what eval authors define. Everything else (endpoint construction,
    output dir, concurrency, etc.) is handled by run_eval_from_spec().

    Attributes:
        name: Eval name (used for output dir, logging)
        prepare_messages: Turn a dataset row into initial messages
        score_fn: Score a completed sample
        make_environment: Factory for the tool environment (or None for no tools)
        default_tasks_path: Where tasks live by default (JSON file)
        per_sample_environment: If True, make_environment receives sample_data dict
    """

    name: str
    prepare_messages: Callable[[dict[str, Any]], list[Message]]
    score_fn: Callable[..., Score]

    # Environment factory — can be nullary or take sample_data
    make_environment: Callable[[], Environment] | Callable[[dict[str, Any]], Environment] | None = (
        None
    )

    # Where tasks live by default
    default_tasks_path: Path | None = None

    # Whether make_environment takes sample data (for per-sample isolation)
    per_sample_environment: bool = False


# ──────────────────────── Task Loading ────────────────────────────────────────


def load_tasks(tasks_path: Path | str) -> list[dict[str, Any]]:
    """Load tasks from a JSON file.

    Expects either:
    - A JSON array of dicts: [{"prompt": "..."}, ...]
    - A JSON object with a "tasks" key: {"tasks": [...]}
    """
    path = Path(tasks_path)
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    raise ValueError(f"Expected JSON array or object with 'tasks' key, got {type(data).__name__}")


# ──────────────────────── Main Runner ─────────────────────────────────────────


def run_eval_from_spec(  # noqa: PLR0913
    spec: EvalSpec,
    tasks: list[dict[str, Any]] | None = None,
    tasks_path: Path | str | None = None,
    *,
    # Config objects (from config/tiers.py)
    endpoint: Any | None = None,  # EndpointConfig
    run: Any | None = None,  # RunConfig
    output: Any | None = None,  # OutputConfig
    # Individual overrides (take precedence over config objects)
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_turns: int | None = None,
    max_concurrent: int | None = None,
    max_api_concurrent: int | None = None,
    limit: int | None = None,
    verbose: bool | None = None,
    show_progress: bool | None = None,
    experiment_name: str | None = None,
    output_dir: Path | str | None = None,
    # Resume
    resume_dir: Path | None = None,
) -> dict[str, Any]:
    """Run an eval from its spec.

    This is the shared orchestration logic. Eval-specific behavior comes from
    the EvalSpec (prompts, scoring, environment).

    Args can be passed as config dataclasses or individual overrides.
    Individual args take precedence over config dataclasses.

    Returns:
        Dict with eval results (total, summary_metrics, etc.)
    """
    import trio

    from .config.tiers import EndpointConfig, OutputConfig, RunConfig
    from .dtypes import Endpoint, EvalConfig
    from .dtypes import RunConfig as AgentRunConfig
    from .evaluation import evaluate

    # ── Resolve configs with defaults ──
    endpoint_cfg = endpoint or EndpointConfig()
    run_cfg = run or RunConfig()
    output_cfg = output or OutputConfig(experiment_name=spec.name)

    # Apply individual overrides
    _model = model if model is not None else endpoint_cfg.model
    _provider = provider if provider is not None else endpoint_cfg.provider
    _temperature = temperature if temperature is not None else endpoint_cfg.temperature
    _max_tokens = max_tokens if max_tokens is not None else endpoint_cfg.max_tokens

    _max_turns = max_turns if max_turns is not None else run_cfg.max_turns
    _max_concurrent = max_concurrent if max_concurrent is not None else run_cfg.max_concurrent
    _max_api_concurrent = (
        max_api_concurrent if max_api_concurrent is not None else run_cfg.max_api_concurrent
    )
    _limit = limit if limit is not None else run_cfg.limit
    _verbose = verbose if verbose is not None else run_cfg.verbose
    _show_progress = show_progress if show_progress is not None else run_cfg.show_progress

    _experiment_name = (
        experiment_name if experiment_name is not None else output_cfg.experiment_name
    )
    _output_dir = output_dir if output_dir is not None else output_cfg.output_dir

    # ── Load tasks ──
    if tasks is None:
        effective_tasks_path = tasks_path or spec.default_tasks_path
        if effective_tasks_path is None:
            raise ValueError("Must provide tasks or tasks_path, or set spec.default_tasks_path")
        tasks = load_tasks(effective_tasks_path)

    if _limit is not None:
        tasks = tasks[:_limit]

    logger.info("[%s] Loaded %s tasks", spec.name, len(tasks))

    # ── Build endpoint ──
    # Try standard env vars for the provider
    api_key_env_vars = {
        "anthropic": ["ANTHROPIC_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "google": ["GOOGLE_API_KEY"],
    }
    api_key = ""
    for var_name in api_key_env_vars.get(_provider, []):
        api_key = os.getenv(var_name, "")
        if api_key:
            break

    assert api_key, (
        f"No API key found for provider '{_provider}'. "
        f"Set one of: {api_key_env_vars.get(_provider, ['<unknown provider>'])}"
    )

    eval_endpoint = Endpoint(
        provider=_provider,
        model=_model,
        api_base="",
        api_key=api_key,
        temperature=_temperature,
        max_tokens=_max_tokens,
        max_completion_tokens=getattr(endpoint_cfg, "max_completion_tokens", None),
        reasoning_effort=getattr(endpoint_cfg, "reasoning_effort", None),
        thinking=getattr(endpoint_cfg, "thinking", None),
    )

    # ── Build environment ──
    environment = None
    environment_factory = None
    if spec.make_environment is not None:
        if spec.per_sample_environment:
            environment_factory = spec.make_environment
        else:
            environment = spec.make_environment()  # type: ignore[missing-argument]  # nullary when not per_sample

    # ── Stop handlers ──
    from .handlers import handle_stop_max_turns
    from .dtypes import AgentState, StopReason

    async def stop_on_no_tool(state: AgentState, run_config: AgentRunConfig) -> AgentState:
        return replace(state, stop=StopReason.TASK_COMPLETED)

    async def on_chunk(_: Any) -> None:
        pass

    agent_run_config = AgentRunConfig(
        on_chunk=on_chunk,
        handle_stop=handle_stop_max_turns(_max_turns),
        handle_no_tool=stop_on_no_tool,
    )

    # ── Output directory ──
    if _output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        _output_dir = Path("results") / f"{_experiment_name}_{timestamp}"

    _output_dir = Path(_output_dir)

    # ── Build EvalConfig ──
    eval_config = EvalConfig(
        endpoint=eval_endpoint,
        score_fn=spec.score_fn,
        prepare_messages=spec.prepare_messages,
        environment=environment,
        environment_factory=environment_factory,
        run_config=agent_run_config,
        max_samples=len(tasks),
        max_concurrent=_max_concurrent,
        max_api_concurrent=_max_api_concurrent,
        verbose=_verbose,
        output_dir=_output_dir,
        eval_name=_experiment_name,
        show_progress=_show_progress,
        resume_dir=resume_dir,
    )

    # ── Run ──
    async def _run() -> dict[str, Any]:
        report = await evaluate(iter(tasks), eval_config)
        return {
            "total": report.total_samples,
            **report.summary_metrics,
        }

    return trio.run(_run)
