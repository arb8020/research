"""Config module contract validation.

These checks live at the runner boundary so malformed config modules fail where
they are actually consumed, rather than relying on unit tests to encode the
contract indirectly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_status import ConfigStatus


def _is_prime_ci_config(config_path: Path) -> bool:
    return "configs/prime_ci" in config_path.as_posix()


def _require_prime_ci_status(config_module: Any, config_path: Path) -> None:
    if not _is_prime_ci_config(config_path):
        return
    status = getattr(config_module, "config_status", None)
    if not isinstance(status, ConfigStatus):
        raise ValueError(f"Prime-CI config {config_path} must export config_status: ConfigStatus")


def validate_train_config_module(config_module: Any, config_path: Path) -> None:
    """Validate the module contract for train/run entrypoints."""
    _require_prime_ci_status(config_module, config_path)
    # TODO(boundary): this module export contract (`config` here, callable
    # `train(...)` at the runner) wants to collapse into one explicit training
    # workload product type instead of split module-level conventions.
    if not hasattr(config_module, "config"):
        raise ValueError(f"Training config {config_path} must export 'config'")


def validate_eval_config_module(config_module: Any, config_path: Path) -> None:
    """Validate the module contract for eval entrypoints."""
    from .eval.configs import AgentRunSpec, EvalTaskSpec

    _require_prime_ci_status(config_module, config_path)
    # TODO(boundary): these eval exports (`tasks`, `run_spec` /
    # `prepare_messages`, scorer, run/output) want to be one explicit eval task
    # product type instead of a bag of top-level module attributes.

    eval_task = getattr(config_module, "eval_task", None)
    if eval_task is not None:
        if not isinstance(eval_task, EvalTaskSpec):
            raise ValueError(f"Eval config {config_path} must export eval_task: EvalTaskSpec")
        return

    has_tasks = hasattr(config_module, "tasks")
    has_tasks_path = hasattr(config_module, "tasks_path")
    if not has_tasks and not has_tasks_path:
        raise ValueError(f"Eval config {config_path} must define 'tasks' or 'tasks_path'")

    run_spec = getattr(config_module, "run_spec", None)
    if run_spec is not None and not isinstance(run_spec, AgentRunSpec):
        raise ValueError(f"Eval config {config_path} must export run_spec: AgentRunSpec")

    prepare_messages = getattr(config_module, "prepare_messages", None)
    attempt_executor = getattr(config_module, "attempt_executor", None)
    if run_spec is None and not callable(prepare_messages) and not callable(attempt_executor):
        raise ValueError(
            f"Eval config {config_path} must export callable prepare_messages or attempt_executor or run_spec"
        )

    scorer = getattr(config_module, "scorer", None)
    has_scoring_path = scorer is not None

    if not has_scoring_path:
        raise ValueError(f"Eval config {config_path} must define an explicit scorer")
