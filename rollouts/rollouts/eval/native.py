"""Composable evaluation framework with first-class rewards.

Design mirrors run_agent/run_agent_step for easy parallelization.
Tiger Style: Pure functions, explicit configuration, no hidden state.

MIGRATION: The core loop in this file (_evaluate_batch + evaluate_sample) should
be replaced by training/loops/eval_loop.py using AsyncRolloutManager.generate_batch().

Reason: eval and RL are the same pipeline (generate_fn → score → [optionally train]).
_evaluate_batch duplicates the rollout collection logic from AsyncRolloutManager,
but without the shared observability infrastructure. The RL path has no access to
eval's per-sample progress/events; eval has no access to RL's oversampling/filtering.

The migration target (training/loops/eval_loop.py) has the full plan.
Until that migration is complete, this file remains the active eval implementation.
"""

import json
import logging
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from inspect import isawaitable
from pathlib import Path
from typing import Any, cast

import trio

from .._logging import EvalLoggingContext, setup_eval_logging
from ..agents import Actor, AgentState, RunConfig, run_agent
from ..core import Environment, EvalConfig, Metric, Score, Trajectory
from ..dtypes import (
    FirstToken,
    LLMCallEnd,
    StopReason,
    StreamChunk,
    TextDelta,
    TextEnd,
    ThinkingDelta,
    ToolExecutionEnd,
)
from ..export_html import run_to_html, sample_to_html
from ..progress import MultiProgress
from ..training.scoring import attach_score, score_result
from ..training.types import (
    AttemptEvaluation,
    AttemptResult,
    ProblemRow,
    Scorer,
    ScoringContext,
    Status,
)

logger = logging.getLogger(__name__)  # Human/operator-oriented module logs.

# Structured eval event stream. This is separate from the module logger above:
# use `_event_logger` for machine-readable operational facts that belong in
# events.jsonl.
_event_logger = logging.getLogger("rollouts.eval.events")


async def _maybe_start_environment_runtime(environment_or_factory: Any) -> None:
    start = getattr(environment_or_factory, "start", None)
    if callable(start):
        await start()


async def _maybe_stop_environment_runtime(environment_or_factory: Any) -> None:
    stop = getattr(environment_or_factory, "stop", None)
    if callable(stop):
        await stop()


async def _maybe_finalize_environment(
    environment: Any,
    *,
    run_config: RunConfig,
    trajectory: Trajectory | None = None,
) -> None:
    finalize_attempt = getattr(environment, "finalize_attempt", None)
    if not callable(finalize_attempt):
        return
    finalized = finalize_attempt(run_config=run_config, trajectory=trajectory)
    if isawaitable(finalized):
        await finalized


# ── Runtime Context ───────────────────────────────────────────────────────────


@dataclass
class EvalRuntime:
    """Runtime context for evaluation execution.

    Bundles EvalConfig with instantiated handles (limiters, progress).
    Config stays pure/serializable; runtime holds live execution state.

    Created once in evaluate(), passed to all evaluate_sample() calls.
    """

    config: EvalConfig
    api_limiter: trio.CapacityLimiter | None = None
    tool_limiter: trio.CapacityLimiter | None = None
    progress: MultiProgress | None = None
    emit_sample_start: bool = True


# JSON-like recursive type for sanitize_api_keys
# Using string literals for forward references to avoid import cycle
JsonValue = dict[str, "JsonValue"] | list["JsonValue"] | str | int | float | bool | None


# ── Progress Helpers ──────────────────────────────────────────────────────────


def _get_progress_status_for_event(event: object) -> str | None:
    """Extract progress status string from a streaming event.

    Returns status string to display, or None if event doesn't affect status.
    Pure function - no side effects.
    """
    # Handle StreamChunk events (turn lifecycle, modal progress)
    if isinstance(event, StreamChunk):
        if event.type == "turn_start":
            return "waiting..."
        elif event.type == "turn_end":
            return ""  # Clear status
        elif event.type == "modal_progress":
            phase = event.data.get("phase", "")
            return {
                "importing": "importing...",
                "compiling": "compiling...",
                "correctness": "checking...",
                "performance": "benchmarking...",
            }.get(phase, phase)
        return None

    # Handle streaming events from LLM (generic - works with any tool names)
    event_type = getattr(event, "type", "")
    if event_type == "start":
        return "streaming..."
    elif event_type == "text_delta":
        return "streaming..."
    elif event_type == "thinking_start":
        return "thinking..."
    elif event_type == "thinking_delta":
        return "thinking..."
    elif event_type == "toolcall_start":
        tool_name = getattr(event, "name", "tool")
        # Truncate long tool names for display
        short_name = tool_name[:12] + "…" if len(tool_name) > 12 else tool_name
        return f"calling {short_name}..."
    elif event_type == "tool_execution_start":
        tool_name = getattr(event, "tool_name", "tool")
        short_name = tool_name[:12] + "…" if len(tool_name) > 12 else tool_name
        return f"→ {short_name}..."
    elif event_type == "tool_result":
        is_error = getattr(event, "is_error", False)
        if is_error:
            return "tool error"
    return None


def _get_turn_from_event(event: object) -> int | None:
    """Extract turn number from event if applicable."""
    if isinstance(event, StreamChunk):
        if event.type == "turn_start":
            return event.data.get("turn", 0)
        elif event.type == "turn_end":
            return event.data.get("turn", 0) + 1
    return None


def _wrap_event_with_sample_id(event: object, sample_id: str) -> StreamChunk:
    """Wrap event with sample_id for concurrent sample tracking."""
    if isinstance(event, StreamChunk):
        return StreamChunk(
            type=event.type,
            data={**event.data, "sample_id": sample_id},
            timestamp=event.timestamp,
        )
    else:
        return StreamChunk(
            type="event_wrapper",
            data={"sample_id": sample_id, "event": event},
        )


def _extract_text_from_content(content: object) -> str:
    """Extract text from message content (str or list of ContentBlocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if hasattr(block, "text"):
                texts.append(block.text)
            elif hasattr(block, "content"):
                texts.append(str(block.content))
        return "\n".join(texts) if texts else ""
    return str(content) if content else ""


async def _evaluate_batch(
    samples: list[tuple[str, dict[str, Any]]],
    runtime: EvalRuntime,
    on_sample_complete: Callable[[AttemptResult, list[AttemptResult]], None] | None = None,
) -> list[AttemptResult]:
    """Evaluate a batch of samples, handling sequential vs parallel execution.

    This is the core evaluation loop, used for both initial runs and retries.

    Args:
        samples: List of (sample_id, sample_data) tuples
        runtime: Runtime context
        on_sample_complete: Optional callback invoked after each sample completes.
            Receives (completed_sample, all_results_so_far). Useful for incremental
            report writing during long evaluations.
    """
    config = runtime.config
    progress = runtime.progress
    results: list[AttemptResult] = []
    results_lock = trio.Lock()

    async def run_one(sample_id: str, sample_data: dict[str, Any]) -> AttemptResult:
        """Evaluate a single sample."""
        task_name = sample_data.get("name", sample_id)
        if progress:
            progress.add_task(sample_id, name=task_name)

        # Get environment: prefer direct environment, fall back to factory
        if config.environment is not None:
            env = config.environment
        elif config.environment_factory is not None:
            env = await _resolve_environment(config.environment_factory, sample_data)
        else:
            env = None

        result = await evaluate_sample(
            sample_data=sample_data,
            sample_id=sample_id,
            runtime=runtime,
            environment=env,
        )

        # Mark task complete
        if progress:
            reward = result.reward
            status = result.metadata.get("status")
            success = status == "success"
            if success:
                message = f"reward={reward:.2f}"
            elif status == "aborted":
                message = "aborted"
            else:
                error = result.metadata.get("error", "failed")
                message = error[:30] if len(error) > 30 else error
            progress.complete_task(sample_id, success=success, message=message)

        return result

    if config.max_concurrent == 1:
        # Sequential
        for sample_id, sample_data in samples:
            result = await run_one(sample_id, sample_data)
            results.append(result)
            if on_sample_complete:
                on_sample_complete(result, results)
    else:
        # Parallel
        async with trio.open_nursery() as nursery:
            limiter = trio.CapacityLimiter(config.max_concurrent)

            async def run_with_limit(sid: str, sdata: dict[str, Any]) -> None:
                result = await run_one(sid, sdata)
                async with results_lock:
                    results.append(result)
                    if on_sample_complete:
                        on_sample_complete(result, results)

            for sample_id, sample_data in samples:
                nursery.start_soon(run_with_limit, sample_id, sample_data)

    return results


async def _resolve_environment(
    environment_factory: Callable[[dict[str, Any]], Any],
    sample_data: dict[str, Any],
) -> Any:
    """Normalize sync or async environment factories."""
    environment = environment_factory(sample_data)
    if isawaitable(environment):
        return await environment
    return environment


async def _run_attempt_executor(
    attempt_executor: Callable[[dict[str, Any], str, Any | None, RunConfig], Any],
    *,
    sample_data: dict[str, Any],
    sample_id: str,
    environment: Environment | None,
    run_config: RunConfig,
) -> AttemptResult:
    """Run a custom per-sample executor and normalize the result shape."""
    sample = attempt_executor(sample_data, sample_id, environment, run_config)
    if isawaitable(sample):
        sample = await sample
    if isinstance(sample, AttemptResult):
        result = sample
    else:
        raise TypeError(f"attempt_executor must return AttemptResult (got {type(sample).__name__})")
    if result.trajectory is None:
        raise ValueError("attempt_executor must populate attempt trajectory")
    if not result.attempt_id:
        result.attempt_id = sample_id
    return result


async def _serialize_environment_state(environment: Environment | None) -> dict[str, Any] | None:
    if environment is None:
        return None
    try:
        return await environment.serialize()
    except Exception as e:
        logger.warning(f"Failed to serialize environment state: {e}")
        return None


async def _compute_score(
    result: AttemptResult,
    scorer: Scorer,
    scoring_context: ScoringContext | None = None,
) -> Score:
    """Compute score from an explicit scorer stage over a raw result."""

    try:
        return await score_result(scorer, result, scoring_context)
    except Exception as e:
        logger.exception(f"❌ SCORE COMPUTATION FAILED: {e}")
        return Score(metrics=(Metric("error", 0.0, weight=1.0, metadata={"error": str(e)}),))


def _log_sample_completion(
    sample_id: str,
    reward: float,
    exec_metadata: dict[str, Any],
    final_trajectory: Trajectory,
    score: Score | None,
    verbose: bool,
) -> None:
    """Log sample completion with structured logging and rollout record."""
    duration_seconds = exec_metadata.get("duration_seconds", 0.0)

    logger.info(
        f"Sample {sample_id} finished: reward={reward:.3f}, "
        f"turns={exec_metadata['turns_used']}, duration={duration_seconds:.2f}s, "
        f"status={exec_metadata['status']}",
        extra={
            "sample_id": sample_id,
            "reward": reward,
            "turns": exec_metadata["turns_used"],
            "duration_seconds": duration_seconds,
            "status": exec_metadata["status"],
            "stop_reason": exec_metadata.get("stop_reason"),
        },
    )

    # Emit rollout record for TUI trace viewer (matches grpo.py format)
    messages = [
        {"role": m.role, "content": _extract_text_from_content(m.content)}
        for m in final_trajectory.messages
    ]
    logger.info(
        "rollout",
        extra={
            "step": sample_id,
            "prompt": messages[0]["content"] if messages else "",
            "response": messages[-1]["content"] if len(messages) > 1 else "",
            "reward": reward,
            "status": exec_metadata["status"],
            "turns": exec_metadata["turns_used"],
            "stop_reason": exec_metadata.get("stop_reason"),
            "messages": messages,
        },
    )

    if verbose and score:
        metric_str = ", ".join(f"{m.name}={m.value:.3f}" for m in score.metrics[:3])
        logger.info(f"  {metric_str}")


def _map_exec_status(status: str) -> Status:
    """Map eval metadata status strings onto canonical AttemptResult status."""
    if status == "aborted":
        return Status.ABORTED
    return Status.COMPLETED


def _build_base_run_config(
    config: "EvalConfig",
    api_limiter: trio.CapacityLimiter | None,
    tool_limiter: trio.CapacityLimiter | None,
) -> RunConfig:
    """Build the base RunConfig from EvalConfig.

    Handles:
    - Using user-provided run_config or creating default
    - Setting up on_chunk handler (streaming vs silent)
    - Injecting concurrency limiters
    """
    show_turn_progress = config.show_progress and config.max_concurrent == 1

    if config.run_config:
        base_run_config = replace(config.run_config, show_progress=show_turn_progress)
    else:
        # Determine on_chunk handler based on stream_tokens flag
        has_stream_tokens = hasattr(config, "stream_tokens")
        stream_tokens_value = getattr(config, "stream_tokens", None)
        logger.debug(
            f"🔍 Checking stream_tokens: hasattr={has_stream_tokens}, value={stream_tokens_value}"
        )

        if has_stream_tokens and stream_tokens_value:
            from ..agents import stdout_handler

            on_chunk_handler = stdout_handler
            logger.debug("🔍 Using stdout_handler for token streaming")
        else:

            async def silent_chunk_handler(_: object) -> None:
                await trio.lowlevel.checkpoint()

            on_chunk_handler = silent_chunk_handler
            logger.debug("🔍 Using silent mode (no token streaming)")

        base_run_config = RunConfig(on_chunk=on_chunk_handler, show_progress=show_turn_progress)
        logger.debug(
            f"🔍 RunConfig.on_chunk: {on_chunk_handler.__name__ if hasattr(on_chunk_handler, '__name__') else type(on_chunk_handler)}"
        )

    # Inject two-level concurrency limiters if provided
    if api_limiter is not None or tool_limiter is not None:
        base_run_config = replace(
            base_run_config,
            api_limiter=api_limiter,
            tool_limiter=tool_limiter,
        )

    return base_run_config


# EvalSample deleted - use Sample from training.types instead


def get_config_path(file_path: str) -> str | None:
    """Get config file path relative to git repository root.

    Args:
        file_path: Absolute or relative path to config file (usually __file__)

    Returns:
        Path relative to git root, or None if not in a git repo
    """
    import subprocess
    from pathlib import Path

    try:
        # Get git root
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        git_root = Path(result.stdout.strip())
        config_abs = Path(file_path).resolve()

        # Get relative path from git root
        try:
            return str(config_abs.relative_to(git_root))
        except ValueError:
            # Config file is outside git repo
            return None

    except Exception:
        return None


def _get_git_info() -> dict[str, Any]:
    """Get git repository info for reproducibility.

    Returns dict with:
        commit: Current commit hash (short)
        branch: Current branch name
        dirty: Whether working directory has uncommitted changes
        commit_full: Full commit hash
    """
    import subprocess

    info: dict[str, Any] = {
        "commit": None,
        "branch": None,
        "dirty": None,
        "commit_full": None,
    }

    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["commit_full"] = result.stdout.strip()
            info["commit"] = info["commit_full"][:8]

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["dirty"] = len(result.stdout.strip()) > 0

    except Exception:
        pass  # Git info is best-effort

    return info


def _extract_evaluator_provenance(results: list[AttemptResult]) -> dict[str, Any] | None:
    """Best-effort scoring-runtime provenance extracted from sample metadata."""
    for sample in results:
        provenance = sample.metadata.get("evaluator_provenance")
        if isinstance(provenance, dict) and provenance:
            return provenance

        turn_history = sample.metadata.get("turn_history", [])
        if isinstance(turn_history, list):
            for turn in turn_history:
                if not isinstance(turn, dict):
                    continue
                provenance = turn.get("runtime_provenance")
                if isinstance(provenance, dict) and provenance:
                    return provenance

    return None


def _build_report_provenance(config: EvalConfig, results: list[AttemptResult]) -> dict[str, Any]:
    """Assemble report-level provenance from config metadata and sample outputs."""
    provenance: dict[str, Any] = {}

    if config.metadata:
        provenance.update(config.metadata)

    evaluator_provenance = _extract_evaluator_provenance(results)
    if evaluator_provenance is not None:
        provenance.setdefault("kernelbench_scoring_runtime", evaluator_provenance)

    return provenance


@dataclass
class EvalReport:
    """Summary report for an evaluation run."""

    eval_name: str
    dataset_path: str
    total_samples: int
    summary_metrics: dict[str, float]
    sample_results: list[AttemptResult]
    config: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    git_info: dict[str, Any] = field(default_factory=_get_git_info)
    config_path: str | None = None  # Path to config file relative to repo root

    async def save(self, output_dir: Path) -> None:
        """Save evaluation results to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save canonical per-sample attempt artifacts.
        samples_dir = output_dir / "samples"
        _write_sample_results(samples_dir, self.sample_results)

        # Save summary report
        summary = {
            "eval_name": self.eval_name,
            "dataset_path": self.dataset_path,
            "total_samples": self.total_samples,
            "summary_metrics": self.summary_metrics,
            "config": self.config,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
            "git_info": self.git_info,
            "config_path": self.config_path,
            "sample_ids": [s.id for s in self.sample_results],
        }
        # Sanitize API keys in the summary before saving
        summary = sanitize_api_keys(summary)
        report_file = output_dir / "report.json"
        report_file.write_text(json.dumps(summary, indent=2))
        _write_html_exports(
            output_dir,
            output_dir.name,
            cast(dict[str, Any], summary),
            self.sample_results,
        )

        logger.info(f"saved evaluation to {output_dir}")
        logger.info(f"  summary: {report_file}")
        logger.info(f"  samples: {samples_dir}")


def _write_partial_report(
    output_dir: Path,
    results: list[AttemptResult],
    config: EvalConfig,
    interrupted: bool = False,
    resume_from: int = 0,
) -> None:
    """Write a partial report to disk for crash recovery.

    Called incrementally during evaluation so results aren't lost on failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = output_dir / "samples"
    _write_sample_results(samples_dir, results)

    # Save partial summary
    summary_metrics = compute_summary_metrics(results)
    partial_report = {
        "eval_name": config.eval_name,
        "total_samples": len(results),
        "summary_metrics": summary_metrics,
        "provenance": _build_report_provenance(config, results),
        "interrupted": interrupted,
        "resume_from": resume_from,
        "sample_ids": [s.id for s in results],
        "timestamp": datetime.now().isoformat(),
    }
    partial_report = sanitize_api_keys(partial_report)
    report_file = output_dir / "report.json"
    report_file.write_text(json.dumps(partial_report, indent=2))
    _write_html_exports(
        output_dir,
        output_dir.name,
        cast(dict[str, Any], partial_report),
        results,
    )


def _sample_result_dict(sample: AttemptResult) -> dict[str, Any]:
    """Create the canonical per-sample attempt artifact."""
    sample_dict = sample.to_dict()

    metadata = dict(sample_dict.get("metadata") or {})
    metadata.pop("sample_data", None)
    sample_dict["metadata"] = metadata
    return sample_dict


def _write_sample_results(
    samples_dir: Path,
    results: list[AttemptResult],
) -> None:
    samples_dir.mkdir(exist_ok=True)
    for sample in results:
        sample_file = samples_dir / f"{sample.id}.json"
        sample_dict = _sample_result_dict(sample)
        sample_dict = sanitize_api_keys(sample_dict)
        sample_file.write_text(json.dumps(sample_dict, indent=2, default=str))


def _write_html_exports(
    output_dir: Path,
    trace_id: str,
    report: dict[str, Any],
    results: list[AttemptResult],
) -> None:
    samples_dir = output_dir / "samples"
    sample_payloads: list[dict[str, Any]] = []
    for sample in results:
        sample_dict = cast(dict[str, Any], sanitize_api_keys(_sample_result_dict(sample)))
        sample_payloads.append(sample_dict)
        sample_html = sample_to_html(trace_id, sample.id, sample_dict)
        (samples_dir / f"{sample.id}.html").write_text(sample_html)

    report_html = run_to_html(
        trace_id,
        report,
        sample_payloads,
        sample_link_prefix="samples",
    )
    (output_dir / "report.html").write_text(report_html)


def sanitize_api_keys(data: JsonValue) -> JsonValue:
    """Recursively sanitize API keys from nested data structures."""
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if key == "api_key" and isinstance(value, str) and value.startswith("sk-"):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = sanitize_api_keys(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_api_keys(item) for item in data]
    else:
        return data


@dataclass
class _AgentRunResult:
    """Result of running agent with error handling."""

    states: list[AgentState]
    final_trajectory: Trajectory
    error_message: str | None = None
    is_provider_error: bool = False


async def _close_environment(environment: Environment | None, sample_id: str) -> None:
    """Close environment, releasing external resources (sandboxes, containers, etc.)."""
    if environment is None:
        return
    close_fn = getattr(environment, "close", None)
    if close_fn is None:
        return
    try:
        await close_fn()
    except Exception as e:
        logger.warning(f"Environment close failed for {sample_id}: {e}")


async def _run_agent_with_error_handling(
    initial_state: AgentState,
    run_config: RunConfig,
    sample_id: str,
) -> _AgentRunResult:
    """Run agent and handle errors, returning structured result.

    Distinguishes provider errors (rate limits, timeouts) from actual failures.
    Provider errors are excluded from accuracy calculation.
    """
    from ..providers.base import ProviderError

    try:
        states = await run_agent(initial_state, run_config)
        return _AgentRunResult(
            states=states,
            final_trajectory=states[-1].actor.trajectory,
        )

    except ProviderError as e:
        error_message = f"ProviderError[{e.provider}]: {str(e)}"
        logger.warning(
            f"Sample {sample_id} provider_error: {error_message} (attempts: {e.attempts})"
        )
        final_trajectory = replace(
            initial_state.actor.trajectory,
            metadata={
                **initial_state.actor.trajectory.metadata,
                "error": error_message,
                "error_type": "provider_error",
                "provider": e.provider,
                "attempts": e.attempts,
            },
        )
        return _AgentRunResult(
            states=[initial_state],
            final_trajectory=final_trajectory,
            error_message=error_message,
            is_provider_error=True,
        )

    except Exception as e:
        error_message = f"{type(e).__name__}: {str(e)}"
        logger.warning(f"Sample {sample_id} failed: {error_message}")
        final_trajectory = replace(
            initial_state.actor.trajectory,
            metadata={
                **initial_state.actor.trajectory.metadata,
                "error": error_message,
                "error_type": "failed",
            },
        )
        return _AgentRunResult(
            states=[initial_state],
            final_trajectory=final_trajectory,
            error_message=error_message,
            is_provider_error=False,
        )


async def evaluate_sample(
    sample_data: dict[str, Any],
    sample_id: str,
    runtime: EvalRuntime,
    environment: Environment | None = None,
) -> AttemptResult:
    """Evaluate a single sample - analogous to run_agent_step.

    This is the atomic unit of evaluation that can be easily parallelized.
    Each call should receive a fresh environment instance to ensure state isolation.

    Args:
        sample_data: The raw sample data
        sample_id: Unique identifier for this sample
        runtime: Runtime context (config + instantiated limiters/progress)
        environment: Fresh Environment instance for this sample (None for tool-free eval)

    Returns:
        AttemptResult with trajectory and derived evaluation attached
    """
    # Unpack runtime for convenience
    config = runtime.config
    progress = runtime.progress

    initial_messages = config.prepare_messages(sample_data) if config.prepare_messages else []

    # Build base run config with concurrency limiters
    base_run_config = _build_base_run_config(config, runtime.api_limiter, runtime.tool_limiter)

    # Wrap on_chunk to inject sample_id context for concurrent sample tracking
    base_on_chunk = base_run_config.on_chunk
    last_status: dict[str, str] = {}  # Track last status to avoid duplicate events
    current_turn: dict[str, int] = {}  # Track current turn per sample for wide events

    async def on_chunk_with_sample_id(event: object) -> None:
        nonlocal last_status, current_turn

        # Update MultiProgress on various events for granular status
        status = _get_progress_status_for_event(event)
        turn = _get_turn_from_event(event)

        if progress is not None:
            if status is not None or turn is not None:
                progress.update_task(
                    sample_id,
                    turn=turn if turn is not None else None,
                    status=status if status is not None else None,
                )

        # Emit to the canonical events.jsonl stream.
        if isinstance(event, StreamChunk):
            if event.type == "turn_start":
                turn_num = event.data.get("turn", 0)
                current_turn[sample_id] = turn_num
                _event_logger.info(
                    "turn",
                    extra={
                        "sample_id": sample_id,
                        "turn": turn_num,
                        "status": "waiting",
                    },
                )
                last_status[sample_id] = "waiting"
            elif event.type == "modal_progress":
                _event_logger.info(
                    "modal_progress",
                    extra={
                        "sample_id": sample_id,
                        "phase": event.data.get("phase", ""),
                    },
                )
            elif event.type == "tool_calls_detected":
                _event_logger.info(
                    "tool_calls_detected",
                    extra={
                        "sample_id": sample_id,
                        "turn": event.data.get("turn", current_turn.get(sample_id, 0)),
                        "count": event.data.get("count", 0),
                        "tool_calls": event.data.get("tool_calls", []),
                    },
                )
            elif event.type == "tool_call_dispatch":
                _event_logger.info(
                    "tool_call_dispatch",
                    extra={
                        "sample_id": sample_id,
                        "turn": event.data.get("turn", current_turn.get(sample_id, 0)),
                        "tool_call_id": event.data.get("tool_call_id"),
                        "tool_name": event.data.get("tool_name"),
                        "action": event.data.get("action"),
                        "error": event.data.get("error"),
                    },
                )
            elif event.type == "kernel_submission":
                _event_logger.info(
                    "kernel_submission",
                    extra={
                        "sample_id": sample_id,
                        "turn": event.data.get("turn", current_turn.get(sample_id, 0)),
                        "source": event.data.get("source"),
                        "submission_state": event.data.get("submission_state"),
                        "code_length": event.data.get("code_length"),
                        "path": event.data.get("path"),
                        "tool_call_id": event.data.get("tool_call_id"),
                    },
                )
            elif event.type == "raw_driver_line":
                raw_line = event.data.get("raw_line")
                driver = event.data.get("driver")
                _event_logger.info(
                    "raw_driver_line",
                    extra={
                        "sample_id": sample_id,
                        "driver": driver,
                        "raw_line": raw_line,
                    },
                )
                if config.verbose and isinstance(raw_line, str):
                    if config.max_concurrent == 1:
                        sys.stderr.write(raw_line + "\n")
                    else:
                        sys.stderr.write(
                            json.dumps({
                                "sample_id": sample_id,
                                "driver": driver,
                                "raw_line": raw_line,
                            })
                            + "\n"
                        )
                    sys.stderr.flush()

        # Emit status changes (dedup to avoid flooding)
        if status is not None and status != last_status.get(sample_id):
            _event_logger.info("turn", extra={"sample_id": sample_id, "status": status})
            last_status[sample_id] = status

        # Wide events: detailed timing for performance analysis
        sample_turn = current_turn.get(sample_id, 0)
        if isinstance(event, FirstToken):
            _event_logger.info(
                "llm_first_token",
                extra={
                    "sample_id": sample_id,
                    "turn": sample_turn,
                    "ttft_ms": round(event.ttft_ms, 1),
                },
            )
        elif isinstance(event, LLMCallEnd):
            _event_logger.info(
                "llm_call",
                extra={
                    "sample_id": sample_id,
                    "turn": sample_turn,
                    "duration_ms": round(event.duration_ms, 1),
                    "ttft_ms": round(event.ttft_ms, 1) if event.ttft_ms is not None else None,
                    "provider": event.provider,
                    "model": event.model,
                    "tokens_in": event.tokens_in,
                    "tokens_out": event.tokens_out,
                    "status": event.status,
                    "error": event.error,
                },
            )
        elif isinstance(event, ToolExecutionEnd):
            _event_logger.info(
                "tool_execution",
                extra={
                    "sample_id": sample_id,
                    "turn": sample_turn,
                    "tool_name": event.tool_name,
                    "duration_ms": round(event.duration_ms, 1),
                    "status": event.status,
                    "is_error": event.is_error,
                    "result_summary": event.result_summary,
                },
            )
        elif isinstance(event, TextEnd):
            # Truncate large assistant messages so events.jsonl stays readable.
            content = event.content
            truncated = len(content) > 2000
            if truncated:
                content = content[:2000] + "..."
            _event_logger.info(
                "assistant_message",
                extra={
                    "sample_id": sample_id,
                    "turn": sample_turn,
                    "content": content,
                    "content_length": len(event.content),
                    "truncated": truncated,
                },
            )

        # DEBUG-only deltas are intentionally not persisted.
        elif isinstance(event, TextDelta):
            _event_logger.debug(
                "text_delta",
                extra={
                    "sample_id": sample_id,
                    "turn": sample_turn,
                    "text": event.delta,
                },
            )
        elif isinstance(event, ThinkingDelta):
            _event_logger.debug(
                "thinking_delta",
                extra={
                    "sample_id": sample_id,
                    "turn": sample_turn,
                    "text": event.delta,
                },
            )

        # Wrap event with sample_id and forward to base handler
        wrapped_event = _wrap_event_with_sample_id(event, sample_id)
        await base_on_chunk(wrapped_event)

    run_config = replace(base_run_config, on_chunk=on_chunk_with_sample_id)

    sample_name = sample_data.get("name", sample_id)
    if runtime.emit_sample_start:
        # Emit sample_start event for frontend live streaming.
        # Include initial_messages so streaming handlers can display them.
        await run_config.on_chunk(
            StreamChunk(
                "sample_start",
                {
                    "sample_id": sample_id,
                    "sample_data": sample_data,
                    "messages": [
                        {
                            "role": m.role,
                            "content": m.content if isinstance(m.content, str) else str(m.content),
                        }
                        for m in initial_messages
                    ],
                },
            )
        )

        # Emit sample_start for progress display.
        _event_logger.info(
            "sample_start", extra={"sample_id": sample_id, "sample_name": sample_name}
        )

    # Tiger Style: Catch operational errors (rate limits, network issues) at boundary
    # These are expected errors that should be reported, not crash the eval
    if config.verbose:
        logger.debug(f"Evaluating {sample_id}")

    start_time = time.time()
    final_env = environment
    try:
        if config.attempt_executor is not None:
            sample = await _run_attempt_executor(
                config.attempt_executor,
                sample_data=sample_data,
                sample_id=sample_id,
                environment=environment,
                run_config=run_config,
            )
            final_trajectory = sample.trajectory
            assert final_trajectory is not None
            final_env = environment
            env_state = (
                sample.environment_state
                if sample.environment_state is not None
                else await _serialize_environment_state(final_env)
            )
            if not sample.attempt_id:
                sample.attempt_id = sample_id
            if sample.problem is None:
                sample.problem = ProblemRow(
                    problem_id=sample_id,
                    payload=sample_data,
                    ground_truth=sample_data.get("ground_truth") or sample_data.get("answer"),
                    metadata=sample_data.get("metadata", {}),
                )
            combined_metadata = {
                **sample_data.get("metadata", {}),
                **final_trajectory.metadata,
                **sample.metadata,
            }
            if final_env is not None:
                runtime_metadata = getattr(final_env, "get_runtime_metadata", None)
                if callable(runtime_metadata):
                    extra_metadata = runtime_metadata()
                    if isinstance(extra_metadata, dict):
                        combined_metadata.update(extra_metadata)
            sample.environment_state = env_state
            sample.metadata = combined_metadata
            exec_metadata = {
                "turns_used": sample.metadata.get("turns_used", 0),
                "stop_reason": sample.metadata.get("stop_reason"),
                "total_tokens": sum(len(m.content or "") for m in final_trajectory.messages),
                "status": sample.metadata.get("status", "success"),
            }
            if sample.metadata.get("error") is not None:
                exec_metadata["error"] = sample.metadata["error"]
        else:
            initial_trajectory = Trajectory(
                messages=initial_messages,
                metadata={"sample_data": sample_data},
            )

            actor = Actor(
                trajectory=initial_trajectory,
                endpoint=config.endpoint,
                tools=environment.get_tools() if environment else [],
            )

            initial_state = AgentState(actor=actor, environment=environment)

            # Run agent with error handling
            result = await _run_agent_with_error_handling(initial_state, run_config, sample_id)
            states = result.states
            final_trajectory = result.final_trajectory
            error_message = result.error_message
            is_provider_error = result.is_provider_error

            final_env = states[-1].environment
            if final_env is not None:
                await _maybe_finalize_environment(
                    final_env,
                    run_config=run_config,
                    trajectory=final_trajectory,
                )
            env_state = await _serialize_environment_state(final_env)

            problem = ProblemRow(
                problem_id=sample_id,
                payload=sample_data,
                ground_truth=sample_data.get("ground_truth") or sample_data.get("answer"),
                metadata=sample_data.get("metadata", {}),
            )

            combined_metadata = {
                **sample_data.get("metadata", {}),
                **final_trajectory.metadata,
            }
            if final_env is not None:
                runtime_metadata = getattr(final_env, "get_runtime_metadata", None)
                if callable(runtime_metadata):
                    extra_metadata = runtime_metadata()
                    if isinstance(extra_metadata, dict):
                        combined_metadata.update(extra_metadata)
            sample = AttemptResult(
                attempt_id=sample_id,
                problem=problem,
                trajectory=final_trajectory,
                environment_state=env_state,
                metadata=combined_metadata,
            )

            exec_metadata = {
                "turns_used": states[-1].turn_idx,
                "stop_reason": str(states[-1].stop) if states[-1].stop else None,
                "total_tokens": sum(len(m.content or "") for m in final_trajectory.messages),
            }

            final_state = states[-1]

            if error_message:
                exec_metadata["error"] = error_message
                exec_metadata["status"] = "provider_error" if is_provider_error else "failed"
            elif final_state.error:
                exec_metadata["error"] = final_state.error
                exec_metadata["status"] = "failed"
            elif final_state.stop in (
                StopReason.ABORTED,
                StopReason.INTERRUPTED,
                StopReason.USER_ABORT,
            ):
                exec_metadata["status"] = "aborted"
            else:
                exec_metadata["status"] = "success"

        assert final_trajectory is not None
        sample.metadata = {**sample.metadata, **exec_metadata}
        sample.status = _map_exec_status(exec_metadata["status"])

        score: Score | None = None
        if exec_metadata["status"] != "aborted":
            score = await _compute_score(
                sample,
                scorer=config.scorer,
                scoring_context=ScoringContext(environment=final_env),
            )
            attach_score(sample, score)

        # Compute duration and log completion
        duration_seconds = time.time() - start_time
        exec_metadata["duration_seconds"] = duration_seconds
        reward = score.reward if score else 0.0

        _log_sample_completion(
            sample_id, reward, exec_metadata, final_trajectory, score, config.verbose
        )

        # Attach derived evaluation to the canonical execution result.
        if score is not None:
            sample.evaluation = AttemptEvaluation(reward=score.reward, score=score)

        # Emit sample_end event for frontend live streaming
        await run_config.on_chunk(
            StreamChunk(
                "sample_end",
                {"sample_id": sample_id, "reward": reward, "metadata": exec_metadata},
            )
        )

        _event_logger.info("sample_end", extra={"sample_id": sample_id, "score": reward})

        return sample
    finally:
        await _close_environment(final_env, sample_id)


async def evaluate(
    dataset: Iterator[dict[str, Any]],
    config: EvalConfig,
) -> EvalReport:
    """Run evaluation on a dataset - analogous to run_agent.

    This orchestrates evaluate_sample calls, potentially in parallel.
    Each sample gets a fresh environment instance to ensure state isolation.

    Args:
        dataset: Iterator of sample dictionaries
        config: Evaluation configuration (includes endpoint, template/prepare_messages,
                environment_factory, scorer, and execution settings)

    Returns:
        EvalReport with results and summary metrics

    Example:
        >>> config = EvalConfig(
        ...     endpoint=Endpoint.from_legacy(provider="openai", model="gpt-4o-mini"),
        ...     scorer=my_scorer,
        ...     template=PromptTemplate(system="...", user_template="{question}"),
        ... )
        >>> report = await evaluate(dataset, config)
    """
    import signal as _signal

    runtime_owner = config.environment_factory or config.environment
    eval_logging: EvalLoggingContext | None = None
    progress: MultiProgress | None = None
    results: list[AttemptResult] = []
    _interrupted = False

    # Cancel scope used by the SIGTERM handler so finally runs cleanly on kill.
    _outer_scope = trio.CancelScope()

    def _handle_sigterm(signum: int, frame: object) -> None:
        nonlocal _interrupted
        _interrupted = True
        _outer_scope.cancel()

    _old_sigterm = _signal.signal(_signal.SIGTERM, _handle_sigterm)

    report: EvalReport | None = None
    try:
        with _outer_scope:
            if runtime_owner is not None:
                await _maybe_start_environment_runtime(runtime_owner)

            samples_to_eval: list[tuple[str, dict[str, Any]]] = []
            samples_to_eval_dict: dict[str, dict[str, Any]] = {}
            for i, sample_data in enumerate(dataset):
                if config.max_samples and len(samples_to_eval) >= config.max_samples:
                    break
                sample_id = f"sample_{i:04d}"
                samples_to_eval.append((sample_id, sample_data))
                samples_to_eval_dict[sample_id] = sample_data

            if config.verbose:
                logger.info(f"starting evaluation: {config.eval_name}")
                logger.info(f"samples to evaluate: {len(samples_to_eval)}")
                logger.info(f"max concurrent: {config.max_concurrent}")
                logger.debug("=" * 50)

            if config.output_dir:
                eval_logging = setup_eval_logging(config.output_dir)
                _event_logger.info(
                    "eval_start",
                    extra={
                        "eval_name": config.eval_name,
                        "total": len(samples_to_eval),
                    },
                )

            if config.show_progress:
                progress = MultiProgress(
                    total=len(samples_to_eval),
                    desc=config.eval_name,
                    unit="sample",
                    verbose=config.verbose,
                )
                progress.__enter__()

            api_limiter = (
                trio.CapacityLimiter(config.max_api_concurrent)
                if config.max_api_concurrent is not None
                else None
            )
            tool_limiter = (
                trio.CapacityLimiter(config.max_tool_concurrent)
                if config.max_tool_concurrent is not None
                else None
            )

            runtime = EvalRuntime(
                config=config,
                api_limiter=api_limiter,
                tool_limiter=tool_limiter,
                progress=progress,
            )

            last_report_count = 0
            resume_from = 0

            def on_sample_complete(
                sample: AttemptResult,
                all_results: list[AttemptResult],
            ) -> None:
                nonlocal last_report_count
                if not config.output_dir:
                    return
                if len(all_results) - last_report_count >= config.report_batch_size:
                    _write_partial_report(
                        config.output_dir,
                        all_results,
                        config,
                        interrupted=False,
                        resume_from=resume_from,
                    )
                    last_report_count = len(all_results)

            results = await _evaluate_batch(samples_to_eval, runtime, on_sample_complete)

            if progress:
                progress.__exit__(None, None, None)
                progress = None

            for retry_attempt in range(config.max_sample_retries):
                failed_samples = [
                    (r.id, samples_to_eval_dict[r.id])
                    for r in results
                    if r.metadata.get("status") == "provider_error"
                ]

                if not failed_samples:
                    break

                wait_seconds = min(30 * (2**retry_attempt), 120)
                retry_msg = (
                    f"Retrying {len(failed_samples)} failed samples "
                    f"(attempt {retry_attempt + 1}/{config.max_sample_retries}, waiting {wait_seconds}s)"
                )
                if progress:
                    progress.log(retry_msg)
                else:
                    logger.info(retry_msg)
                await trio.sleep(wait_seconds)

                failed_ids = {sid for sid, _ in failed_samples}
                results = [r for r in results if r.id not in failed_ids]
                retry_runtime = EvalRuntime(
                    config=config,
                    api_limiter=api_limiter,
                    tool_limiter=tool_limiter,
                    progress=None,
                    emit_sample_start=False,
                )
                retry_results = await _evaluate_batch(failed_samples, retry_runtime)
                results.extend(retry_results)

                still_failed = sum(
                    1 for r in retry_results if r.metadata.get("status") == "provider_error"
                )
                succeeded = len(retry_results) - still_failed
                retry_result_msg = f"Retry {retry_attempt + 1}: {succeeded} succeeded, {still_failed} still failing"
                if progress:
                    progress.log(retry_result_msg)
                else:
                    logger.info(retry_result_msg)

    finally:
        _signal.signal(_signal.SIGTERM, _old_sigterm)
        if progress is not None:
            progress.__exit__(None, None, None)

        # Write report and emit eval_end regardless of how we exited.
        # results is [] if we were killed before any samples completed.
        if config.output_dir:
            summary_metrics = compute_summary_metrics(results)
            endpoint_config = (
                sanitize_api_keys(asdict(config.endpoint)) if config.endpoint else None
            )
            report = EvalReport(
                eval_name=config.eval_name,
                dataset_path=config.eval_name,
                total_samples=len(results),
                summary_metrics=summary_metrics,
                sample_results=results,
                config={
                    "endpoint": endpoint_config,
                    "max_samples": config.max_samples,
                    "max_concurrent": config.max_concurrent,
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "interrupted": _interrupted,
                },
                provenance=_build_report_provenance(config, results),
                config_path=config.config_path,
            )
            await report.save(config.output_dir)
        elif not _interrupted:
            # output_dir not set — build report in memory for return value only
            summary_metrics = compute_summary_metrics(results)
            endpoint_config = (
                sanitize_api_keys(asdict(config.endpoint)) if config.endpoint else None
            )
            report = EvalReport(
                eval_name=config.eval_name,
                dataset_path=config.eval_name,
                total_samples=len(results),
                summary_metrics=summary_metrics,
                sample_results=results,
                config={
                    "endpoint": endpoint_config,
                    "max_samples": config.max_samples,
                    "max_concurrent": config.max_concurrent,
                    "evaluation_timestamp": datetime.now().isoformat(),
                },
                provenance=_build_report_provenance(config, results),
                config_path=config.config_path,
            )

        if eval_logging:
            _event_logger.info(
                "eval_end",
                extra={
                    "eval_name": config.eval_name,
                    "total": len(results),
                    "interrupted": _interrupted,
                },
            )
            eval_logging.teardown()

        if config.verbose and results and report is not None:
            logger.info("")
            logger.debug("=" * 50)
            logger.info(f"Evaluation Summary: {config.eval_name}")
            logger.debug("=" * 50)
            logger.info(f"Samples evaluated: {len(results)}")
            for key, value in report.summary_metrics.items():
                if isinstance(value, int | float):
                    logger.info(f"{key}: {value:.3f}")
                else:
                    logger.info(f"{key}: {value}")

        if runtime_owner is not None:
            with trio.CancelScope(shield=True):
                await _maybe_stop_environment_runtime(runtime_owner)

    if report is None:
        raise RuntimeError(
            f"evaluate() produced no report for {config.eval_name!r} — "
            "eval was interrupted before any samples completed"
        )
    return report


def compute_summary_metrics(results: list[AttemptResult]) -> dict[str, float]:
    """Compute summary statistics from results using Score.

    Aggregates metrics from Score objects across all results.

    Provider errors are tracked separately from actual failed samples and are
    excluded from `success_rate` so transient infrastructure failures do not
    count as model/task failures.
    """
    if not results:
        return {}

    summary: dict[str, Any] = {}

    # Get all unique metric names from Score objects
    all_metric_names: set[str] = set()
    for r in results:
        if r.score:
            for m in r.score.metrics:
                all_metric_names.add(m.name)

    # Compute mean, min, max, std for each metric
    for metric_name in all_metric_names:
        values = []
        for r in results:
            if r.score:
                for m in r.score.metrics:
                    if m.name == metric_name:
                        values.append(m.value)
                        break
        if values:
            mean_val = sum(values) / len(values)
            summary[f"mean_{metric_name}"] = mean_val
            summary[f"min_{metric_name}"] = min(values)
            summary[f"max_{metric_name}"] = max(values)
            summary[f"std_{metric_name}"] = (
                sum((v - mean_val) ** 2 for v in values) / len(values)
            ) ** 0.5

    # Compute reward summary (the weighted score)
    rewards = [r.score.reward if r.score else 0.0 for r in results]
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        summary["mean_reward"] = mean_reward
        summary["min_reward"] = min(rewards)
        summary["max_reward"] = max(rewards)
        summary["std_reward"] = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5

    # Add metadata summaries
    summary["total_samples"] = len(results)
    summary["avg_turns"] = sum(r.metadata.get("turns_used", 0) for r in results) / len(results)
    summary["avg_tokens"] = sum(r.metadata.get("total_tokens", 0) for r in results) / len(results)

    # Separate provider errors from actual failures
    # Provider errors (rate limits, timeouts) are excluded from accuracy calculation
    provider_errors = [r for r in results if r.metadata.get("status") == "provider_error"]
    failed_samples = [r for r in results if r.metadata.get("status") == "failed"]
    aborted_samples = [r for r in results if r.metadata.get("status") == "aborted"]
    successful_samples = [r for r in results if r.metadata.get("status") == "success"]

    summary["provider_errors"] = len(provider_errors)
    summary["failed_samples"] = len(failed_samples)
    summary["aborted_samples"] = len(aborted_samples)
    summary["successful_samples"] = len(successful_samples)

    # Success rate excludes provider errors and operator-aborted runs from the denominator.
    # Those attempts did not produce a normal task outcome.
    valid_samples = len(results) - len(provider_errors) - len(aborted_samples)
    summary["success_rate"] = len(successful_samples) / valid_samples if valid_samples > 0 else 0.0

    # Completion rate counts terminal task outcomes (success or failure), excluding
    # provider errors and operator-aborted runs.
    completed_samples = len(successful_samples) + len(failed_samples)
    summary["completion_rate"] = completed_samples / len(results) if results else 0.0

    # Breakdown errors by type (for failed samples only, not provider errors)
    error_types: dict[str, int] = {}
    for r in failed_samples:
        error = r.metadata.get("error", "Unknown error")
        # Extract error type (e.g., "ValueError" from "ValueError: ...")
        error_type = error.split(":")[0] if ":" in error else error
        error_types[error_type] = error_types.get(error_type, 0) + 1

    if error_types:
        summary["error_breakdown"] = error_types

    # Breakdown provider errors by provider
    provider_breakdown: dict[str, int] = {}
    for r in provider_errors:
        # Extract provider from error message or metadata
        error = r.metadata.get("error", "")
        if "ProviderError[" in error:
            # Extract provider name from "ProviderError[anthropic]: ..."
            provider = error.split("[")[1].split("]")[0]
        else:
            provider = "unknown"
        provider_breakdown[provider] = provider_breakdown.get(provider, 0) + 1

    if provider_breakdown:
        summary["provider_error_breakdown"] = provider_breakdown

    return summary


# Dataset loaders
def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Load JSONL dataset."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_csv(path: Path) -> Iterator[dict[str, Any]]:
    """Load CSV dataset."""
    import csv

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


# Convenience function for simple evaluation
async def simple_evaluate(
    dataset_path: Path,
    config: EvalConfig,
) -> EvalReport:
    """Simple evaluation interface for common cases.

    Auto-detects dataset format (.jsonl or .csv) and runs evaluation.

    Args:
        dataset_path: Path to dataset file (.jsonl or .csv)
        config: Evaluation configuration (includes endpoint, template/prepare_messages,
                environment_factory, scorer, etc.)

    Returns:
        EvalReport with results and summary metrics

    Example:
        >>> config = EvalConfig(
        ...     endpoint=Endpoint(...),
        ...     scorer=my_scorer,
        ...     template=PromptTemplate(system="...", user_template="{question}"),
        ... )
        >>> report = await simple_evaluate(Path("data.jsonl"), config)
    """
    # Auto-detect dataset format
    if dataset_path.suffix == ".jsonl":
        dataset = load_jsonl(dataset_path)
    elif dataset_path.suffix == ".csv":
        dataset = load_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported format: {dataset_path.suffix}")

    return await evaluate(dataset, config)


# ── Analysis Helpers ──────────────────────────────────────────────────────────


def group_by(
    results: list[AttemptResult],
    key: Callable[[AttemptResult], str],
) -> dict[str, list[AttemptResult]]:
    """Group evaluation results by a key function.

    Pure function for slicing results by metadata.

    Examples:
        >>> by_difficulty = group_by(results, key=lambda r: r.metadata["difficulty"])
        >>> by_category = group_by(results, key=lambda r: r.input.get("category", "unknown"))

    Args:
        results: List of evaluation samples
        key: Function to extract grouping key from each sample

    Returns:
        Dict mapping group keys to lists of samples
    """
    groups: dict[str, list[AttemptResult]] = {}
    for result in results:
        k = key(result)
        if k not in groups:
            groups[k] = []
        groups[k].append(result)
    return groups


def summarize(results: list[AttemptResult]) -> dict[str, float]:
    """Compute summary statistics for a list of evaluation results.

    Pure function for aggregating metrics.

    Examples:
        >>> stats = summarize(results)
        >>> print(f"Mean reward: {stats['mean']:.2%}, n={stats['n']}")

    Args:
        results: List of evaluation samples

    Returns:
        Dict with mean, std, min, max, n for the reward signal
    """
    if not results:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}

    # Extract rewards - prefer Score.reward, fall back to metrics["reward"]
    rewards = []
    for r in results:
        if r.score is not None:
            rewards.append(r.score.reward)
        else:
            rewards.append(0.0)

    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = variance**0.5

    return {
        "mean": mean,
        "std": std,
        "min": min(rewards),
        "max": max(rewards),
        "n": n,
    }
