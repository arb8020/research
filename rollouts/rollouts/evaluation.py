"""Composable evaluation framework with first-class rewards.

Design mirrors run_agent/run_agent_step for easy parallelization.
Tiger Style: Pure functions, explicit configuration, no hidden state.
"""

import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import trio

from .agents import run_agent
from .dtypes import (
    Actor,
    AgentState,
    Environment,
    EvalConfig,
    Metric,
    RunConfig,
    Score,
    StreamChunk,
    Trajectory,
)
from .progress import MultiProgress
from .training.types import Sample

logger = logging.getLogger(__name__)

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

    # Handle streaming events from LLM
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
        short_name = tool_name.replace("write_kernel", "eval").replace("ask_docs", "docs")
        return f"calling {short_name}..."
    elif event_type == "tool_execution_start":
        tool_name = getattr(event, "tool_name", "tool")
        short_name = tool_name.replace("write_kernel", "eval").replace("ask_docs", "docs")
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
    config: "EvalConfig",
    api_limiter: trio.CapacityLimiter | None,
    tool_limiter: trio.CapacityLimiter | None,
    progress: MultiProgress | None,
) -> list[Sample]:
    """Evaluate a batch of samples, handling sequential vs parallel execution.

    This is the core evaluation loop, used for both initial runs and retries.
    """
    results: list[Sample] = []

    async def run_one(sample_id: str, sample_data: dict[str, Any]) -> Sample:
        """Evaluate a single sample."""
        task_name = sample_data.get("name", sample_id)
        if progress:
            progress.add_task(sample_id, name=task_name)

        env = await config.environment_factory(sample_data) if config.environment_factory else None
        result = await evaluate_sample(
            sample_data=sample_data,
            sample_id=sample_id,
            config=config,
            environment=env,
            api_limiter=api_limiter,
            tool_limiter=tool_limiter,
            progress=progress,
        )

        # Mark task complete
        if progress:
            reward = result.score.reward if result.score else 0.0
            success = result.metadata.get("status") == "success"
            if success:
                message = f"reward={reward:.2f}"
            else:
                error = result.metadata.get("error", "failed")
                message = error[:30] if len(error) > 30 else error
            progress.complete_task(sample_id, success=success, message=message)

        return result

    if config.max_concurrent == 1:
        # Sequential
        for sample_id, sample_data in samples:
            results.append(await run_one(sample_id, sample_data))
    else:
        # Parallel
        async with trio.open_nursery() as nursery:
            limiter = trio.CapacityLimiter(config.max_concurrent)

            async def run_with_limit(sid: str, sdata: dict[str, Any]) -> None:
                async with limiter:
                    results.append(await run_one(sid, sdata))

            for sample_id, sample_data in samples:
                nursery.start_soon(run_with_limit, sample_id, sample_data)

    return results


async def _compute_score(score_fn: Callable[..., Any], sample: Sample) -> Score:
    """Compute score, handling both sync and async score functions."""
    import inspect
    from typing import cast

    try:
        score_result = score_fn(sample)
        if inspect.iscoroutine(score_result):
            return await score_result
        else:
            return cast(Score, score_result)
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
        f"Sample {sample_id} completed: reward={reward:.3f}, "
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
            from .agents import stdout_handler

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


@dataclass
class EvalReport:
    """Summary report for an evaluation run."""

    eval_name: str
    dataset_path: str
    total_samples: int
    summary_metrics: dict[str, float]
    sample_results: list[Sample]
    config: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    async def save(self, output_dir: Path) -> None:
        """Save evaluation results to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual samples
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        for sample in self.sample_results:
            sample_file = samples_dir / f"{sample.id}.json"
            sample_dict = sample.to_dict()
            sample_dict = sanitize_api_keys(sample_dict)
            sample_file.write_text(json.dumps(sample_dict, indent=2, default=str))

        # Save summary report
        summary = {
            "eval_name": self.eval_name,
            "dataset_path": self.dataset_path,
            "total_samples": self.total_samples,
            "summary_metrics": self.summary_metrics,
            "config": self.config,
            "timestamp": self.timestamp,
            "sample_ids": [s.id for s in self.sample_results],
        }
        # Sanitize API keys in the summary before saving
        summary = sanitize_api_keys(summary)
        report_file = output_dir / "report.json"
        report_file.write_text(json.dumps(summary, indent=2))

        # Save trajectories separately for easy loading
        trajectories_dir = output_dir / "trajectories"
        trajectories_dir.mkdir(exist_ok=True)
        for sample in self.sample_results:
            if sample.trajectory:
                traj_file = trajectories_dir / f"{sample.id}.jsonl"
                Trajectory.save_jsonl([sample.trajectory], str(traj_file))

        logger.info(f"saved evaluation to {output_dir}")
        logger.info(f"  summary: {report_file}")
        logger.info(f"  samples: {samples_dir}")
        logger.info(f"  trajectories: {trajectories_dir}")


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


async def evaluate_sample(
    sample_data: dict[str, Any],
    sample_id: str,
    config: EvalConfig,
    environment: Environment | None = None,
    api_limiter: trio.CapacityLimiter | None = None,
    tool_limiter: trio.CapacityLimiter | None = None,
    progress: MultiProgress | None = None,
) -> Sample:
    """Evaluate a single sample - analogous to run_agent_step.

    This is the atomic unit of evaluation that can be easily parallelized.
    Each call should receive a fresh environment instance to ensure state isolation.

    Args:
        sample_data: The raw sample data
        sample_id: Unique identifier for this sample
        config: Evaluation configuration (includes endpoint, template/prepare_messages)
        environment: Fresh Environment instance for this sample (None for tool-free eval)
        api_limiter: Optional limiter for API calls (for two-level concurrency)
        tool_limiter: Optional limiter for tool executions (for two-level concurrency)
        progress: Optional MultiProgress for turn-level progress display

    Returns:
        Sample with trajectory, score, and computed reward
    """
    # Prepare initial messages from sample
    initial_messages = config.prepare_messages(sample_data)

    # Inject sample_data into trajectory metadata for score function access
    initial_trajectory = Trajectory(
        messages=initial_messages,
        metadata={"sample_data": sample_data},  # Ground truth available to score_fn
    )

    actor = Actor(
        trajectory=initial_trajectory,
        endpoint=config.endpoint,
        tools=environment.get_tools() if environment else [],
    )

    initial_state = AgentState(actor=actor, environment=environment)

    # Build base run config with concurrency limiters
    base_run_config = _build_base_run_config(config, api_limiter, tool_limiter)

    # Wrap on_chunk to inject sample_id context for concurrent sample tracking
    base_on_chunk = base_run_config.on_chunk

    async def on_chunk_with_sample_id(event: object) -> None:
        # Update MultiProgress on various events for granular status
        if progress is not None:
            status = _get_progress_status_for_event(event)
            turn = _get_turn_from_event(event)
            if status is not None or turn is not None:
                progress.update_task(
                    sample_id,
                    turn=turn if turn is not None else None,
                    status=status if status is not None else None,
                )

        # Wrap event with sample_id and forward to base handler
        wrapped_event = _wrap_event_with_sample_id(event, sample_id)
        await base_on_chunk(wrapped_event)

    run_config = replace(base_run_config, on_chunk=on_chunk_with_sample_id)

    # Run agent
    # Tiger Style: Catch operational errors (rate limits, network issues) at boundary
    # These are expected errors that should be reported, not crash the eval
    if config.verbose:
        logger.debug(f"Evaluating {sample_id}")

    # Track timing for structured logging
    start_time = time.time()

    # Emit sample_start event for frontend live streaming
    # Include initial_messages so streaming handlers can display them
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

    # Distinguish provider errors from actual sample failures
    # Provider errors (rate limits, timeouts, 5xx) are excluded from accuracy calculation
    # Actual failures (model got it wrong) count against accuracy
    from .providers.base import ProviderError

    error_message = None
    is_provider_error = False

    try:
        states = await run_agent(initial_state, run_config)
        final_trajectory = states[-1].actor.trajectory

    except ProviderError as e:
        # Provider infrastructure error - exclude from accuracy calculation
        is_provider_error = True
        error_message = f"ProviderError[{e.provider}]: {str(e)}"
        logger.warning(
            f"Sample {sample_id} provider_error: {error_message} (attempts: {e.attempts})"
        )

        # Create minimal trajectory with error
        states = [initial_state]
        final_trajectory = initial_state.actor.trajectory
        final_trajectory = replace(
            final_trajectory,
            metadata={
                **final_trajectory.metadata,
                "error": error_message,
                "error_type": "provider_error",
                "provider": e.provider,
                "attempts": e.attempts,
            },
        )

    except Exception as e:
        # Actual failure - counts against accuracy
        error_message = f"{type(e).__name__}: {str(e)}"
        logger.warning(f"Sample {sample_id} failed: {error_message}")

        # Create minimal trajectory with error
        states = [initial_state]
        final_trajectory = initial_state.actor.trajectory
        final_trajectory = replace(
            final_trajectory,
            metadata={
                **final_trajectory.metadata,
                "error": error_message,
                "error_type": "failed",
            },
        )

    # Serialize environment state for score function (agentic evals)
    env_state = None
    final_env = states[-1].environment
    if final_env is not None:
        try:
            env_state = await final_env.serialize()
        except Exception as e:
            logger.warning(f"Failed to serialize environment state: {e}")

    # Build Sample with trajectory for score function
    sample = Sample(
        id=sample_id,
        input=sample_data,
        ground_truth=sample_data.get("ground_truth") or sample_data.get("answer"),
        trajectory=final_trajectory,
        environment_state=env_state,
        metadata=sample_data.get("metadata", {}),
    )

    # Compute score
    score = await _compute_score(config.score_fn, sample)

    # Add execution metadata
    exec_metadata = {
        "turns_used": states[-1].turn_idx,
        "stop_reason": str(states[-1].stop) if states[-1].stop else None,
        "total_tokens": sum(len(m.content or "") for m in final_trajectory.messages),
    }

    # Include error if agent execution failed
    if error_message:
        exec_metadata["error"] = error_message
        exec_metadata["status"] = "provider_error" if is_provider_error else "failed"
    else:
        exec_metadata["status"] = "success"

    # Compute duration and log completion
    duration_seconds = time.time() - start_time
    exec_metadata["duration_seconds"] = duration_seconds
    reward = score.reward if score else 0.0

    _log_sample_completion(
        sample_id, reward, exec_metadata, final_trajectory, score, config.verbose
    )

    # Cleanup environment if it has a cleanup method
    if environment is not None:
        cleanup_fn = getattr(environment, "cleanup", None)
        if cleanup_fn is not None:
            try:
                await cleanup_fn()
            except Exception as e:
                logger.warning(f"Environment cleanup failed for {sample_id}: {e}")

    # Update sample with score and reward
    sample.score = score
    sample.reward = score.reward if score else 0.0
    sample.metadata = {**sample.metadata, **exec_metadata}

    # Emit sample_end event for frontend live streaming
    await run_config.on_chunk(
        StreamChunk(
            "sample_end",
            {"sample_id": sample_id, "reward": reward, "metadata": exec_metadata},
        )
    )

    return sample


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
                environment_factory, score_fn, and execution settings)

    Returns:
        EvalReport with results and summary metrics

    Example:
        >>> config = EvalConfig(
        ...     endpoint=Endpoint(provider="openai", model="gpt-4o-mini"),
        ...     score_fn=my_score_fn,
        ...     template=PromptTemplate(system="...", user_template="{question}"),
        ... )
        >>> report = await evaluate(dataset, config)
    """
    # Collect samples to evaluate
    samples_to_eval: list[tuple[str, dict[str, Any]]] = []
    samples_to_eval_dict: dict[str, dict[str, Any]] = {}  # For retry lookup
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

    # Evaluate samples (with concurrency control)
    results = []

    # Initialize progress display for sample-level tracking
    # MultiProgress shows each concurrent sample with turn-by-turn updates
    progress: MultiProgress | None = None
    if config.show_progress:
        progress = MultiProgress(
            total=len(samples_to_eval),
            desc=config.eval_name,
            unit="sample",
            verbose=config.verbose,  # verbose=True shows INFO logs, False shows only WARNING+
        )
        progress.__enter__()

    # Create two-level concurrency limiters if configured
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

    # Run initial evaluation batch
    results = await _evaluate_batch(samples_to_eval, config, api_limiter, tool_limiter, progress)

    # Close progress display
    if progress:
        progress.__exit__(None, None, None)

    # Sample-level retry for provider errors (rate limits, connection errors)
    for retry_attempt in range(config.max_sample_retries):
        failed_samples = [
            (r.id, samples_to_eval_dict[r.id])
            for r in results
            if r.metadata.get("status") == "provider_error"
        ]

        if not failed_samples:
            break  # All samples succeeded

        # Wait before retry (exponential backoff: 30s, 60s, 120s)
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

        # Remove failed samples and retry
        failed_ids = {sid for sid, _ in failed_samples}
        results = [r for r in results if r.id not in failed_ids]
        retry_results = await _evaluate_batch(
            failed_samples,
            config,
            api_limiter,
            tool_limiter,
            None,  # No progress for retries
        )
        results.extend(retry_results)

        # Log retry results
        still_failed = sum(1 for r in retry_results if r.metadata.get("status") == "provider_error")
        succeeded = len(retry_results) - still_failed
        retry_result_msg = (
            f"Retry {retry_attempt + 1}: {succeeded} succeeded, {still_failed} still failing"
        )
        if progress:
            progress.log(retry_result_msg)
        else:
            logger.info(retry_result_msg)

    # Compute summary metrics
    summary_metrics = compute_summary_metrics(results)

    # Create report
    # Sanitize endpoint config to exclude sensitive data
    endpoint_config = sanitize_api_keys(asdict(config.endpoint))

    report = EvalReport(
        eval_name=config.eval_name,
        dataset_path=config.eval_name,  # Use eval_name as dataset identifier
        total_samples=len(results),
        summary_metrics=summary_metrics,
        sample_results=results,
        config={
            "endpoint": endpoint_config,
            "max_samples": config.max_samples,
            "max_concurrent": config.max_concurrent,
            "evaluation_timestamp": datetime.now().isoformat(),
        },
    )

    # Save if output directory specified
    if config.output_dir:
        await report.save(config.output_dir)

    # Print summary
    if config.verbose:
        logger.info("")
        logger.debug("=" * 50)
        logger.info(f"Evaluation Summary: {config.eval_name}")
        logger.debug("=" * 50)
        logger.info(f"Samples evaluated: {len(results)}")
        for key, value in summary_metrics.items():
            # Handle both numeric and non-numeric values
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.3f}")
            else:
                logger.info(f"{key}: {value}")

    return report


def compute_summary_metrics(results: list[Sample]) -> dict[str, float]:
    """Compute summary statistics from results using Score.

    Aggregates metrics from Score objects across all results.

    TODO: Separate provider_error from failed samples in accuracy calculation
    Article quote: "As these samples get scored as failure, the scores for the
    corresponding provider are affected substantially."

    Problem: Currently failed_samples includes both actual failures AND provider errors.
    This inflates the failure rate when providers have issues (rate limits, timeouts, etc.)

    Fix: Track provider_errors separately and exclude from success_rate calculation:
        provider_errors = [r for r in results if r.metadata.get("status") == "provider_error"]
        actual_failures = [r for r in results if r.metadata.get("status") == "failed"]
        success_rate = (total - len(actual_failures)) / (total - len(provider_errors))
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
    successful_samples = [r for r in results if r.metadata.get("status") == "success"]

    summary["provider_errors"] = len(provider_errors)
    summary["failed_samples"] = len(failed_samples)
    summary["successful_samples"] = len(successful_samples)

    # Success rate excludes provider errors from denominator
    # (we can't count them as failures if the model never got to run)
    valid_samples = len(results) - len(provider_errors)
    summary["success_rate"] = len(successful_samples) / valid_samples if valid_samples > 0 else 0.0

    # Also provide raw completion rate (including provider errors as failures)
    summary["completion_rate"] = len(successful_samples) / len(results) if results else 0.0

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
                environment_factory, score_fn, etc.)

    Returns:
        EvalReport with results and summary metrics

    Example:
        >>> config = EvalConfig(
        ...     endpoint=Endpoint(...),
        ...     score_fn=my_score_fn,
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
    results: list[Sample],
    key: Callable[[Sample], str],
) -> dict[str, list[Sample]]:
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
    groups: dict[str, list[Sample]] = {}
    for result in results:
        k = key(result)
        if k not in groups:
            groups[k] = []
        groups[k].append(result)
    return groups


def summarize(results: list[Sample]) -> dict[str, float]:
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
