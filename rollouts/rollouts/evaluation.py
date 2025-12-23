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
from .progress import tqdm
from .training.types import Sample

logger = logging.getLogger(__name__)


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


def sanitize_api_keys(data: Any) -> Any:
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
) -> Sample:
    """Evaluate a single sample - analogous to run_agent_step.

    This is the atomic unit of evaluation that can be easily parallelized.
    Each call should receive a fresh environment instance to ensure state isolation.

    Args:
        sample_data: The raw sample data
        sample_id: Unique identifier for this sample
        config: Evaluation configuration (includes endpoint, template/prepare_messages)
        environment: Fresh Environment instance for this sample (None for tool-free eval)

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

    # Use run_config from EvalConfig (or default silent)
    # If user provided run_config, respect it but override show_progress
    # Disable inner turn-level progress bar during parallel execution to avoid conflicts
    show_turn_progress = config.show_progress and config.max_concurrent == 1

    if config.run_config:
        base_run_config = replace(config.run_config, show_progress=show_turn_progress)
    else:
        # Determine on_chunk handler based on stream_tokens flag
        # Debug logging
        has_stream_tokens = hasattr(config, "stream_tokens")
        stream_tokens_value = getattr(config, "stream_tokens", None)
        logger.debug(
            f"🔍 Checking stream_tokens: hasattr={has_stream_tokens}, value={stream_tokens_value}"
        )

        if has_stream_tokens and stream_tokens_value:
            # Import stdout_handler for streaming
            from rollouts.agents import stdout_handler

            on_chunk_handler = stdout_handler
            logger.debug("🔍 Using stdout_handler for token streaming")
        else:
            # Silent mode (default)
            async def silent_chunk_handler(_: object) -> None:
                await trio.lowlevel.checkpoint()

            on_chunk_handler = silent_chunk_handler
            logger.debug("🔍 Using silent mode (no token streaming)")

        base_run_config = RunConfig(on_chunk=on_chunk_handler, show_progress=show_turn_progress)
        logger.debug(
            f"🔍 RunConfig.on_chunk: {on_chunk_handler.__name__ if hasattr(on_chunk_handler, '__name__') else type(on_chunk_handler)}"
        )

    # Wrap on_chunk to inject sample_id context for concurrent sample tracking
    # This allows handlers to know which sample each event belongs to
    base_on_chunk = base_run_config.on_chunk

    async def on_chunk_with_sample_id(event: object) -> None:
        # Inject sample_id into StreamChunk events
        if isinstance(event, StreamChunk):
            event = StreamChunk(
                type=event.type,
                data={**event.data, "sample_id": sample_id},
                timestamp=event.timestamp,
            )
        else:
            # Wrap other event types in a StreamChunk with sample_id context
            # This allows handlers to track which sample each event belongs to
            event = StreamChunk(
                type="event_wrapper",
                data={"sample_id": sample_id, "event": event},
            )
        await base_on_chunk(event)

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

    error_message = None
    try:
        states = await run_agent(initial_state, run_config)
        final_trajectory = states[-1].actor.trajectory
    except Exception as e:
        # Tiger Style: Operational errors go in report, not traceback
        error_message = f"{type(e).__name__}: {str(e)}"
        logger.warning(f"Sample {sample_id} failed: {error_message}")

        # Create minimal trajectory with error
        states = [initial_state]
        final_trajectory = initial_state.actor.trajectory
        # Add error to metadata for analysis
        final_trajectory = replace(
            final_trajectory, metadata={**final_trajectory.metadata, "error": error_message}
        )

    # Build Sample with trajectory for score function
    sample = Sample(
        id=sample_id,
        input=sample_data,
        ground_truth=sample_data.get("ground_truth") or sample_data.get("answer"),
        trajectory=final_trajectory,
        metadata=sample_data.get("metadata", {}),
    )

    # Compute score: (Sample) -> Score
    # Support both sync and async score functions
    score: Score | None = None
    try:
        import inspect

        score_result = config.score_fn(sample)
        if inspect.iscoroutine(score_result):
            score = await score_result
        else:
            score = score_result
    except Exception as e:
        logger.exception(f"❌ SCORE COMPUTATION FAILED: {e}")
        # Return zero score on error
        score = Score(metrics=(Metric("error", 0.0, weight=1.0, metadata={"error": str(e)}),))

    # Add execution metadata
    exec_metadata = {
        "turns_used": states[-1].turn_idx,
        "stop_reason": str(states[-1].stop) if states[-1].stop else None,
        "total_tokens": sum(len(m.content or "") for m in final_trajectory.messages),
    }

    # Include error if agent execution failed
    if error_message:
        exec_metadata["error"] = error_message
        exec_metadata["status"] = "failed"
    else:
        exec_metadata["status"] = "success"

    # Compute duration
    duration_seconds = time.time() - start_time
    exec_metadata["duration_seconds"] = duration_seconds

    # Structured logging for sample completion (always logged at INFO level)
    reward = score.reward if score else 0.0
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
    def extract_text(content):
        """Extract text from message content (str or list of ContentBlocks)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Join text from TextContent blocks
            texts = []
            for block in content:
                if hasattr(block, "text"):
                    texts.append(block.text)
                elif hasattr(block, "content"):
                    texts.append(str(block.content))
            return "\n".join(texts) if texts else ""
        return str(content) if content else ""

    messages = [
        {"role": m.role, "content": extract_text(m.content)}
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

    if config.verbose and score:
        # Print metrics from Score
        metric_str = ", ".join(f"{m.name}={m.value:.3f}" for m in score.metrics[:3])
        logger.info(f"  {metric_str}")

    # Cleanup environment if it has a cleanup method
    if environment and hasattr(environment, "cleanup"):
        try:
            await environment.cleanup()
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
    samples_to_eval = []
    for i, sample_data in enumerate(dataset):
        if config.max_samples and len(samples_to_eval) >= config.max_samples:
            break
        sample_id = f"sample_{i:04d}"
        samples_to_eval.append((sample_id, sample_data))

    if config.verbose:
        logger.info(f"starting evaluation: {config.eval_name}")
        logger.info(f"samples to evaluate: {len(samples_to_eval)}")
        logger.info(f"max concurrent: {config.max_concurrent}")
        logger.debug("=" * 50)

    # Evaluate samples (with concurrency control)
    results = []

    # Initialize outer progress bar for sample-level tracking
    sample_pbar = None
    if config.show_progress:
        sample_pbar = tqdm(
            total=len(samples_to_eval),
            desc=f"{config.eval_name}",
            unit="sample",
            disable=False,
            # Show s/sample instead of sample/s for slow operations
            # This makes more sense when each sample takes multiple seconds
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]",
        )

    if config.max_concurrent == 1:
        # Sequential evaluation - create fresh environment for each sample
        for sample_id, sample_data in samples_to_eval:
            env = (
                await config.environment_factory(sample_data)
                if config.environment_factory
                else None
            )
            result = await evaluate_sample(
                sample_data=sample_data,
                sample_id=sample_id,
                config=config,
                environment=env,
            )
            results.append(result)

            # Update outer progress bar with reward
            if sample_pbar:
                sample_pbar.update(1)
                reward = result.score.reward if result.score else 0.0
                postfix = {"reward": f"{reward:.3f}"}
                if "turns_used" in result.metadata:
                    postfix["turns"] = result.metadata["turns_used"]
                sample_pbar.set_postfix(postfix)
    else:
        # Parallel evaluation with trio nursery - create fresh environment for each sample
        results = []

        async def eval_task(sample_id: str, sample_data: dict[str, Any]) -> None:
            env = (
                await config.environment_factory(sample_data)
                if config.environment_factory
                else None
            )
            result = await evaluate_sample(
                sample_data=sample_data,
                sample_id=sample_id,
                config=config,
                environment=env,
            )
            results.append(result)

            # Update outer progress bar for parallel execution
            if sample_pbar:
                sample_pbar.update(1)
                reward = result.score.reward if result.score else 0.0
                postfix = {"reward": f"{reward:.3f}"}
                if "turns_used" in result.metadata:
                    postfix["turns"] = result.metadata["turns_used"]
                sample_pbar.set_postfix(postfix)

        # Run tasks in parallel with bounded concurrency
        async with trio.open_nursery() as nursery:
            limiter = trio.CapacityLimiter(config.max_concurrent)
            for sample_id, sample_data in samples_to_eval:

                async def run_with_limit(
                    sid: str = sample_id, sdata: dict[str, Any] = sample_data
                ) -> None:
                    async with limiter:
                        await eval_task(sid, sdata)

                nursery.start_soon(run_with_limit)

    # Close outer progress bar
    if sample_pbar:
        sample_pbar.close()

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

    # Add error statistics
    failed_samples = [r for r in results if r.metadata.get("status") == "failed"]
    summary["failed_samples"] = len(failed_samples)
    summary["success_rate"] = (
        (len(results) - len(failed_samples)) / len(results) if results else 0.0
    )

    # Breakdown errors by type
    error_types: dict[str, int] = {}
    for r in failed_samples:
        error = r.metadata.get("error", "Unknown error")
        # Extract error type (e.g., "RateLimitError" from "RateLimitError: ...")
        error_type = error.split(":")[0] if ":" in error else error
        error_types[error_type] = error_types.get(error_type, 0) + 1

    if error_types:
        summary["error_breakdown"] = error_types

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
        elif "reward" in r.metrics:
            rewards.append(r.metrics["reward"])
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
