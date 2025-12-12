"""Composable evaluation framework with first-class rewards.

Design mirrors run_agent/run_agent_step for easy parallelization.
Tiger Style: Pure functions, explicit configuration, no hidden state.
"""

import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import trio

from .agents import run_agent
from .dtypes import (
    Actor,
    AgentState,
    Endpoint,
    Environment,
    EvalConfig,
    Message,
    Metric,
    RunConfig,
    Sample,
    Score,
    StreamChunk,
    Trajectory,
)
from .progress import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample with its result.

    Attributes:
        sample_id: Unique identifier for this sample
        input_data: Raw input data from dataset
        trajectory: Full agent execution trace
        agent_states: List of agent states from run_agent
        score: Structured score with metrics breakdown
        metadata: Execution metadata (turns, tokens, status, etc.)
        timestamp: When this sample was evaluated
    """
    sample_id: str
    input_data: dict[str, Any]
    trajectory: Trajectory
    agent_states: list[AgentState]
    score: Score | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self) -> str:
        """Serialize to JSON."""
        # Manually construct dict to avoid deep copying unpicklable objects (e.g., RLocks in environments)

        # Serialize Score if present
        score_data = None
        if self.score is not None:
            score_data = {
                'metrics': [
                    {'name': m.name, 'value': m.value, 'weight': m.weight, 'metadata': m.metadata}
                    for m in self.score.metrics
                ],
                'reward': self.score.reward,
            }

        data = {
            'sample_id': self.sample_id,
            'input_data': self.input_data,
            'trajectory': json.loads(self.trajectory.to_json()),
            'agent_states': [],
            'score': score_data,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
        }

        # Serialize agent states carefully, excluding unpicklable environment objects
        for state in self.agent_states:
            state_dict = {
                'actor': asdict(state.actor) if hasattr(state.actor, '__dataclass_fields__') else str(state.actor),
                'environment': None,  # Skip environment - contains unpicklable objects like RLocks
                'stop': asdict(state.stop) if state.stop and hasattr(state.stop, '__dataclass_fields__') else str(state.stop) if state.stop else None,
                'turn_idx': state.turn_idx,
                'pending_tool_calls': [asdict(tc) if hasattr(tc, '__dataclass_fields__') else str(tc) for tc in state.pending_tool_calls],
                'next_tool_idx': state.next_tool_idx,
                'timestamp': state.timestamp,
            }
            data['agent_states'].append(state_dict)

        # Sanitize all API keys recursively
        data = sanitize_api_keys(data)
        return json.dumps(data, indent=2, default=str)  # default=str handles datetime objects

    @staticmethod
    def from_json(json_str: str) -> 'EvalSample':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        data['trajectory'] = Trajectory.from_json(json.dumps(data['trajectory']))

        # Deserialize Score if present
        score_data = data.pop('score', None)
        if score_data is not None:
            metrics = tuple(
                Metric(
                    name=m['name'],
                    value=m['value'],
                    weight=m.get('weight', 0.0),
                    metadata=m.get('metadata', {}),
                )
                for m in score_data['metrics']
            )
            data['score'] = Score(metrics=metrics)

        # Note: Full AgentState deserialization would require complex reconstruction
        # For now, store as simplified data for analysis
        data['agent_states'] = data.get('agent_states', [])

        # Only keep expected fields
        return EvalSample(
            sample_id=data['sample_id'],
            input_data=data['input_data'],
            trajectory=data['trajectory'],
            agent_states=data['agent_states'],
            score=data.get('score'),
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
        )


@dataclass
class EvalReport:
    """Summary report for an evaluation run."""
    eval_name: str
    dataset_path: str
    total_samples: int
    summary_metrics: dict[str, float]
    sample_results: list[EvalSample]
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
            sample_file = samples_dir / f"{sample.sample_id}.json"
            sample_file.write_text(sample.to_json())

        # Save summary report
        summary = {
            "eval_name": self.eval_name,
            "dataset_path": self.dataset_path,
            "total_samples": self.total_samples,
            "summary_metrics": self.summary_metrics,
            "config": self.config,
            "timestamp": self.timestamp,
            "sample_ids": [s.sample_id for s in self.sample_results]
        }
        # Sanitize API keys in the summary before saving
        summary = sanitize_api_keys(summary)
        report_file = output_dir / "report.json"
        report_file.write_text(json.dumps(summary, indent=2))

        # Save trajectories separately for easy loading
        trajectories_dir = output_dir / "trajectories"
        trajectories_dir.mkdir(exist_ok=True)
        for sample in self.sample_results:
            traj_file = trajectories_dir / f"{sample.sample_id}.jsonl"
            Trajectory.save_jsonl([sample.trajectory], str(traj_file))

        # Save agent states separately for detailed analysis
        states_dir = output_dir / "agent_states"
        states_dir.mkdir(exist_ok=True)
        failed_serializations = []
        for sample in self.sample_results:
            states_file = states_dir / f"{sample.sample_id}.json"
            try:
                # Serialize agent states using environment's serialize() method
                states_data = []
                for state in sample.agent_states:
                    # First serialize environment separately to avoid thread handle issues
                    # (Environment may contain SSH connections with unpicklable thread handles)
                    if state.environment and hasattr(state.environment, 'serialize'):
                        env_data = await state.environment.serialize()
                    else:
                        env_data = None

                    # Create a temporary state without environment to avoid asdict() issues
                    # with unpicklable objects (e.g., thread handles in SSH connections)
                    temp_state = replace(state, environment=None)
                    state_dict = asdict(temp_state)

                    # Add back the serialized environment
                    state_dict['environment'] = env_data
                    states_data.append(state_dict)

                # Sanitize all API keys recursively
                states_data = sanitize_api_keys(states_data)
                states_file.write_text(json.dumps(states_data, indent=2, default=str))
            except (TypeError, ValueError) as e:
                # Some environments contain unpicklable objects (e.g., thread locks)
                # Log warning but don't fail the entire evaluation
                logger.warning(f"Failed to serialize agent states for {sample.sample_id}: {e}")
                failed_serializations.append(sample.sample_id)
                # Save a placeholder file indicating serialization failed
                states_file.write_text(json.dumps({
                    "error": "Serialization failed",
                    "reason": str(e),
                    "sample_id": sample.sample_id
                }, indent=2))

        logger.info(f"saved evaluation to {output_dir}")
        logger.info(f"  summary: {report_file}")
        logger.info(f"  samples: {samples_dir}")
        logger.info(f"  trajectories: {trajectories_dir}")
        logger.info(f"  agent states: {states_dir}")
        if failed_serializations:
            logger.warning(f"  Failed to serialize {len(failed_serializations)} agent states: {failed_serializations[:5]}")


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
    prepare_messages: Callable[[dict[str, Any]], list[Message]],
    environment: Environment | None,
    endpoint: Endpoint,
    config: EvalConfig,
) -> EvalSample:
    """Evaluate a single sample - analogous to run_agent_step.

    This is the atomic unit of evaluation that can be easily parallelized.
    Each call should receive a fresh environment instance to ensure state isolation.

    Args:
        sample_data: The raw sample data
        sample_id: Unique identifier for this sample
        prepare_messages: Function to create initial messages
        environment: Fresh Environment instance for this sample (None for tool-free eval)
        endpoint: LLM endpoint
        config: Evaluation configuration

    Returns:
        EvalSample with trajectory and computed metrics
    """
    # Prepare initial state
    initial_messages = prepare_messages(sample_data)

    # Inject sample_data into trajectory metadata for reward function access
    initial_trajectory = Trajectory(
        messages=initial_messages,
        metadata={"sample_data": sample_data}  # Ground truth available to reward_fn
    )

    actor = Actor(
        trajectory=initial_trajectory,
        endpoint=endpoint,
        tools=environment.get_tools() if environment else []
    )

    initial_state = AgentState(
        actor=actor,
        environment=environment
    )

    # Use run_config from EvalConfig (or default silent)
    # If user provided run_config, respect it but override show_progress
    # Disable inner turn-level progress bar during parallel execution to avoid conflicts
    show_turn_progress = config.show_progress and config.max_concurrent == 1

    if config.run_config:
        run_config = replace(config.run_config, show_progress=show_turn_progress)
    else:
        # Determine on_chunk handler based on stream_tokens flag
        # Debug logging
        has_stream_tokens = hasattr(config, 'stream_tokens')
        stream_tokens_value = getattr(config, 'stream_tokens', None)
        logger.debug(f"🔍 Checking stream_tokens: hasattr={has_stream_tokens}, value={stream_tokens_value}")

        if has_stream_tokens and stream_tokens_value:
            # Import stdout_handler for streaming
            from rollouts.agents import stdout_handler
            on_chunk_handler = stdout_handler
            logger.debug("🔍 Using stdout_handler for token streaming")
        else:
            # Silent mode (default)
            on_chunk_handler = lambda _: trio.lowlevel.checkpoint()
            logger.debug("🔍 Using silent mode (no token streaming)")

        run_config = RunConfig(
            on_chunk=on_chunk_handler,
            show_progress=show_turn_progress
        )
        logger.debug(f"🔍 RunConfig.on_chunk: {on_chunk_handler.__name__ if hasattr(on_chunk_handler, '__name__') else type(on_chunk_handler)}")

    # Run agent
    # Tiger Style: Catch operational errors (rate limits, network issues) at boundary
    # These are expected errors that should be reported, not crash the eval
    if config.verbose:
        logger.debug(f"Evaluating {sample_id}")

    # Track timing for structured logging
    start_time = time.time()

    # Emit sample_start event for frontend live streaming
    await run_config.on_chunk(StreamChunk(
        "sample_start",
        {"sample_id": sample_id, "sample_data": sample_data},
    ))

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
            final_trajectory,
            metadata={**final_trajectory.metadata, "error": error_message}
        )

    # Build Sample for score function
    sample = Sample(
        id=sample_id,
        input=sample_data,
        ground_truth=sample_data.get("ground_truth") or sample_data.get("answer"),
        metadata=sample_data.get("metadata", {}),
    )

    # Compute score (Trajectory, Sample) -> Score
    # Support both sync and async score functions
    score: Score | None = None
    try:
        import inspect
        score_result = config.score_fn(final_trajectory, sample)
        if inspect.iscoroutine(score_result):
            score = await score_result
        else:
            score = score_result
    except Exception as e:
        logger.error(f"❌ SCORE COMPUTATION FAILED: {e}")
        if config.verbose:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # Return zero score on error
        score = Score(metrics=(Metric("error", 0.0, weight=1.0, metadata={"error": str(e)}),))

    # Add execution metadata
    metadata = {
        "turns_used": states[-1].turn_idx,
        "stop_reason": str(states[-1].stop) if states[-1].stop else None,
        "total_tokens": sum(len(m.content or "") for m in final_trajectory.messages),
    }

    # Include error if agent execution failed
    if error_message:
        metadata["error"] = error_message
        metadata["status"] = "failed"
    else:
        metadata["status"] = "success"

    # Compute duration
    duration_seconds = time.time() - start_time
    metadata["duration_seconds"] = duration_seconds

    # Structured logging for sample completion (always logged at INFO level)
    reward = score.reward if score else 0.0
    logger.info(
        f"Sample {sample_id} completed: reward={reward:.3f}, "
        f"turns={metadata['turns_used']}, duration={duration_seconds:.2f}s, "
        f"status={metadata['status']}",
        extra={
            "sample_id": sample_id,
            "reward": reward,
            "turns": metadata["turns_used"],
            "duration_seconds": duration_seconds,
            "status": metadata["status"],
            "stop_reason": metadata.get("stop_reason"),
        }
    )

    if config.verbose and score:
        # Print metrics from Score
        metric_str = ", ".join(f"{m.name}={m.value:.3f}" for m in score.metrics[:3])
        logger.info(f"  {metric_str}")

    # Cleanup environment if it has a cleanup method
    if environment and hasattr(environment, 'cleanup'):
        try:
            await environment.cleanup()
        except Exception as e:
            logger.warning(f"Environment cleanup failed for {sample_id}: {e}")

    eval_sample = EvalSample(
        sample_id=sample_id,
        input_data=sample_data,
        trajectory=final_trajectory,
        agent_states=states,
        score=score,
        metadata=metadata
    )

    # Emit sample_end event for frontend live streaming
    await run_config.on_chunk(StreamChunk(
        "sample_end",
        {"sample_id": sample_id, "reward": reward, "metadata": metadata},
    ))

    return eval_sample


async def evaluate(
    dataset: Iterator[dict[str, Any]],
    prepare_messages: Callable[[dict[str, Any]], list[Message]],
    endpoint: Endpoint,
    config: EvalConfig,
    dataset_path: str = "unknown",
    environment_factory: Callable[[dict[str, Any]], Awaitable[Environment]] | None = None,
) -> EvalReport:
    """Run evaluation on a dataset - analogous to run_agent.

    This orchestrates evaluate_sample calls, potentially in parallel.
    Each sample gets a fresh environment instance to ensure state isolation.

    Args:
        dataset: Iterator of sample dictionaries
        prepare_messages: Function to create initial messages from sample
        endpoint: LLM endpoint configuration
        config: Evaluation configuration
        dataset_path: Path/name of dataset for logging
        environment_factory: Optional async factory function that takes sample_data and returns
                           a fresh Environment instance. Example: `async def factory(sample): return MyEnv(sample)`
                           If None, samples run without environment (tool-free evaluation).

    Returns:
        EvalReport with results and summary metrics
    """
    # Collect samples to evaluate
    samples_to_eval = []
    for i, sample_data in enumerate(dataset):
        if config.max_samples and len(samples_to_eval) >= config.max_samples:
            break
        sample_id = config.sample_id_fn(i, sample_data)
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
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]'
        )

    if config.max_concurrent == 1:
        # Sequential evaluation - create fresh environment for each sample
        for sample_id, sample_data in samples_to_eval:
            env = await environment_factory(sample_data) if environment_factory else None
            result = await evaluate_sample(
                sample_data=sample_data,
                sample_id=sample_id,
                prepare_messages=prepare_messages,
                environment=env,
                endpoint=endpoint,
                config=config,
            )
            results.append(result)

            # Update outer progress bar with reward
            if sample_pbar:
                sample_pbar.update(1)
                reward = result.score.reward if result.score else 0.0
                postfix = {'reward': f"{reward:.3f}"}
                if 'turns_used' in result.metadata:
                    postfix['turns'] = result.metadata['turns_used']
                sample_pbar.set_postfix(postfix)
    else:
        # Parallel evaluation with trio nursery - create fresh environment for each sample
        results = []

        async def eval_task(sample_id: str, sample_data: dict[str, Any]) -> None:
            env = await environment_factory(sample_data) if environment_factory else None
            result = await evaluate_sample(
                sample_data=sample_data,
                sample_id=sample_id,
                prepare_messages=prepare_messages,
                environment=env,
                endpoint=endpoint,
                config=config,
            )
            results.append(result)

            # Update outer progress bar for parallel execution
            if sample_pbar:
                sample_pbar.update(1)
                reward = result.score.reward if result.score else 0.0
                postfix = {'reward': f"{reward:.3f}"}
                if 'turns_used' in result.metadata:
                    postfix['turns'] = result.metadata['turns_used']
                sample_pbar.set_postfix(postfix)

        # Run tasks in parallel with bounded concurrency
        async with trio.open_nursery() as nursery:
            limiter = trio.CapacityLimiter(config.max_concurrent)
            for sample_id, sample_data in samples_to_eval:
                async def run_with_limit(sid=sample_id, sdata=sample_data):
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
    endpoint_config = sanitize_api_keys(asdict(endpoint))

    report = EvalReport(
        eval_name=config.eval_name,
        dataset_path=dataset_path,
        total_samples=len(results),
        summary_metrics=summary_metrics,
        sample_results=results,
        config={
            "endpoint": endpoint_config,
            "max_samples": config.max_samples,
            "max_concurrent": config.max_concurrent,
            "evaluation_timestamp": datetime.now().isoformat(),
        }
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


def compute_summary_metrics(results: list[EvalSample]) -> dict[str, float]:
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
    summary["success_rate"] = (len(results) - len(failed_samples)) / len(results) if results else 0.0

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
    prepare_messages: Callable[[dict[str, Any]], list[Message]],
    environment_factory: Callable[[], Environment],
    endpoint: Endpoint,
    config: EvalConfig,
) -> EvalReport:
    """Simple evaluation interface for common cases.

    Auto-detects dataset format and provides sensible defaults.

    Args:
        dataset_path: Path to dataset file (.jsonl or .csv)
        prepare_messages: Function to create initial messages from sample
        environment_factory: Factory function returning fresh Environment instances
        endpoint: LLM endpoint configuration
        config: Evaluation configuration
    """
    # Auto-detect dataset format
    if dataset_path.suffix == ".jsonl":
        dataset = load_jsonl(dataset_path)
    elif dataset_path.suffix == ".csv":
        dataset = load_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported format: {dataset_path.suffix}")

    return await evaluate(
        dataset=dataset,
        prepare_messages=prepare_messages,
        environment_factory=environment_factory,
        endpoint=endpoint,
        config=config,
        dataset_path=str(dataset_path),
    )


# ── Analysis Helpers ──────────────────────────────────────────────────────────


def group_by(
    results: list[EvalSample],
    key: Callable[[EvalSample], str],
) -> dict[str, list[EvalSample]]:
    """Group evaluation results by a key function.

    Pure function for slicing results by metadata.

    Examples:
        >>> by_difficulty = group_by(results, key=lambda r: r.metadata["difficulty"])
        >>> by_category = group_by(results, key=lambda r: r.input_data.get("category", "unknown"))

    Args:
        results: List of evaluation samples
        key: Function to extract grouping key from each sample

    Returns:
        Dict mapping group keys to lists of samples
    """
    groups: dict[str, list[EvalSample]] = {}
    for result in results:
        k = key(result)
        if k not in groups:
            groups[k] = []
        groups[k].append(result)
    return groups


def summarize(results: list[EvalSample]) -> dict[str, float]:
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
    std = variance ** 0.5

    return {
        "mean": mean,
        "std": std,
        "min": min(rewards),
        "max": max(rewards),
        "n": n,
    }
