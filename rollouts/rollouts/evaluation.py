"""Composable evaluation framework with first-class rewards.

Design mirrors run_agent/run_agent_step for easy parallelization.
Tiger Style: Pure functions, explicit configuration, no hidden state.
"""

import json
import trio
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional, Iterator, Callable
from datetime import datetime

from .dtypes import (
    Trajectory, Message, Endpoint, Actor, AgentState,
    Environment, EvalConfig, RunConfig, RewardFunction
)
from .agents import run_agent
from .progress import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample with its result."""
    sample_id: str
    input_data: Dict[str, Any]
    trajectory: Trajectory
    agent_states: List[AgentState]  # Full list of agent states from run_agent
    metrics: Dict[str, float]  # All metrics are floats for RL compatibility
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self) -> str:
        """Serialize to JSON."""
        # Manually construct dict to avoid deep copying unpicklable objects (e.g., RLocks in environments)
        data = {
            'sample_id': self.sample_id,
            'input_data': self.input_data,
            'trajectory': json.loads(self.trajectory.to_json()),
            'agent_states': [],
            'metrics': self.metrics,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
        }

        # Serialize agent states carefully, excluding unpicklable environment objects
        for state in self.agent_states:
            state_dict = {
                'actor': asdict(state.actor) if hasattr(state.actor, '__dataclass_fields__') else str(state.actor),
                'environment': None,  # Skip environment - contains unpicklable objects like RLocks
                'max_turns': state.max_turns,
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
        # Note: Full AgentState deserialization would require complex reconstruction
        # For now, store as simplified data for analysis
        data['agent_states'] = data.get('agent_states', [])
        return EvalSample(**data)


@dataclass
class EvalReport:
    """Summary report for an evaluation run."""
    eval_name: str
    dataset_path: str
    total_samples: int
    summary_metrics: Dict[str, float]
    sample_results: List[EvalSample]
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, output_dir: Path) -> None:
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
                states_data = [asdict(state) for state in sample.agent_states]
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

        logger.info(f"Saved evaluation to {output_dir}")
        logger.info(f"  Summary: {report_file}")
        logger.info(f"  Samples: {samples_dir}")
        logger.info(f"  Trajectories: {trajectories_dir}")
        logger.info(f"  Agent States: {states_dir}")
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
    sample_data: Dict[str, Any],
    sample_id: str,
    prepare_messages: Callable[[Dict[str, Any]], List[Message]],
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
        environment=environment,
        max_turns=config.max_turns
    )

    # Use run_config from EvalConfig (or default silent)
    # If user provided run_config, respect it but override show_progress
    # Disable inner turn-level progress bar during parallel execution to avoid conflicts
    show_turn_progress = config.show_progress and config.max_concurrent == 1

    if config.run_config:
        run_config = replace(config.run_config, show_progress=show_turn_progress)
    else:
        # Determine on_chunk handler based on stream_tokens flag
        if hasattr(config, 'stream_tokens') and config.stream_tokens:
            # Import stdout_handler for streaming
            from rollouts.agents import stdout_handler
            on_chunk_handler = stdout_handler
        else:
            # Silent mode (default)
            on_chunk_handler = lambda _: trio.sleep(0)

        run_config = RunConfig(
            on_chunk=on_chunk_handler,
            show_progress=show_turn_progress
        )

    # Run agent
    # Tiger Style: Catch operational errors (rate limits, network issues) at boundary
    # These are expected errors that should be reported, not crash the eval
    if config.verbose:
        logger.info(f"Evaluating {sample_id}")

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

    # Compute reward (Trajectory -> Trajectory with rewards populated)
    # Support both sync and async reward functions
    try:
        reward_result = config.reward_fn(final_trajectory)
        # Check if result is a coroutine (async function)
        import inspect
        if inspect.iscoroutine(reward_result):
            scored_trajectory = await reward_result
        else:
            scored_trajectory = reward_result
    except Exception as e:
        # Always log reward computation errors loudly - these are critical issues
        logger.error(f"❌ REWARD COMPUTATION FAILED: {e}")
        if config.verbose:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # Return trajectory with 0 reward on error
        scored_trajectory = replace(final_trajectory, rewards=0.0)

    # Extract metrics
    metrics = {"reward": scored_trajectory.rewards}

    # If user stored breakdown in metadata, extract it
    if "reward_breakdown" in scored_trajectory.metadata:
        breakdown = scored_trajectory.metadata["reward_breakdown"]
        if isinstance(breakdown, dict):
            for name, value in breakdown.items():
                if isinstance(value, (int, float)):
                    metrics[name] = float(value)
                elif isinstance(value, dict) and "raw" in value:
                    metrics[name] = float(value["raw"])

    # Add execution metadata
    metadata = {
        "turns_used": states[-1].turn_idx,
        "stop_reason": str(states[-1].stop) if states[-1].stop else None,
        "total_tokens": sum(len(m.content or "") for m in scored_trajectory.messages),
    }

    # Include error if agent execution failed
    if error_message:
        metadata["error"] = error_message
        metadata["status"] = "failed"
    else:
        metadata["status"] = "success"

    if config.verbose and metrics:
        # Print key metrics inline
        metric_str = ", ".join(f"{k}={v:.3f}" for k, v in list(metrics.items())[:3])
        logger.info(f"  {metric_str}")

        # Also log detailed metrics from metadata if available
        # Look for any dict in metadata with rich metric information
        # (useful for environments that provide detailed feedback beyond just reward)
        for key, value in scored_trajectory.metadata.items():
            if isinstance(value, dict) and any(isinstance(v, (int, float)) for v in value.values()):
                # Found a dict with numeric values - likely detailed metrics
                detail_items = [(k, v) for k, v in value.items()
                               if isinstance(v, (int, float))][:3]
                if detail_items:
                    detail_str = ", ".join(f"{k}={v:.3f}" for k, v in detail_items)
                    # Use the metadata key name (e.g., "prime_metrics" -> "prime")
                    label = key.replace('_metrics', '').replace('_', ' ')
                    logger.info(f"  {label}: {detail_str}")
                break  # Only show first metrics dict to avoid spam

    # Cleanup environment if it has a cleanup method
    if environment and hasattr(environment, 'cleanup'):
        try:
            await environment.cleanup()
        except Exception as e:
            logger.warning(f"Environment cleanup failed for {sample_id}: {e}")

    return EvalSample(
        sample_id=sample_id,
        input_data=sample_data,
        trajectory=scored_trajectory,
        agent_states=states,
        metrics=metrics,
        metadata=metadata
    )


async def evaluate(
    dataset: Iterator[Dict[str, Any]],
    prepare_messages: Callable[[Dict[str, Any]], List[Message]],
    endpoint: Endpoint,
    config: EvalConfig,
    dataset_path: str = "unknown",
    environment_factory: Callable[[Dict[str, Any]], Environment] | None = None,
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
        environment_factory: Optional factory function that takes sample_data and returns
                           a fresh Environment instance. Example: `lambda sample: MyEnv(sample)`
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
        logger.info(f"Starting evaluation: {config.eval_name}")
        logger.info(f"Samples to evaluate: {len(samples_to_eval)}")
        logger.info(f"Max concurrent: {config.max_concurrent}")
        logger.info("="*50)

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
                postfix = {'reward': f"{result.metrics.get('reward', 0):.3f}"}
                if 'turns_used' in result.metadata:
                    postfix['turns'] = result.metadata['turns_used']
                sample_pbar.set_postfix(postfix)
    else:
        # Parallel evaluation with trio nursery - create fresh environment for each sample
        results = []

        async def eval_task(sample_id: str, sample_data: Dict[str, Any]) -> None:
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
                postfix = {'reward': f"{result.metrics.get('reward', 0):.3f}"}
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
            "max_turns": config.max_turns,
            "max_samples": config.max_samples,
            "max_concurrent": config.max_concurrent,
            "evaluation_timestamp": datetime.now().isoformat(),
        }
    )

    # Save if output directory specified
    if config.output_dir:
        report.save(config.output_dir)

    # Print summary
    if config.verbose:
        logger.info("")
        logger.info("="*50)
        logger.info(f"Evaluation Summary: {config.eval_name}")
        logger.info("="*50)
        logger.info(f"Samples evaluated: {len(results)}")
        for key, value in summary_metrics.items():
            # Handle both numeric and non-numeric values
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.3f}")
            else:
                logger.info(f"{key}: {value}")

    return report


def compute_summary_metrics(results: List[EvalSample]) -> Dict[str, float]:
    """Compute summary statistics from results."""
    if not results:
        return {}

    summary = {}

    # Get all unique metric names from results
    all_metric_names = set()
    for r in results:
        all_metric_names.update(r.metrics.keys())

    # Compute mean, min, max, std for each metric
    for metric_name in all_metric_names:
        values = [r.metrics.get(metric_name, 0.0) for r in results]
        if values:
            mean_val = sum(values) / len(values)
            summary[f"mean_{metric_name}"] = mean_val
            summary[f"min_{metric_name}"] = min(values)
            summary[f"max_{metric_name}"] = max(values)
            summary[f"std_{metric_name}"] = (
                sum((v - mean_val) ** 2 for v in values) / len(values)
            ) ** 0.5

    # Add metadata summaries
    summary["total_samples"] = len(results)
    summary["avg_turns"] = sum(r.metadata.get("turns_used", 0) for r in results) / len(results)
    summary["avg_tokens"] = sum(r.metadata.get("total_tokens", 0) for r in results) / len(results)

    # Add error statistics
    failed_samples = [r for r in results if r.metadata.get("status") == "failed"]
    summary["failed_samples"] = len(failed_samples)
    summary["success_rate"] = (len(results) - len(failed_samples)) / len(results) if results else 0.0

    # Breakdown errors by type
    error_types = {}
    for r in failed_samples:
        error = r.metadata.get("error", "Unknown error")
        # Extract error type (e.g., "RateLimitError" from "RateLimitError: ...")
        error_type = error.split(":")[0] if ":" in error else error
        error_types[error_type] = error_types.get(error_type, 0) + 1

    if error_types:
        summary["error_breakdown"] = error_types

    return summary


# Dataset loaders
def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Load JSONL dataset."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_csv(path: Path) -> Iterator[Dict[str, Any]]:
    """Load CSV dataset."""
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


# Convenience function for simple evaluation
async def simple_evaluate(
    dataset_path: Path,
    prepare_messages: Callable[[Dict[str, Any]], List[Message]],
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
