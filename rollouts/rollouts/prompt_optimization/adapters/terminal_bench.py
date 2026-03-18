"""Terminal-Bench adapter for GEPA.

Optimizes instruction prompts for terminal-use agents using our agent system.
Uses TerminalBenchEnvironment to run tasks in Docker containers with tmux.

This adapter provides:
- TerminalBenchConfig: frozen config dataclass
- evaluate_terminal_bench: pure async function for evaluation
- make_terminal_bench_reflective: pure function for reflective dataset
- TerminalBenchAdapter: wrapper class (backwards compat)
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import trio

from ...agents import Actor, AgentState, RunConfig, run_agent
from ...core import (
    Endpoint,
    Message,
    StopReason,
    Trajectory,
)
from ...dtypes import StreamEvent
from ...environments.terminal_bench import TerminalBenchEnvironment, run_tests_and_score
from ..types import Candidate, EvaluationBatch

logger = logging.getLogger(__name__)


# ─── Types ────────────────────────────────────────────────────────────────────


@dataclass
class TerminalBenchTask:
    """A terminal-bench task definition."""

    task_id: str
    dataset_name: str = "terminal-bench-core"
    dataset_version: str = "head"


@dataclass(frozen=True)
class TerminalBenchConfig:
    """Configuration for terminal-bench adapter.

    Immutable config passed to pure evaluation functions.
    """

    endpoint: Endpoint
    max_turns: int = 50
    max_concurrent: int = 4
    runs_dir: Path = Path("runs")
    no_rebuild: bool = True
    cleanup: bool = True


# ─── Pure Functions ───────────────────────────────────────────────────────────


def _handle_stop_max_turns(max_turns: int) -> Callable[[AgentState], AgentState]:
    """Create a stop handler that limits turns."""
    from dataclasses import replace

    def handle_stop(state: AgentState) -> AgentState:
        turns = sum(1 for m in state.actor.trajectory.messages if m.role == "assistant")
        if turns >= max_turns:
            return replace(state, stop=StopReason.MAX_TURNS)
        return state

    return handle_stop


async def _silent_chunk_handler(event: StreamEvent) -> None:
    """Silent handler for streaming events."""
    await trio.lowlevel.checkpoint()


async def _eval_single_task(
    config: TerminalBenchConfig,
    task_data: dict,
    instruction_prompt: str,
    capture_traces: bool,
) -> tuple[str, float, dict]:
    """Evaluate a single task."""
    task_id = task_data["task_id"]
    dataset_name = task_data.get("dataset_name", "terminal-bench-core")
    dataset_version = task_data.get("dataset_version", "head")

    run_id = f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging_dir = config.runs_dir / run_id

    env = None
    try:
        # Create environment
        env = await TerminalBenchEnvironment.create(
            task_id=task_id,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            logging_dir=logging_dir,
            no_rebuild=config.no_rebuild,
            cleanup=config.cleanup,
        )

        # Build initial messages
        env_system = env.get_system_prompt() or ""
        full_system = f"{instruction_prompt}\n\n{env_system}" if instruction_prompt else env_system

        initial_messages = [
            Message(role="system", content=full_system),
            Message(role="user", content=f"Please solve this task:\n\n{env.instruction}"),
        ]

        # Create actor and initial state
        actor = Actor(
            trajectory=Trajectory(messages=initial_messages),
            endpoint=config.endpoint,
            tools=env.get_tools(),
        )
        initial_state = AgentState(actor=actor, environment=env)

        # Run agent
        run_config = RunConfig(
            on_chunk=_silent_chunk_handler,
            handle_stop=_handle_stop_max_turns(config.max_turns),
        )

        states = await run_agent(initial_state, run_config)
        final_state = states[-1] if isinstance(states, list) else states

        # Run tests and score
        score, success, failure_reason = await run_tests_and_score(env)

        # Build trajectory for GEPA
        trajectory = {
            "task_id": task_id,
            "instruction": env.instruction,
            "messages": [
                {"role": m.role, "content": str(m.content)[:2000]}
                for m in final_state.actor.trajectory.messages
            ],
            "interactions": env.interactions,
            "instruction_prompt": instruction_prompt,
            "score": score,
            "success": success,
            "failure_reason": failure_reason,
        }

        output = f"Task {task_id}: {'SUCCESS' if success else 'FAILED'} (score: {score:.2f})"
        if failure_reason:
            output += f" - {failure_reason}"

        return output, score, trajectory

    except Exception as e:
        logger.exception(f"Error evaluating task {task_id}: {e}")
        return (
            f"Error: {e}",
            0.0,
            {
                "task_id": task_id,
                "error": str(e),
                "instruction_prompt": instruction_prompt,
            },
        )
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                logger.warning(f"Error cleaning up environment: {e}")


async def evaluate_terminal_bench(
    config: TerminalBenchConfig,
    batch: Sequence[dict],
    candidate: Candidate,
    capture_traces: bool = False,
) -> EvaluationBatch:
    """Evaluate candidate on a batch of terminal-bench tasks.

    Pure async function - takes config explicitly.

    Args:
        config: TerminalBenchConfig with endpoint, max_turns, etc.
        batch: List of task dicts with "task_id" key
        candidate: Must have "instruction_prompt" key
        capture_traces: If True, capture full trajectories

    Returns:
        EvaluationBatch with outputs, scores, and optional trajectories
    """
    instruction_prompt = candidate["instruction_prompt"]

    outputs: list[str] = []
    scores: list[float] = []
    trajectories: list[dict[str, Any]] = []

    # Run with concurrency limit
    limiter = trio.CapacityLimiter(config.max_concurrent)
    results: list[tuple[str, float, dict] | None] = [None] * len(batch)

    async def eval_one(idx: int, task_data: dict) -> None:
        async with limiter:
            result = await _eval_single_task(config, task_data, instruction_prompt, capture_traces)
            results[idx] = result

    async with trio.open_nursery() as nursery:
        for idx, task_data in enumerate(batch):
            nursery.start_soon(eval_one, idx, task_data)

    # Collect results
    for result in results:
        if result is None:
            outputs.append("Error: evaluation failed")
            scores.append(0.0)
            if capture_traces:
                trajectories.append({})
        else:
            output, score, trajectory = result
            outputs.append(output)
            scores.append(score)
            if capture_traces:
                trajectories.append(trajectory)

    return EvaluationBatch(
        outputs=tuple(outputs),
        scores=tuple(scores),
        trajectories=tuple(trajectories) if capture_traces else None,
    )


def make_terminal_bench_reflective(
    candidate: Candidate,
    eval_batch: EvaluationBatch,
    components_to_update: list[str],
) -> dict[str, list[dict]]:
    """Build reflective dataset from execution traces.

    Pure function - no config needed.

    For terminal-bench, includes:
    - Task instruction
    - Full message history (tool calls, outputs)
    - Success/failure status with reason
    """
    if "instruction_prompt" not in components_to_update:
        return {}

    if eval_batch.trajectories is None:
        logger.warning("No trajectories in eval_batch")
        return {"instruction_prompt": []}

    items = []
    for trajectory in eval_batch.trajectories:
        if not trajectory:
            continue

        success = trajectory.get("success", False)
        if success:
            feedback = "Task completed successfully!"
        else:
            reason = trajectory.get("failure_reason", "Unknown error")
            feedback = f"Task failed. Reason: {reason}"

        # Format message history
        messages = trajectory.get("messages", [])
        formatted_messages = []
        for msg in messages[-20:]:  # Last 20 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 1000:
                content = content[:1000] + "... [truncated]"
            formatted_messages.append({"role": role, "content": content})

        items.append({
            "Task ID": trajectory.get("task_id", "unknown"),
            "Task Instruction": trajectory.get("instruction", "")[:500],
            "Message History": formatted_messages,
            "System Prompt Used": trajectory.get("instruction_prompt", ""),
            "Score": trajectory.get("score", 0.0),
            "Feedback": feedback,
        })

    return {"instruction_prompt": items}
