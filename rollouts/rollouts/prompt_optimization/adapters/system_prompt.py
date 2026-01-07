"""System prompt optimization adapter.

Optimizes the system prompt while keeping the user template fixed.
For optimizing both system and user prompts, see system_user_prompt.py.

Supports both:
- Single-turn evaluation (no tools, stops after first response)
- Multi-turn tool-using agents (runs until agent stops or hits max_turns)
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import trio

from ...agents import handle_stop_max_turns
from ...dtypes import (
    AgentState,
    Endpoint,
    EvalConfig,
    Message,
    RunConfig,
    Score,
    StopReason,
    StreamEvent,
)
from ...evaluation import EvalRuntime, evaluate_sample
from ...training.types import Sample
from ..types import Candidate, EvaluationBatch

logger = logging.getLogger(__name__)

# Type aliases
ScoreFn = Callable[[Sample], Score] | Callable[[Sample], Awaitable[Score]]
EnvironmentFactory = Callable[[dict[str, Any]], Awaitable[Any]]


# ─── Config ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SystemPromptConfig:
    """Configuration for system prompt optimization.

    Immutable config that gets passed to pure evaluation functions.
    The user_template is fixed; only the system prompt is optimized.
    """

    endpoint: Endpoint
    user_template: str
    score_fn: ScoreFn
    environment_factory: EnvironmentFactory | None = None
    max_concurrent: int = 10
    max_turns: int | None = None  # None = single-turn, int = multi-turn with tools


# ─── Pure Functions ───────────────────────────────────────────────────────────


def extract_output(sample: Sample) -> str:
    """Extract output text from Sample's trajectory."""
    if not sample.trajectory or not sample.trajectory.messages:
        return ""

    for msg in reversed(sample.trajectory.messages):
        if msg.role == "assistant":
            if isinstance(msg.content, str):
                return msg.content
            if isinstance(msg.content, list):
                parts = []
                for block in msg.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                return "".join(parts)
    return ""


def format_trajectory(messages: list[Message]) -> str:
    """Format a multi-turn trajectory for reflection.

    Includes tool calls and their results.
    """
    parts = []
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "user":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            parts.append(f"User: {content}")
        elif msg.role == "assistant":
            if isinstance(msg.content, str):
                parts.append(f"Assistant: {msg.content}")
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text:
                        parts.append(f"Assistant: {block.text}")
                    elif hasattr(block, "type") and block.type == "tool_use":
                        tool_name = getattr(block, "name", "unknown")
                        tool_input = getattr(block, "input", {})
                        parts.append(f"Assistant [tool_call]: {tool_name}({tool_input})")
        elif msg.role == "tool_result":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            parts.append(f"Tool Result: {content}")

    return "\n".join(parts)


def make_prepare_messages(
    system_prompt: str, user_template: str
) -> Callable[[dict], list[Message]]:
    """Create prepare_messages function for EvalConfig."""

    def prepare_messages(sample: dict) -> list[Message]:
        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_template.format(**sample)),
        ]

    return prepare_messages


def build_run_config(max_turns: int | None) -> RunConfig:
    """Build RunConfig based on single-turn vs multi-turn mode."""
    if max_turns is not None:
        return RunConfig(
            on_chunk=_silent_chunk_handler,
            handle_stop=handle_stop_max_turns(max_turns),
            handle_no_tool=_default_no_tool_handler,
        )
    return RunConfig(
        on_chunk=_silent_chunk_handler,
        handle_no_tool=_stop_after_response,
    )


async def evaluate_system_prompt(
    config: SystemPromptConfig,
    batch: Sequence[dict],
    candidate: Candidate,
    capture_traces: bool = False,
) -> EvaluationBatch:
    """Evaluate single-prompt candidate on batch.

    Delegates to rollouts/evaluation.evaluate_sample for each sample,
    with concurrency control.

    Args:
        config: Adapter configuration
        batch: List of sample dicts
        candidate: Must have key "system" with system prompt
        capture_traces: If True, include execution traces

    Returns:
        EvaluationBatch with outputs, scores, and optional traces
    """
    system_prompt = candidate["system"]

    run_config = build_run_config(config.max_turns)

    eval_config = EvalConfig(
        endpoint=config.endpoint,
        score_fn=config.score_fn,
        prepare_messages=make_prepare_messages(system_prompt, config.user_template),
        environment_factory=config.environment_factory,
        run_config=run_config,
        max_concurrent=config.max_concurrent,
    )

    # Create runtime context for evaluate_sample calls
    runtime = EvalRuntime(config=eval_config)

    async def eval_one(idx: int, sample_data: dict) -> Sample:
        env = await config.environment_factory(sample_data) if config.environment_factory else None
        return await evaluate_sample(
            sample_data=sample_data,
            sample_id=f"gepa_{idx}",
            runtime=runtime,
            environment=env,
        )

    # Run with concurrency limit
    async with trio.open_nursery() as nursery:
        limiter = trio.CapacityLimiter(config.max_concurrent)
        results: list[Sample | None] = [None] * len(batch)

        async def eval_with_limit(idx: int, sample_data: dict) -> None:
            async with limiter:
                results[idx] = await eval_one(idx, sample_data)

        for idx, sample_data in enumerate(batch):
            nursery.start_soon(eval_with_limit, idx, sample_data)

    samples = [r for r in results if r is not None]

    outputs = tuple(extract_output(s) for s in samples)
    scores = tuple(s.score.reward if s.score else 0.0 for s in samples)

    trajectories = None
    if capture_traces:
        trajectories = tuple(
            {
                "sample": s.input,
                "messages": s.trajectory.messages if s.trajectory else [],
                "output": extract_output(s),
                "score": s.score.reward if s.score else 0.0,
                "ground_truth": s.ground_truth,
            }
            for s in samples
        )

    return EvaluationBatch(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories,
    )


def make_system_prompt_reflective(
    candidate: Candidate,
    eval_batch: EvaluationBatch,
    components_to_update: list[str],
) -> dict[str, list[dict]]:
    """Extract feedback for system prompt from traces.

    For tool-using agents, includes the full trajectory with tool calls.

    Args:
        config: Adapter configuration
        candidate: Current candidate (unused but kept for protocol)
        eval_batch: Evaluation with trajectories
        components_to_update: Should include "system"

    Returns:
        Dict with "system" key containing feedback items
    """
    if "system" not in components_to_update:
        return {}

    if eval_batch.trajectories is None:
        logger.warning("No trajectories in eval_batch, cannot make reflective dataset")
        return {"system": []}

    items = []
    for trace in eval_batch.trajectories:
        score = trace["score"]
        ground_truth = trace.get("ground_truth")

        if score >= 0.9:
            feedback = "Excellent response. This is correct."
        elif score >= 0.5:
            feedback = f"Partially correct. Expected: {ground_truth}"
        else:
            feedback = f"Incorrect. Expected: {ground_truth}"

        input_text = ""
        for msg in trace["messages"]:
            if msg.role == "user":
                input_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        # Check if multi-turn by counting assistant messages
        assistant_msgs = [m for m in trace["messages"] if m.role == "assistant"]
        is_multi_turn = len(assistant_msgs) > 1

        if is_multi_turn:
            trajectory_text = format_trajectory(trace["messages"])
            items.append({
                "Inputs": input_text,
                "Trajectory": trajectory_text,
                "Final Output": trace["output"],
                "Feedback": feedback,
            })
        else:
            items.append({
                "Inputs": input_text,
                "Generated Outputs": trace["output"],
                "Feedback": feedback,
            })

    return {"system": items}


# ─── Handlers ─────────────────────────────────────────────────────────────────


async def _silent_chunk_handler(_: StreamEvent) -> None:
    """Silent handler for streaming events."""
    await trio.lowlevel.checkpoint()


async def _stop_after_response(state: AgentState, run_config: RunConfig) -> AgentState:
    """Stop after first response - for simple evaluation without tools."""
    from dataclasses import replace

    return replace(state, stop=StopReason.TASK_COMPLETED)


async def _default_no_tool_handler(state: AgentState, run_config: RunConfig) -> AgentState:
    """Default handler when agent produces no tool call - stop the agent."""
    from dataclasses import replace

    return replace(state, stop=StopReason.TASK_COMPLETED)
