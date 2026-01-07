"""System + user prompt optimization adapter.

Optimizes both the system prompt and user template (with wildcards preserved).
Based on Synth's GEPA which optimizes multiple message patterns.

Provides:
- SystemUserPromptConfig: frozen config dataclass
- evaluate_system_user_prompt: pure async function for evaluation
- make_system_user_prompt_reflective: pure function for reflective dataset

Candidate structure:
    {"system": "...", "user": "..."}

The user template must contain wildcards like {query} that get filled from sample data.
These wildcards are preserved during optimization - only the surrounding text changes.

Example:
    from functools import partial

    config = SystemUserPromptConfig(
        endpoint=endpoint,
        wildcards=("query",),
        score_fn=score_fn,
    )

    result = await run_gepa(
        seed_candidate={
            "system": "You are a classifier.",
            "user": "Classify this query: {query}",
        },
        dataset=my_dataset,
        evaluate_fn=partial(evaluate_system_user_prompt, config),
        make_reflective_fn=partial(make_system_user_prompt_reflective, config),
        config=GEPAConfig(max_evaluations=500),
        reflection_endpoint=endpoint,
    )
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
class SystemUserPromptConfig:
    """Configuration for system + user prompt optimization.

    Both system and user prompts are optimized.
    Wildcards in the user template (e.g., {query}) are preserved.
    """

    endpoint: Endpoint
    wildcards: tuple[str, ...]  # e.g., ("query", "context") - fields from sample data
    score_fn: ScoreFn
    environment_factory: EnvironmentFactory | None = None
    max_concurrent: int = 10
    max_turns: int | None = None


# ─── Helpers ──────────────────────────────────────────────────────────────────


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
    """Format a multi-turn trajectory for reflection."""
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


def fill_user_template(user_template: str, sample: dict, wildcards: tuple[str, ...]) -> str:
    """Fill wildcards in user template from sample data."""
    result = user_template
    for wildcard in wildcards:
        if wildcard in sample:
            result = result.replace(f"{{{wildcard}}}", str(sample[wildcard]))
    return result


def make_prepare_messages(
    system_prompt: str, user_template: str, wildcards: tuple[str, ...]
) -> Callable[[dict], list[Message]]:
    """Create prepare_messages function for EvalConfig."""

    def prepare_messages(sample: dict) -> list[Message]:
        user_content = fill_user_template(user_template, sample, wildcards)
        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_content),
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


async def _silent_chunk_handler(_: StreamEvent) -> None:
    """Silent handler for streaming events."""
    await trio.lowlevel.checkpoint()


async def _stop_after_response(state: AgentState, run_config: RunConfig) -> AgentState:
    """Stop after first response."""
    from dataclasses import replace

    return replace(state, stop=StopReason.TASK_COMPLETED)


async def _default_no_tool_handler(state: AgentState, run_config: RunConfig) -> AgentState:
    """Stop when agent produces no tool call."""
    from dataclasses import replace

    return replace(state, stop=StopReason.TASK_COMPLETED)


# ─── Pure Functions ───────────────────────────────────────────────────────────


async def evaluate_system_user_prompt(
    config: SystemUserPromptConfig,
    batch: Sequence[dict],
    candidate: Candidate,
    capture_traces: bool = False,
) -> EvaluationBatch:
    """Evaluate system+user candidate on batch.

    Pure async function - takes config explicitly.

    Args:
        config: SystemUserPromptConfig with endpoint, wildcards, score_fn, etc.
        batch: List of sample dicts
        candidate: Must have keys "system" and "user"
        capture_traces: If True, include execution traces

    Returns:
        EvaluationBatch with outputs, scores, and optional traces
    """
    system_prompt = candidate["system"]
    user_template = candidate["user"]

    run_config = build_run_config(config.max_turns)

    eval_config = EvalConfig(
        endpoint=config.endpoint,
        score_fn=config.score_fn,
        prepare_messages=make_prepare_messages(system_prompt, user_template, config.wildcards),
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
                "user_template": user_template,
            }
            for s in samples
        )

    return EvaluationBatch(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories,
    )


def make_system_user_prompt_reflective(
    config: SystemUserPromptConfig,
    candidate: Candidate,
    eval_batch: EvaluationBatch,
    components_to_update: list[str],
) -> dict[str, list[dict]]:
    """Extract feedback for system and/or user prompt from traces.

    Pure function.

    Args:
        config: SystemUserPromptConfig (for max_turns check)
        candidate: Current candidate
        eval_batch: Evaluation with trajectories
        components_to_update: Should include "system" and/or "user"

    Returns:
        Dict with "system" and/or "user" keys containing feedback items
    """
    if eval_batch.trajectories is None:
        logger.warning("No trajectories in eval_batch")
        return {c: [] for c in components_to_update if c in ("system", "user")}

    result: dict[str, list[dict]] = {}

    for component in components_to_update:
        if component not in ("system", "user"):
            continue

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

        result[component] = items

    return result
