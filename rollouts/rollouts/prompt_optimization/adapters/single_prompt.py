"""SinglePromptAdapter - adapter for single system prompt optimization.

The most common use case: optimize just the system prompt.
Delegates to rollouts/evaluation.py for actual evaluation.

Supports both:
- Single-turn evaluation (no tools, stops after first response)
- Multi-turn tool-using agents (runs until agent stops or hits max_turns)
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
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
from ...evaluation import evaluate_sample
from ...training.types import Sample
from ..types import Candidate, EvaluationBatch

logger = logging.getLogger(__name__)

# Type aliases
ScoreFn = Callable[[Sample], Score] | Callable[[Sample], Awaitable[Score]]
EnvironmentFactory = Callable[[dict[str, Any]], Awaitable[Any]]


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


# TODO: Refactor to dataclass + pure functions. This class has no mutable state,
# just bundles config with methods. Would be cleaner as:
#   SinglePromptConfig (dataclass) + evaluate(config, batch, candidate) + make_reflective_dataset(...)
class SinglePromptAdapter:
    """Adapter for single system prompt optimization.

    Implements GEPAAdapter protocol for the common case of
    optimizing just a system prompt.

    The candidate is expected to have a single key "system"
    containing the system prompt text.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        user_template: str,
        score_fn: ScoreFn,
        environment_factory: EnvironmentFactory | None = None,
        max_concurrent: int = 10,
        max_turns: int | None = None,
    ) -> None:
        """Initialize adapter.

        Args:
            endpoint: LLM endpoint for task evaluation
            user_template: Template for user messages with {placeholders}
            score_fn: Function to compute score from Sample
            environment_factory: Optional factory for tool-using agents
            max_concurrent: Maximum parallel evaluations (default 10)
            max_turns: Max agent turns for tool-using agents. If None, stops after
                first response (single-turn mode). If set, runs multi-turn with tools.
        """
        self.endpoint = endpoint
        self.user_template = user_template
        self.score_fn = score_fn
        self.environment_factory = environment_factory
        self.max_concurrent = max_concurrent
        self.max_turns = max_turns

    def _make_prepare_messages(self, system_prompt: str) -> Callable[[dict], list[Message]]:
        """Create prepare_messages function for EvalConfig."""
        user_template = self.user_template

        def prepare_messages(sample: dict) -> list[Message]:
            return [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_template.format(**sample)),
            ]

        return prepare_messages

    async def evaluate(
        self,
        batch: Sequence[dict],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate single-prompt candidate on batch.

        Delegates to rollouts/evaluation.evaluate_sample for each sample,
        with concurrency control.

        Args:
            batch: List of sample dicts
            candidate: Must have key "system" with system prompt
            capture_traces: If True, include execution traces

        Returns:
            EvaluationBatch with outputs, scores, and optional traces
        """
        system_prompt = candidate["system"]

        # Build RunConfig based on whether we're doing single-turn or multi-turn
        if self.max_turns is not None:
            # Multi-turn mode: run until agent stops or hits max_turns
            run_config = RunConfig(
                on_chunk=_silent_chunk_handler,
                handle_stop=handle_stop_max_turns(self.max_turns),
                handle_no_tool=_default_no_tool_handler,
            )
        else:
            # Single-turn mode: stop after first response
            run_config = RunConfig(
                on_chunk=_silent_chunk_handler,
                handle_no_tool=_stop_after_response,
            )

        config = EvalConfig(
            endpoint=self.endpoint,
            score_fn=self.score_fn,
            prepare_messages=self._make_prepare_messages(system_prompt),
            environment_factory=self.environment_factory,
            run_config=run_config,
            max_concurrent=self.max_concurrent,
        )

        # Evaluate samples with concurrency
        samples: list[Sample] = []

        async def eval_one(idx: int, sample_data: dict) -> Sample:
            env = await self.environment_factory(sample_data) if self.environment_factory else None
            return await evaluate_sample(
                sample_data=sample_data,
                sample_id=f"gepa_{idx}",
                config=config,
                environment=env,
            )

        # Run with concurrency limit
        async with trio.open_nursery() as nursery:
            limiter = trio.CapacityLimiter(self.max_concurrent)
            results: list[Sample | None] = [None] * len(batch)

            async def eval_with_limit(idx: int, sample_data: dict) -> None:
                async with limiter:
                    results[idx] = await eval_one(idx, sample_data)

            for idx, sample_data in enumerate(batch):
                nursery.start_soon(eval_with_limit, idx, sample_data)

        samples = [r for r in results if r is not None]

        # Convert Sample list -> EvaluationBatch
        outputs = tuple(self._extract_output(s) for s in samples)
        scores = tuple(s.score.reward if s.score else 0.0 for s in samples)

        trajectories = None
        if capture_traces:
            # Build trace dicts for make_reflective_dataset
            trajectories = tuple(
                {
                    "sample": s.input,
                    "messages": s.trajectory.messages if s.trajectory else [],
                    "output": self._extract_output(s),
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

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """Extract feedback for system prompt from traces.

        For tool-using agents, includes the full trajectory with tool calls.

        Args:
            candidate: Current candidate
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

            # Build feedback based on score
            if score >= 0.9:
                feedback = "Excellent response. This is correct."
            elif score >= 0.5:
                feedback = f"Partially correct. Expected: {ground_truth}"
            else:
                feedback = f"Incorrect. Expected: {ground_truth}"

            # Get input (first user message)
            input_text = ""
            for msg in trace["messages"]:
                if msg.role == "user":
                    input_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

            # For multi-turn, include full trajectory
            if self.max_turns is not None:
                trajectory_text = self._format_trajectory(trace["messages"])
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

    def _format_trajectory(self, messages: list[Message]) -> str:
        """Format a multi-turn trajectory for reflection.

        Includes tool calls and their results.
        """
        parts = []
        for msg in messages:
            if msg.role == "system":
                continue  # Skip system message
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
                # Truncate long tool results
                if len(content) > 500:
                    content = content[:500] + "... [truncated]"
                parts.append(f"Tool Result: {content}")

        return "\n".join(parts)

    def _extract_output(self, sample: Sample) -> str:
        """Extract output text from Sample's trajectory."""
        if not sample.trajectory or not sample.trajectory.messages:
            return ""

        # Get last assistant message
        for msg in reversed(sample.trajectory.messages):
            if msg.role == "assistant":
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    parts = []
                    for block in msg.content:
                        if hasattr(block, "text"):
                            parts.append(block.text)
                    return "".join(parts)
        return ""
