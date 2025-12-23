"""SinglePromptAdapter - adapter for single system prompt optimization.

The most common use case: optimize just the system prompt.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import trio

from rollouts.agents import run_agent
from rollouts.dtypes import (
    Actor,
    AgentState,
    Endpoint,
    Message,
    RunConfig,
    Score,
    StreamEvent,
    Trajectory,
)
from rollouts.training.types import Sample

from ..types import Candidate, EvaluationBatch

logger = logging.getLogger(__name__)

# Type aliases
ScoreFn = Callable[[Sample], Score] | Callable[[Sample], Awaitable[Score]]
EnvironmentFactory = Callable[[dict[str, Any]], Awaitable[Any]]


async def _silent_chunk_handler(_: StreamEvent) -> None:
    """Silent handler for streaming events."""
    await trio.lowlevel.checkpoint()


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
    ) -> None:
        """Initialize adapter.

        Args:
            endpoint: LLM endpoint for task evaluation
            user_template: Template for user messages with {placeholders}
            score_fn: Function to compute score from Sample
            environment_factory: Optional factory for tool-using agents
        """
        self.endpoint = endpoint
        self.user_template = user_template
        self.score_fn = score_fn
        self.environment_factory = environment_factory

    async def evaluate(
        self,
        batch: Sequence[dict],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate single-prompt candidate on batch.

        Args:
            batch: List of sample dicts
            candidate: Must have key "system" with system prompt
            capture_traces: If True, include execution traces

        Returns:
            EvaluationBatch with outputs, scores, and optional traces
        """
        system_prompt = candidate["system"]

        outputs: list[Any] = []
        scores: list[float] = []
        trajectories: list[dict] = []

        for sample in batch:
            # Format messages
            user_content = self.user_template.format(**sample)
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_content),
            ]

            # Build trajectory
            trajectory = Trajectory(messages=messages)

            # Build actor
            env = None
            tools = []
            if self.environment_factory:
                env = await self.environment_factory(sample)
                tools = env.get_tools()

            actor = Actor(trajectory=trajectory, endpoint=self.endpoint, tools=tools)
            state = AgentState(actor=actor, environment=env)
            run_config = RunConfig(on_chunk=_silent_chunk_handler)

            # Run agent
            try:
                states = await run_agent(state, run_config)
                final_trajectory = states[-1].actor.trajectory
            except Exception as e:
                logger.warning(f"Evaluation failed for sample: {e}")
                outputs.append("")
                scores.append(0.0)
                if capture_traces:
                    trajectories.append({
                        "sample": sample,
                        "messages": messages,
                        "output": "",
                        "score": 0.0,
                        "error": str(e),
                    })
                continue

            # Extract output
            output = self._extract_output(final_trajectory)
            outputs.append(output)

            # Score
            eval_sample = Sample(
                id=str(len(outputs)),
                input=sample,
                trajectory=final_trajectory,
                ground_truth=sample.get("answer") or sample.get("label"),
            )

            # Support both sync and async score functions
            import inspect

            score_result = self.score_fn(eval_sample)
            if inspect.iscoroutine(score_result):
                score = await score_result
            else:
                score = score_result

            scores.append(score.reward)

            if capture_traces:
                trajectories.append({
                    "sample": sample,
                    "messages": messages,
                    "output": output,
                    "score": score.reward,
                    "ground_truth": sample.get("answer") or sample.get("label"),
                })

        return EvaluationBatch(
            outputs=tuple(outputs),
            scores=tuple(scores),
            trajectories=tuple(trajectories) if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """Extract feedback for system prompt from traces.

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

            # Get input (user message)
            input_text = ""
            for msg in trace["messages"]:
                if msg.role == "user":
                    input_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

            items.append({
                "Inputs": input_text,
                "Generated Outputs": trace["output"],
                "Feedback": feedback,
            })

        return {"system": items}

    def _extract_output(self, trajectory: Trajectory) -> str:
        """Extract output text from trajectory."""
        if not trajectory.messages:
            return ""

        # Get last assistant message
        for msg in reversed(trajectory.messages):
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
