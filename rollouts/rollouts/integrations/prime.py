"""Prime Intellect verifiers integration.

Adapter to use Prime Intellect environments and rubrics with rollouts framework.

Example:
    >>> from verifiers import SingleTurnEnv, Rubric, Parser
    >>> from rollouts.integrations.prime import prime_reward_fn
    >>>
    >>> # Create verifiers environment
    >>> verifiers_env = SingleTurnEnv(
    ...     dataset=my_dataset,
    ...     rubric=Rubric(),
    ...     parser=Parser(),
    ... )
    >>>
    >>> # Create reward function for rollouts
    >>> reward_fn = prime_reward_fn(verifiers_env)
    >>>
    >>> # Use in EvalConfig
    >>> config = EvalConfig(reward_fn=reward_fn, ...)
"""

import asyncio
import trio_asyncio
from dataclasses import replace
from typing import Any, Dict

from verifiers import Environment as VerifiersEnv
from verifiers import Parser, Rubric

from ..dtypes import Trajectory, RewardFunction


def prime_reward_fn(
    verifiers_env: VerifiersEnv,
    rubric: Rubric | None = None,
    parser: Parser | None = None,
    ground_truth_key: str = "answer",
) -> RewardFunction:
    """Create a reward function from Prime Intellect verifiers environment.

    This factory function creates a RewardFunction (Trajectory -> Trajectory)
    that uses Prime's rubric and parser to score model responses.

    Args:
        verifiers_env: Prime Intellect environment (has rubric/parser if None provided)
        rubric: Optional rubric override (uses verifiers_env.rubric if None)
        parser: Optional parser override (uses verifiers_env.parser if None)
        ground_truth_key: Key in sample_data to extract ground truth

    Returns:
        RewardFunction compatible with rollouts.evaluation

    Example:
        >>> from verifiers import SingleTurnEnv, Rubric
        >>> verifiers_env = SingleTurnEnv(dataset=ds, rubric=Rubric())
        >>> reward_fn = prime_reward_fn(verifiers_env)
        >>> config = EvalConfig(reward_fn=reward_fn)
    """
    # Use environment's rubric/parser if not explicitly provided
    _rubric = rubric or verifiers_env.rubric
    _parser = parser or verifiers_env.parser
    _env = verifiers_env

    async def reward(trajectory: Trajectory) -> Trajectory:
        """Compute reward using Prime rubric.

        Extracts answer from trajectory, parses it, grades against ground truth.
        Stores breakdown in metadata for debugging.
        """
        # Get sample data (injected by evaluate_sample)
        sample_data = trajectory.metadata.get("sample_data", {})
        ground_truth = sample_data.get(ground_truth_key)

        # Extract model's response (last assistant message)
        model_response = ""
        for msg in reversed(trajectory.messages):
            if msg.role == "assistant":
                content = msg.content
                # Prime environments don't use vision messages - content should be string
                assert isinstance(content, (str, type(None))), \
                    f"Prime integration expects string content, got {type(content)}. Vision messages not supported."
                model_response = content or ""
                break

        # Parse response using Prime's parser
        try:
            parsed_answer = _parser.parse(model_response)
        except Exception as e:
            # Parse error - return 0 reward
            metadata = {
                **trajectory.metadata,
                "prime_parse_error": str(e),
                "prime_raw_response": model_response,
            }
            return replace(trajectory, rewards=0.0, metadata=metadata)

        # Grade using Prime's rubric with trio-asyncio bridge
        try:
            # Convert trajectory messages to Prime's OpenAI format
            prompt = []
            completion = []
            for msg in trajectory.messages:
                oai_msg = {"role": msg.role, "content": msg.content or ""}
                if msg.role == "assistant":
                    completion.append(oai_msg)
                else:
                    prompt.append(oai_msg)

            # Call Prime's asyncio functions from trio context
            # Both init_state and score_rollout are asyncio functions
            async with trio_asyncio.open_loop():
                # Initialize state with required structure
                state = await trio_asyncio.aio_as_trio(
                    _env.init_state
                )(
                    prompt=prompt,
                    completion=completion,
                    answer=ground_truth,
                    task="default",
                    info=sample_data,
                    example_id=sample_data.get("example_id", 0)
                )

                # Call Prime's scoring
                score_result = await trio_asyncio.aio_as_trio(
                    _rubric.score_rollout
                )(
                    prompt=prompt,
                    completion=completion,
                    answer=ground_truth,
                    state=state,
                    info=sample_data,
                    example_id=sample_data.get("example_id", 0)
                )

            # Extract reward and metrics from Prime's result
            reward_score = float(score_result.reward)
            extra_metrics = {
                "prime_metrics": score_result.metrics
            }

        except Exception as e:
            # Grading error - return 0 reward
            metadata = {
                **trajectory.metadata,
                "prime_grade_error": str(e),
                "prime_parsed_answer": str(parsed_answer),
                "prime_ground_truth": str(ground_truth),
            }
            return replace(trajectory, rewards=0.0, metadata=metadata)

        # Store Prime-specific metadata for debugging
        metadata = {
            **trajectory.metadata,
            "prime_parsed_answer": str(parsed_answer),
            "prime_ground_truth": str(ground_truth),
            "prime_raw_response": model_response,
            **extra_metrics,
        }

        return replace(trajectory, rewards=reward_score, metadata=metadata)

    return reward


def create_composite_prime_reward(
    verifiers_env: VerifiersEnv,
    additional_signals: Dict[str, tuple[RewardFunction, float]] | None = None,
    ground_truth_key: str = "answer",
) -> RewardFunction:
    """Create a composite reward combining Prime rubric with additional signals.

    This shows how to compose Prime's reward with custom reward signals.

    Args:
        verifiers_env: Prime Intellect environment
        additional_signals: Dict mapping signal_name -> (reward_fn, weight)
        ground_truth_key: Key for ground truth in sample_data

    Returns:
        Composite reward function

    Example:
        >>> def efficiency_signal(t: Trajectory) -> Trajectory:
        ...     turns = len([m for m in t.messages if m.role == "assistant"])
        ...     score = max(0, 1.0 - (turns - 3) * 0.1)
        ...     return replace(t, rewards=score)
        >>>
        >>> reward_fn = create_composite_prime_reward(
        ...     verifiers_env=env,
        ...     additional_signals={
        ...         "efficiency": (efficiency_signal, 0.2),
        ...     }
        ... )
    """
    # Get Prime reward function
    prime_fn = prime_reward_fn(verifiers_env, ground_truth_key=ground_truth_key)

    def composite_reward(trajectory: Trajectory) -> Trajectory:
        """Combine Prime reward with additional signals."""

        # Compute Prime reward
        prime_traj = prime_fn(trajectory)
        prime_score = prime_traj.rewards

        # Start with Prime score (weight 1.0 implicitly)
        total_reward = prime_score
        breakdown = {"prime_rubric": prime_score}

        # Add additional signals if provided
        if additional_signals:
            for name, (signal_fn, weight) in additional_signals.items():
                signal_traj = signal_fn(trajectory)
                signal_score = signal_traj.rewards
                weighted_score = signal_score * weight

                total_reward += weighted_score
                breakdown[name] = signal_score
                breakdown[f"{name}_weight"] = weight
                breakdown[f"{name}_weighted"] = weighted_score

        # Store breakdown in metadata
        metadata = {
            **prime_traj.metadata,  # Preserve Prime metadata
            "reward_breakdown": breakdown,
        }

        return replace(trajectory, rewards=total_reward, metadata=metadata)

    return composite_reward
