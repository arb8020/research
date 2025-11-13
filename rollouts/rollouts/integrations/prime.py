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
import logging
import trio_asyncio
from dataclasses import replace
from typing import Any, Dict, List, Tuple

from verifiers import Environment as VerifiersEnv
from verifiers import Parser, Rubric

from ..dtypes import Trajectory, RewardFunction, Message

logger = logging.getLogger(__name__)


def _extract_model_response(trajectory: Trajectory) -> str:
    """Extract last assistant message from trajectory.

    Tiger Style: Pure function, push fors down.
    """
    assert trajectory is not None
    assert trajectory.messages is not None

    for msg in reversed(trajectory.messages):
        if msg.role == "assistant":
            content = msg.content
            # Prime environments don't use vision messages
            assert isinstance(content, (str, type(None))), \
                f"Prime integration expects string content, got {type(content)}"
            return content or ""

    return ""


def _convert_to_prime_format(trajectory: Trajectory) -> Tuple[List[Dict], List[Dict]]:
    """Convert rollouts messages to Prime's OpenAI format.

    Tiger Style: Pure function, push fors down.
    Returns (prompt, completion) tuple.
    """
    assert trajectory is not None
    assert trajectory.messages is not None

    prompt = []
    completion = []

    for msg in trajectory.messages:
        oai_msg = {"role": msg.role, "content": msg.content or ""}
        if msg.role == "assistant":
            completion.append(oai_msg)
        else:
            prompt.append(oai_msg)

    assert len(prompt) > 0  # Must have at least system or user message
    return prompt, completion


async def _call_prime_scoring(
    env: VerifiersEnv,
    rubric: Rubric,
    prompt: List[Dict],
    completion: List[Dict],
    ground_truth: str | None,
    sample_data: Dict[str, Any],
    backend_bench_state: Dict[str, Any] | None = None
) -> Any:
    """Call Prime's asyncio scoring from trio context.

    Tiger Style: Isolate trio-asyncio bridge in single function.

    Args:
        ground_truth: Can be None for environments that don't use reference answers
        backend_bench_state: Pre-computed state for stateful environments (e.g., backend-bench)
    """
    assert env is not None
    assert rubric is not None
    assert prompt is not None
    assert completion is not None

    async with trio_asyncio.open_loop():
        # Check if stateful environment already provided updated state
        # (e.g., backend-bench stores state with test results in metadata)
        if backend_bench_state is not None:
            state = backend_bench_state
        else:
            # No pre-computed state - initialize fresh (for stateless environments)
            state = await trio_asyncio.aio_as_trio(env.init_state)(
                prompt=prompt,
                completion=completion,
                answer=ground_truth or "",  # Use empty string if no ground truth
                task="default",
                info=sample_data,
                example_id=sample_data.get("example_id", 0)
            )

        # Score the rollout
        score_result = await trio_asyncio.aio_as_trio(rubric.score_rollout)(
            prompt=prompt,
            completion=completion,
            answer=ground_truth or "",  # Use empty string if no ground truth
            state=state,
            info=sample_data,
            example_id=sample_data.get("example_id", 0)
        )

    assert score_result is not None
    assert hasattr(score_result, 'reward')
    return score_result


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

        Tiger Style: Keep control flow in parent, delegate work to helpers.
        """
        # Preconditions
        assert trajectory is not None
        assert trajectory.metadata is not None

        # Get sample data (injected by evaluate_sample)
        sample_data = trajectory.metadata.get("sample_data", {})
        ground_truth = sample_data.get(ground_truth_key)
        # Note: ground_truth can be None for environments like backend-bench
        # that test code execution rather than comparing against reference answers

        # Check if stateful environment stored updated state (e.g., backend-bench)
        backend_bench_state = trajectory.metadata.get("backend_bench_state")
        if backend_bench_state is not None:
            logger.info(f"✅ Using stateful environment state with keys: {list(backend_bench_state.keys())}")
            logger.info(f"   State has 'results': {'results' in backend_bench_state}")
            logger.info(f"   State has 'best_result': {'best_result' in backend_bench_state}")

        # Extract model response (push for down to helper)
        model_response = _extract_model_response(trajectory)

        # Convert to Prime format for parsing
        # Some parsers (like backend-bench) expect list of messages, not just string
        _, completion_messages = _convert_to_prime_format(trajectory)

        # Parse response
        try:
            # Try passing full completion messages first (for env-specific parsers like backend-bench)
            # Fall back to string if that fails
            try:
                parsed_answer = _parser.parse(completion_messages)
            except (TypeError, KeyError):
                # Parser expects string, not list
                parsed_answer = _parser.parse(model_response)
        except Exception as e:
            # Parse error - log loudly and return 0 reward
            logger.error(f"❌ PARSE ERROR: {e}")
            logger.error(f"   Parser type: {type(_parser).__name__}")
            logger.error(f"   Response preview: {model_response[:200]}...")
            metadata = {
                **trajectory.metadata,
                "prime_parse_error": str(e),
                "prime_raw_response": model_response,
            }
            return replace(trajectory, rewards=0.0, metadata=metadata)

        # Convert format (push for down to helper)
        prompt, completion = _convert_to_prime_format(trajectory)

        # Score with Prime (isolate async bridge in helper)
        try:
            score_result = await _call_prime_scoring(
                _env, _rubric, prompt, completion, ground_truth, sample_data,
                backend_bench_state=backend_bench_state
            )
        except Exception as e:
            # Scoring error - log loudly and return 0 reward
            logger.error(f"❌ GRADING ERROR: {e}")
            logger.error(f"   Rubric type: {type(_rubric).__name__}")
            logger.error(f"   Parsed answer: {parsed_answer}")
            if trajectory.metadata.get("verbose"):
                import traceback
                logger.error(f"   Full traceback:\n{traceback.format_exc()}")
            metadata = {
                **trajectory.metadata,
                "prime_grade_error": str(e),
                "prime_parsed_answer": str(parsed_answer),
                "prime_ground_truth": str(ground_truth) if ground_truth is not None else None,
            }
            return replace(trajectory, rewards=0.0, metadata=metadata)

        # Extract score
        reward_score = float(score_result.reward)
        assert 0.0 <= reward_score <= 1.0, f"Invalid reward: {reward_score}"

        # Serialize metrics to avoid unpicklable objects (e.g., thread locks)
        # Convert Pydantic model to plain dict for safe serialization
        serialized_metrics = {}
        try:
            # Try model_dump() for Pydantic v2
            dumped = score_result.model_dump()
            if 'metrics' in dumped:
                serialized_metrics = dumped['metrics']
            else:
                # No 'metrics' key, use all fields except known internal ones
                serialized_metrics = {k: v for k, v in dumped.items()
                                     if k not in ('reward',) and isinstance(v, (int, float, str, bool, type(None)))}
        except (AttributeError, KeyError, TypeError) as e:
            # Fallback: try direct dict conversion
            try:
                if hasattr(score_result, 'metrics') and score_result.metrics:
                    serialized_metrics = dict(score_result.metrics)
                elif hasattr(score_result, '__dict__'):
                    # Last resort: extract numeric fields from __dict__
                    serialized_metrics = {k: v for k, v in score_result.__dict__.items()
                                         if k not in ('reward',) and isinstance(v, (int, float, str, bool, type(None)))}
            except Exception as inner_e:
                logger.warning(f"Failed to serialize Prime metrics: {inner_e}")
                serialized_metrics = {}

        # Build metadata
        metadata = {
            **trajectory.metadata,
            "prime_parsed_answer": str(parsed_answer),
            "prime_ground_truth": str(ground_truth) if ground_truth is not None else None,
            "prime_raw_response": model_response,
            "prime_metrics": serialized_metrics
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
