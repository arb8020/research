"""Agent Framework -> Rollout Training Integration

Bridge between rollouts.agents (multi-turn execution) and rollouts.training (RL).

Design pattern:
- User provides Environment class
- We run_agent() to get trajectory
- Convert trajectory -> AttemptRow with attached TrainingSample
- Return AttemptRow ready for training

Tiger Style: Pure functions, explicit transformations, all parameters visible.
Casey Muratori: Both high-level (coarse) and low-level (fine) APIs.
"""

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import trio

from ..agents import Actor, AgentState, RunConfig, handle_stop_max_turns, run_agent
from ..dtypes import (
    Endpoint,
    Message,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    Trajectory,
)
from ..training.types import AttemptRow, ProblemRow, Status, TrainingSample

if TYPE_CHECKING:
    from ..dtypes import Environment


def _content_to_str(content: str | list | None) -> str:
    """Convert message content to string for training.

    Handles:
    - str: return as-is
    - list[ContentBlock]: extract text from TextContent/ThinkingContent blocks
    - list[dict]: extract text from dict-based content blocks
    - None: return empty string

    Tiger Style: Handle all content types explicitly.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            # Handle dataclass ContentBlock types
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ThinkingContent):
                text_parts.append(f"<thinking>{block.thinking}</thinking>")
            elif isinstance(block, ToolCallContent):
                # Tool calls don't contribute text content
                pass
            # Handle dict-based content blocks (from JSON deserialization)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    text_parts.append(f"<thinking>{block.get('thinking', '')}</thinking>")
                elif "text" in block:
                    text_parts.append(block["text"])
        return " ".join(text_parts) if text_parts else ""
    else:
        return ""


# ──────────────────────── High-Level API (Coarse-Grained) ────────────────────


async def agent_rollout_to_sample(
    prompt: str | list[dict[str, str]],
    environment_cls: "Callable[[], Environment] | type[Environment] | None",
    endpoint: Endpoint,
    tokenizer: Any,  # HuggingFace tokenizer
    max_turns: int = 10,
    metadata: dict[str, Any] | None = None,
    environment_factory: Callable[[dict[str, Any]], Any] | None = None,
    sample_data: dict[str, Any] | None = None,
) -> AttemptRow:
    """Single agent rollout: prompt → multi-turn execution → training sample.

    Based on clicker/run_rollouts.py:46-120 pattern.

    For token-level generation (TI/TO), use the dedicated TI/TO path in grpo.py
    which calls rollout_sglang_token_level() directly.

    Args:
        prompt: Either a string (becomes user message) or list of message dicts
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        environment_cls: Zero-arg environment constructor for simple cases.
            Kept for backwards compatibility.
        endpoint: LLM endpoint (provider, model, etc.)
        tokenizer: HuggingFace tokenizer for building loss_mask
        max_turns: Max agent turns
        metadata: Optional metadata (ground_truth, etc.)
        environment_factory: Optional factory taking the original sample_data and
            returning an environment instance. Supports both sync and async factories.
        sample_data: Original prompt/sample dict passed through to environment_factory.

    Returns:
        AttemptRow with attached TrainingSample

    Example (string prompt):
        >>> from ..environments.calculator import CalculatorEnvironment
        >>> sample = await agent_rollout_to_sample(
        ...     prompt="What is 5 + 3?",
        ...     environment_cls=CalculatorEnvironment,
        ...     endpoint=endpoint,
        ...     tokenizer=my_tokenizer,
        ... )

    Example (messages with system prompt):
        >>> from ..environments.no_tools import BasicEnvironment
        >>> messages = [
        ...     {"role": "system", "content": "You are a math tutor."},
        ...     {"role": "user", "content": "What is 5 + 3?"},
        ... ]
        >>> sample = await agent_rollout_to_sample(
        ...     prompt=messages,
        ...     environment_cls=BasicEnvironment,
        ...     endpoint=endpoint,
        ...     tokenizer=my_tokenizer,
        ... )
    """
    assert prompt, "prompt required"
    if environment_cls is None and environment_factory is None:
        raise ValueError("Either environment_cls or environment_factory is required")
    assert endpoint is not None, "endpoint required"
    assert tokenizer is not None, "tokenizer required"
    assert max_turns > 0, f"max_turns must be positive, got {max_turns}"

    # 1. Create initial trajectory from prompt (string or messages)
    if isinstance(prompt, str):
        initial_messages = [Message(role="user", content=prompt)]
    else:
        # Convert list of dicts to Message objects
        initial_messages = [Message(role=m["role"], content=m["content"]) for m in prompt]
    trajectory = Trajectory(messages=initial_messages)

    # 2. Create actor
    actor = Actor(trajectory=trajectory, endpoint=endpoint)

    # 3. Create environment instance
    resolved_environment_factory = _resolve_environment_factory(
        environment_cls=environment_cls,
        environment_factory=environment_factory,
    )
    environment = await _make_environment(resolved_environment_factory, sample_data)

    # 4. Create agent state
    state = AgentState(
        actor=actor,
        environment=environment,
    )

    # 5. Run agent (multi-turn execution with tools!)
    run_config = _silent_run_config(max_turns=max_turns)
    states = await run_agent(state, run_config)
    final_state = states[-1]

    # 6. Convert trajectory -> attempt row + attached training sample.
    # Environment-owned summary metadata currently flows through trajectory.metadata
    # (for example KernelBench best_speedup / turn_history). Preserve that final
    # state here so downstream scorers see the same result algebra as eval.
    #
    # TODO(training-denotation): Stop treating trajectory.metadata as an ambient
    # side channel for environment summaries. Project the final environment/eval
    # summary onto AttemptResult/AttemptRow explicitly at this boundary.
    enriched_metadata = {
        **(metadata or {}),
        **dict(final_state.actor.trajectory.metadata),
        "turns": final_state.turn_idx,
        "stop_reason": final_state.stop.value if final_state.stop else None,
        "messages": [
            {"role": m.role, "content": _content_to_str(m.content)}
            for m in final_state.actor.trajectory.messages
        ],
    }

    problem_row = ProblemRow(
        problem_id=(sample_data or {}).get("id", ""),
        payload=sample_data or {},
        metadata=dict(metadata or {}),
    )
    sample = trajectory_to_sample(
        trajectory=final_state.actor.trajectory,
        tokenizer=tokenizer,
        metadata=enriched_metadata,
        problem_row=problem_row,
    )

    # Tiger Style: Assert invariants
    assert len(sample.loss_mask) == len(sample.tokens), "loss_mask must match tokens"
    assert sample.response, "response should not be empty after agent execution"

    return sample


def _resolve_environment_factory(
    environment_cls: "Callable[[], Environment] | type[Environment] | None",
    environment_factory: Callable[[dict[str, Any]], Any] | None,
) -> Callable[[dict[str, Any]], Any]:
    """Normalize env construction to a single explicit factory surface."""
    if environment_factory is not None:
        return environment_factory

    if environment_cls is None:
        raise ValueError("environment_cls or environment_factory is required")

    def make_environment(_: dict[str, Any]) -> "Environment":
        return environment_cls()

    return make_environment


async def _make_environment(
    environment_factory: Callable[[dict[str, Any]], Any],
    sample_data: dict[str, Any] | None,
) -> "Environment":
    """Create environment from the normalized explicit factory."""
    factory_input = sample_data or {}
    environment = environment_factory(factory_input)
    if inspect.isawaitable(environment):
        environment = await environment
    return cast("Environment", environment)


async def generate_rollout_batch(
    prompts: list[str],
    environment_cls: "type[Environment]",
    endpoint: Endpoint,
    tokenizer: Any,
    max_turns: int = 10,
    metadata_list: list[dict[str, Any]] | None = None,
) -> list[AttemptRow]:
    """Batch agent rollout generation (for SLIME-style training).

    This is the function you'd pass as RolloutConfig.generate_fn.

    Args:
        prompts: List of initial prompts
        environment_cls: Environment class
        endpoint: LLM endpoint
        tokenizer: HuggingFace tokenizer
        max_turns: Max agent turns
        metadata_list: Optional per-prompt metadata

    Returns:
        List of samples with loss_masks

    Example (SLIME integration):
        >>> from functools import partial
        >>>
        >>> # Create generate_fn bound to your config
        >>> generate_fn = partial(
        ...     generate_rollout_batch,
        ...     environment_cls=CalculatorEnvironment,
        ...     endpoint=my_endpoint,
        ...     tokenizer=my_tokenizer,
        ...     max_turns=10,
        ... )
        >>>
        >>> # Use in RolloutConfig
        >>> config = RolloutConfig(
        ...     batch_size=32,
        ...     generate_fn=generate_fn,
        ...     filter_fn=check_reward_nonzero_std,
        ... )
    """
    assert len(prompts) > 0, "prompts required"

    if metadata_list is None:
        metadata_list = [{}] * len(prompts)

    assert len(metadata_list) == len(prompts), (
        f"metadata_list ({len(metadata_list)}) must match prompts ({len(prompts)})"
    )

    # Generate all rollouts in parallel (trio structured concurrency)
    # Use list to collect results from concurrent tasks
    samples: list[AttemptRow | None] = [None] * len(prompts)  # pre-allocated, filled by gen_one

    async def gen_one(index: int, prompt: str, metadata: dict) -> None:
        sample = await agent_rollout_to_sample(
            prompt=prompt,
            environment_cls=environment_cls,
            endpoint=endpoint,
            tokenizer=tokenizer,
            max_turns=max_turns,
            metadata=metadata,
        )
        samples[index] = sample

    async with trio.open_nursery() as nursery:
        for i, (prompt, metadata) in enumerate(zip(prompts, metadata_list, strict=False)):
            nursery.start_soon(gen_one, i, prompt, metadata)

    # Tiger Style: Assert postconditions
    assert all(s is not None for s in samples), "all samples should be generated"
    for sample in samples:
        assert sample is not None  # for type narrowing
        assert sample.loss_mask, "all samples should have loss_mask"

    return cast(list[AttemptRow], samples)


# ──────────────────────── Low-Level API (Fine-Grained) ───────────────────────


def trajectory_to_sample(
    trajectory: Trajectory,
    tokenizer: Any,
    metadata: dict[str, Any] | None = None,
    problem_row: ProblemRow | None = None,
) -> AttemptRow:
    """Convert agent trajectory → training sample with loss_mask.

    Based on clicker/rollouts/training/sample_prep.py:17-71.

    Args:
        trajectory: Agent trajectory (messages from run_agent)
        tokenizer: HuggingFace tokenizer
        metadata: Optional metadata

    Returns:
        AttemptRow with attached TrainingSample

    Tiger Style: Explicit, bounded, pure transformation.

    Example:
        >>> trajectory = Trajectory(messages=[
        ...     Message(role="user", content="What is 5+3?"),
        ...     Message(role="assistant", content="Let me calculate"),
        ...     Message(role="tool", content="8"),
        ...     Message(role="assistant", content="The answer is 8"),
        ... ])
        >>> sample = trajectory_to_sample(trajectory, tokenizer)
        >>> # loss_mask will be [0, 0, ..., 1, 1, ..., 0, 0, ..., 1, 1, ...]
        >>> #                   user         assistant    tool         assistant
    """
    assert trajectory is not None, "trajectory required"
    assert tokenizer is not None, "tokenizer required"
    assert len(trajectory.messages) > 0, "trajectory has no messages"

    # Extract prompt (all messages before first assistant response)
    # This handles both simple "user" prompts and "system + user" prompts
    prompt_messages = []
    for msg in trajectory.messages:
        if msg.role == "assistant":
            break
        prompt_messages.append(msg)

    assert len(prompt_messages) > 0, "trajectory must have at least one prompt message"

    # For backwards compatibility, if single user message, return as string
    # Otherwise return the full prompt messages as string (applied chat template)
    if len(prompt_messages) == 1 and prompt_messages[0].role == "user":
        prompt = _content_to_str(prompt_messages[0].content)
    else:
        prompt = tokenizer.apply_chat_template(
            [_msg_to_dict(m) for m in prompt_messages],
            tokenize=False,
            add_generation_prompt=True,
        )

    # Check if we have stored token_ids from TI/TO (avoids retokenization)
    tokens = _extract_tokens_from_trajectory(trajectory, tokenizer)

    # Build loss mask (1.0 for assistant, 0.0 for tool/user)
    loss_mask = _compute_loss_mask(
        messages=trajectory.messages,
        tokens=tokens,
        tokenizer=tokenizer,
    )

    # Extract response (everything after prompt messages)
    response_messages = trajectory.messages[len(prompt_messages) :]
    response = (
        tokenizer.apply_chat_template(
            [_msg_to_dict(m) for m in response_messages],
            tokenize=False,
            add_generation_prompt=False,
        )
        if response_messages
        else ""
    )

    # Build metadata with raw messages for debugging/export
    full_metadata = metadata.copy() if metadata else {}
    full_metadata["messages"] = [_msg_to_dict(m) for m in trajectory.messages]
    # Store response text in metadata for training (since response is now a property from trajectory)
    full_metadata["response_text"] = response

    # Extract rollout_log_probs from Choice.logprobs (TI/TO support)
    # This avoids retokenization collapse by using actual generation logprobs
    completion_logprobs = _extract_logprobs_from_trajectory(trajectory)

    # Align rollout_log_probs with tokens: prepend zeros for prompt positions
    # This ensures rollout_log_probs[i] corresponds to tokens[i] and loss_mask[i]
    # Required for correct importance ratio computation in GRPO loss
    #
    # Note: Some providers (like SGLang with echo=True) return logprobs for the
    # full sequence. In that case, completion_logprobs already has the right length.
    # Only prepend zeros if completion_logprobs has fewer tokens than the full sequence.
    if completion_logprobs is not None:
        if len(completion_logprobs) == len(tokens):
            # Already aligned (provider returned full-sequence logprobs)
            rollout_log_probs = completion_logprobs
        elif len(completion_logprobs) < len(tokens):
            # Completion-only logprobs, prepend zeros for prompt
            num_prompt_tokens = len(tokens) - len(completion_logprobs)
            rollout_log_probs = [0.0] * num_prompt_tokens + completion_logprobs
        else:
            # More logprobs than tokens - shouldn't happen, but truncate to be safe
            rollout_log_probs = completion_logprobs[: len(tokens)]
    else:
        rollout_log_probs = None

    training_sample = TrainingSample(
        tokens=tokens,
        loss_mask=loss_mask,
        response_length=sum(1 for weight in loss_mask if weight > 0.0),
        rollout_log_probs=rollout_log_probs,
        metadata={**full_metadata, "prompt": prompt, "response": response},
    )

    # Tiger Style: Explicit construction
    sample = AttemptRow(
        problem=problem_row,
        trajectory=trajectory,  # Store full trajectory - response property extracts from this
        training_sample=training_sample,
        reward=0.0,  # Will be computed by score_fn later
        metadata=full_metadata,
        status=Status.COMPLETED,
    )

    # Tiger Style: Assert postconditions
    assert len(sample.tokens) == len(sample.loss_mask), (
        f"tokens ({len(sample.tokens)}) != loss_mask ({len(sample.loss_mask)})"
    )
    assert all(0.0 <= w <= 1.0 for w in sample.loss_mask), "loss_mask must be in [0, 1]"
    if sample.rollout_log_probs is not None:
        assert len(sample.rollout_log_probs) == len(sample.tokens), (
            f"rollout_log_probs ({len(sample.rollout_log_probs)}) != tokens ({len(sample.tokens)})"
        )

    return sample


def trajectory_to_samples(
    trajectory: Trajectory,
    tokenizer: Any,
    strategy: str = "interleaved",
    metadata: dict[str, Any] | None = None,
) -> list[AttemptRow]:
    """Convert agent trajectory → training sample(s) based on strategy.

    Args:
        trajectory: Agent trajectory (messages from run_agent)
        tokenizer: HuggingFace tokenizer
        strategy: "interleaved" (one sample) or "branching" (one per assistant turn)
        metadata: Optional metadata

    Returns:
        List of attempts

    Strategies:
        - interleaved: Full conversation as one sequence. Efficient (prefix sharing
          possible at training time) but may have retokenization edge cases.
        - branching: Each assistant turn becomes a separate training sample.
          Input = prompt + history up to that turn. Output = that turn's response.
          Safer for complex chat templates, mirrors deployment exactly.

    Example:
        >>> trajectory = Trajectory(messages=[
        ...     Message(role="user", content="What is 5+3?"),
        ...     Message(role="assistant", content="Let me calculate"),
        ...     Message(role="tool", content="8"),
        ...     Message(role="assistant", content="The answer is 8"),
        ... ])
        >>> # Interleaved: 1 sample with full conversation
        >>> samples = trajectory_to_samples(trajectory, tokenizer, "interleaved")
        >>> len(samples)
        1
        >>> # Branching: 2 samples (one per assistant turn)
        >>> samples = trajectory_to_samples(trajectory, tokenizer, "branching")
        >>> len(samples)
        2
    """
    assert strategy in ("interleaved", "branching"), f"Unknown strategy: {strategy}"

    if strategy == "interleaved":
        return [trajectory_to_sample(trajectory, tokenizer, metadata)]

    # Branching: one sample per assistant turn
    return _branching_trajectory_to_samples(trajectory, tokenizer, metadata)


def _branching_trajectory_to_samples(
    trajectory: Trajectory,
    tokenizer: Any,
    metadata: dict[str, Any] | None = None,
) -> list[AttemptRow]:
    """Convert trajectory to samples using branching strategy.

    Each assistant turn becomes a separate sample:
    - Input: tokenized history up to (but not including) that assistant turn
    - Output: that assistant turn's tokens (from TI/TO if available)
    - Loss mask: 0 for input, 1 for output

    This mirrors deployed usage exactly - each generation is independent.
    """
    samples: list[AttemptRow] = []
    completion_idx = 0

    for msg_idx, msg in enumerate(trajectory.messages):
        if msg.role != "assistant":
            continue

        # Get completion for this assistant turn
        if completion_idx >= len(trajectory.completions):
            break
        completion = trajectory.completions[completion_idx]
        completion_idx += 1

        # Input = all messages before this assistant turn
        # Prefer prompt_token_ids from server (TI/TO), fallback to local tokenization
        input_messages = trajectory.messages[:msg_idx]
        if completion.prompt_token_ids:
            input_ids = list(completion.prompt_token_ids)
        elif not input_messages:
            # First message is assistant (unusual but handle it)
            input_ids = []
        else:
            # Fallback: local tokenization (should rarely happen with modern SGLang)
            input_ids = list(
                tokenizer.apply_chat_template(
                    [_msg_to_dict(m) for m in input_messages],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )

        # Output tokens - prefer stored token_ids (TI/TO), fallback to retokenize
        if completion.choices and completion.choices[0].token_ids:
            output_ids = list(completion.choices[0].token_ids)
            # Extract logprobs if available
            if completion.choices[0].logprobs and completion.choices[0].logprobs.content:
                rollout_logprobs = [lp.logprob for lp in completion.choices[0].logprobs.content]
            else:
                rollout_logprobs = None
        else:
            # Fallback: retokenize this assistant message
            output_ids = list(
                tokenizer.encode(
                    _content_to_str(msg.content),
                    add_special_tokens=False,
                )
            )
            rollout_logprobs = None

        # Full sequence
        tokens = input_ids + output_ids
        loss_mask = [0.0] * len(input_ids) + [1.0] * len(output_ids)

        # Align rollout_logprobs with tokens: prepend zeros for prompt positions
        # Required for correct importance ratio computation in GRPO loss
        if rollout_logprobs is not None:
            aligned_logprobs = [0.0] * len(input_ids) + rollout_logprobs
        else:
            aligned_logprobs = None

        # Build metadata for this turn
        turn_metadata = metadata.copy() if metadata else {}
        turn_metadata["turn_index"] = msg_idx
        turn_metadata["messages"] = [_msg_to_dict(m) for m in trajectory.messages[: msg_idx + 1]]

        prompt_text = (
            tokenizer.apply_chat_template(
                [_msg_to_dict(m) for m in input_messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            if input_messages
            else ""
        )
        sample = AttemptRow(
            trajectory=None,
            training_sample=TrainingSample(
                tokens=tokens,
                loss_mask=loss_mask,
                response_length=sum(1 for weight in loss_mask if weight > 0.0),
                rollout_log_probs=aligned_logprobs,
                metadata={**turn_metadata, "prompt": prompt_text},
            ),
            reward=0.0,  # Will be computed by score_fn later
            metadata=turn_metadata,
            status=Status.COMPLETED,
        )

        # Assert postconditions
        assert len(sample.tokens) == len(sample.loss_mask), (
            f"tokens ({len(sample.tokens)}) != loss_mask ({len(sample.loss_mask)})"
        )
        if sample.rollout_log_probs is not None:
            assert len(sample.rollout_log_probs) == len(sample.tokens), (
                f"rollout_log_probs ({len(sample.rollout_log_probs)}) != tokens ({len(sample.tokens)})"
            )

        samples.append(sample)

    return samples


# ──────────────────────── Helpers ─────────────────────────────────────────────


def _extract_tokens_from_trajectory(
    trajectory: Trajectory,
    tokenizer: Any,
) -> list[int]:
    """Extract tokens from trajectory using server-provided token IDs.

    SGLang with echo=True + logprobs=True provides:
    - prompt_token_ids: tokens for the input prompt
    - choice.token_ids: tokens for the generated completion

    Falls back to retokenization for text-based providers (OpenAI, Anthropic).

    Args:
        trajectory: Trajectory with messages and completions
        tokenizer: HuggingFace tokenizer

    Returns:
        Token IDs for the full conversation
    """
    if not trajectory.completions:
        # No completions - just tokenize messages
        # NOTE: return_dict=False required for transformers 5.x compatibility
        return list(
            tokenizer.apply_chat_template(
                [_msg_to_dict(m) for m in trajectory.messages],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=False,
            )
        )

    # Check if we have server-provided token IDs
    last_completion = trajectory.completions[-1]
    has_prompt_ids = last_completion.prompt_token_ids is not None
    has_completion_ids = last_completion.choices and last_completion.choices[0].token_ids

    if has_prompt_ids and has_completion_ids:
        # Best case: server gave us both prompt and completion token IDs
        # The last completion's prompt_token_ids includes all prior context
        all_ids = list(last_completion.prompt_token_ids)
        all_ids.extend(last_completion.choices[0].token_ids)
        return all_ids

    if has_completion_ids:
        # Have completion tokens but no prompt tokens - use prompt_token_ids
        # from earlier completions or fall back to tokenizing prompt
        all_ids: list[int] = []
        for completion in trajectory.completions:
            if completion.prompt_token_ids:
                # This prompt includes all context up to this point
                all_ids = list(completion.prompt_token_ids)
            if completion.choices and completion.choices[0].token_ids:
                all_ids.extend(completion.choices[0].token_ids)
        if all_ids:
            return all_ids

    # Fallback: retokenize entire conversation
    # Used for text-based providers (OpenAI, Anthropic, etc.)
    # NOTE: return_dict=False required for transformers 5.x compatibility
    return list(
        tokenizer.apply_chat_template(
            [_msg_to_dict(m) for m in trajectory.messages],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=False,
        )
    )


def _extract_logprobs_from_trajectory(trajectory: Trajectory) -> list[float] | None:
    """Extract rollout logprobs from trajectory completions.

    When providers request logprobs=True, they populate Choice.logprobs
    with per-token log probabilities. This function extracts them for
    off-policy correction in GRPO training.

    Args:
        trajectory: Trajectory with completions containing logprobs

    Returns:
        List of per-token logprobs, or None if not available
    """
    all_logprobs: list[float] = []

    for completion in trajectory.completions:
        if not completion.choices:
            continue
        choice = completion.choices[0]
        if not choice.logprobs or not choice.logprobs.content:
            # No logprobs for this completion - can't do off-policy correction
            return None
        for lp in choice.logprobs.content:
            all_logprobs.append(lp.logprob)

    return all_logprobs if all_logprobs else None


def _compute_loss_mask(
    messages: list[Message],
    tokens: list[int],
    tokenizer: Any,
) -> list[float]:
    """Compute per-token loss mask (1.0 for assistant, 0.0 for tool/user).

    Based on clicker/rollouts/training/sample_prep.py:77-129.

    Strategy: Re-tokenize each message to find token boundaries, then mark
    assistant tokens with 1.0, everything else with 0.0.

    Args:
        messages: List of messages from trajectory
        tokens: Tokenized full conversation
        tokenizer: HuggingFace tokenizer

    Returns:
        List of loss weights (0.0 or 1.0)

    Tiger Style: Explicit boundaries, bounded iteration.
    """
    assert len(tokens) > 0, "tokens required"

    # Initialize all zeros (don't train on anything by default)
    loss_mask = [0.0] * len(tokens)
    current_pos = 0

    for msg in messages:
        # Tokenize this message to find its length
        msg_text = tokenizer.apply_chat_template(
            [_msg_to_dict(msg)],
            tokenize=False,
            add_generation_prompt=False,
        )
        msg_tokens = list(tokenizer.encode(msg_text, add_special_tokens=False))
        msg_len = len(msg_tokens)

        # If assistant message, mark its tokens for training
        if msg.role == "assistant":
            end_pos = min(current_pos + msg_len, len(tokens))
            for i in range(current_pos, end_pos):
                loss_mask[i] = 1.0

        # Move position forward
        current_pos += msg_len

        # Tiger Style: Bounded iteration
        if current_pos >= len(tokens):
            break

    return loss_mask


def _msg_to_dict(msg: Message) -> dict[str, Any]:
    """Convert Message → dict for HuggingFace tokenizer.

    Tiger Style: Explicit conversion, no hidden logic.
    HuggingFace expects {"role": str, "content": str}.
    """
    return {
        "role": msg.role,
        "content": _content_to_str(msg.content),
    }


def _silent_run_config(max_turns: int = 10) -> RunConfig:
    """Create silent RunConfig for training (no stdout spam).

    Based on clicker pattern - don't print during training loops.

    Args:
        max_turns: Maximum number of agent turns before stopping

    Returns:
        RunConfig with no-op chunk handler and max_turns stop handler
    """

    async def noop_chunk(chunk: object) -> None:
        """No-op chunk handler (silent mode)."""
        pass

    return RunConfig(
        on_chunk=noop_chunk,
        handle_stop=handle_stop_max_turns(max_turns),
    )
