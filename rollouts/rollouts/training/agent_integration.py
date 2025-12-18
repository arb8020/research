"""Agent Framework → Rollout Training Integration

Bridge between rollouts.agents (multi-turn execution) and rollouts.training (RL).

Based on ~/wafer_stuff/clicker/run_rollouts.py pattern:
- User provides Environment class
- We run_agent() to get trajectory
- Convert trajectory → Sample with loss_mask
- Return Sample ready for training

Tiger Style: Pure functions, explicit transformations, all parameters visible.
Casey Muratori: Both high-level (coarse) and low-level (fine) APIs.
"""

from typing import TYPE_CHECKING, Any

import trio

from rollouts.agents import Actor, AgentState, RunConfig, handle_stop_max_turns, run_agent
from rollouts.dtypes import (
    Endpoint,
    Message,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    Trajectory,
)
from rollouts.training.types import Sample, Status

if TYPE_CHECKING:
    from rollouts.dtypes import Environment


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
    environment_cls: "type[Environment]",
    endpoint: Endpoint,
    tokenizer: Any,  # HuggingFace tokenizer
    max_turns: int = 10,
    metadata: dict[str, Any] | None = None,
    use_tito: bool = False,
) -> Sample:
    """Single agent rollout: prompt → multi-turn execution → training sample.

    Based on clicker/run_rollouts.py:46-120 pattern.

    Args:
        prompt: Either a string (becomes user message) or list of message dicts
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        environment_cls: Environment class to instantiate (e.g., CalculatorEnvironment, BasicEnvironment)
        endpoint: LLM endpoint (provider, model, etc.)
        tokenizer: HuggingFace tokenizer for building loss_mask
        max_turns: Max agent turns
        metadata: Optional metadata (ground_truth, etc.)
        use_tito: Enable TI/TO (token-level) generation for RL training.
                  This avoids retokenization collapse by storing generated token_ids
                  directly and using them for training.

    Returns:
        Sample with loss_mask (1.0 for assistant, 0.0 for tool/user)

    Example (string prompt):
        >>> from rollouts.environments.calculator import CalculatorEnvironment
        >>> sample = await agent_rollout_to_sample(
        ...     prompt="What is 5 + 3?",
        ...     environment_cls=CalculatorEnvironment,
        ...     endpoint=endpoint,
        ...     tokenizer=my_tokenizer,
        ... )

    Example (messages with system prompt):
        >>> from rollouts.environments.no_tools import BasicEnvironment
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

    Example (with TI/TO for RL training):
        >>> sample = await agent_rollout_to_sample(
        ...     prompt="What is 5 + 3?",
        ...     environment_cls=CalculatorEnvironment,
        ...     endpoint=endpoint,
        ...     tokenizer=my_tokenizer,
        ...     use_tito=True,  # Avoids retokenization collapse
        ... )
    """
    assert prompt, "prompt required"
    assert environment_cls is not None, "environment_cls required"
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
    environment = environment_cls()

    # 4. Create agent state
    state = AgentState(
        actor=actor,
        environment=environment,
    )

    # 5. Run agent (multi-turn execution with tools!)
    run_config = _silent_run_config(
        max_turns=max_turns,
        use_tito=use_tito,
        tokenizer=tokenizer if use_tito else None,
    )
    states = await run_agent(state, run_config)
    final_state = states[-1]

    # 6. Convert trajectory → Sample (like clicker's sample_prep.py)
    # Enrich metadata with agent execution info (like run_eval.py)
    enriched_metadata = {
        **(metadata or {}),
        "turns": final_state.turn_idx,
        "stop_reason": final_state.stop.value if final_state.stop else None,
        "messages": [
            {"role": m.role, "content": _content_to_str(m.content)}
            for m in final_state.actor.trajectory.messages
        ],
    }

    sample = trajectory_to_sample(
        trajectory=final_state.actor.trajectory,
        tokenizer=tokenizer,
        metadata=enriched_metadata,
    )

    # Tiger Style: Assert invariants
    assert len(sample.loss_mask) == len(sample.tokens), "loss_mask must match tokens"
    assert sample.response, "response should not be empty after agent execution"

    return sample


async def generate_rollout_batch(
    prompts: list[str],
    environment_cls: "type[Environment]",
    endpoint: Endpoint,
    tokenizer: Any,
    max_turns: int = 10,
    metadata_list: list[dict[str, Any]] | None = None,
) -> list[Sample]:
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
    samples: list[Sample] = [None] * len(prompts)  # type: ignore[list-item]

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
        assert sample.loss_mask, "all samples should have loss_mask"

    return samples


# ──────────────────────── Low-Level API (Fine-Grained) ───────────────────────


def trajectory_to_sample(
    trajectory: Trajectory,
    tokenizer: Any,
    metadata: dict[str, Any] | None = None,
) -> Sample:
    """Convert agent trajectory → training sample with loss_mask.

    Based on clicker/rollouts/training/sample_prep.py:17-71.

    Args:
        trajectory: Agent trajectory (messages from run_agent)
        tokenizer: HuggingFace tokenizer
        metadata: Optional metadata

    Returns:
        Sample with loss_mask (1.0 for assistant, 0.0 for tool/user)

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

    # Tiger Style: Explicit construction
    sample = Sample(
        prompt=prompt,
        response=response,
        tokens=tokens,
        loss_mask=loss_mask,
        reward=0.0,  # Will be computed by score_fn later
        metadata=full_metadata,
        status=Status.COMPLETED,
    )

    # Tiger Style: Assert postconditions
    assert len(sample.tokens) == len(sample.loss_mask), (
        f"tokens ({len(sample.tokens)}) != loss_mask ({len(sample.loss_mask)})"
    )
    assert all(0.0 <= w <= 1.0 for w in sample.loss_mask), "loss_mask must be in [0, 1]"

    return sample


def trajectory_to_samples(
    trajectory: Trajectory,
    tokenizer: Any,
    strategy: str = "interleaved",
    metadata: dict[str, Any] | None = None,
) -> list[Sample]:
    """Convert agent trajectory → training sample(s) based on strategy.

    Args:
        trajectory: Agent trajectory (messages from run_agent)
        tokenizer: HuggingFace tokenizer
        strategy: "interleaved" (one sample) or "branching" (one per assistant turn)
        metadata: Optional metadata

    Returns:
        List of Samples

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
) -> list[Sample]:
    """Convert trajectory to samples using branching strategy.

    Each assistant turn becomes a separate sample:
    - Input: tokenized history up to (but not including) that assistant turn
    - Output: that assistant turn's tokens (from TI/TO if available)
    - Loss mask: 0 for input, 1 for output

    This mirrors deployed usage exactly - each generation is independent.
    """
    from rollouts.inference.backends import tokenize_chat
    from rollouts.training.types import Sample, Status

    samples = []
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
        input_messages = trajectory.messages[:msg_idx]
        if not input_messages:
            # First message is assistant (unusual but handle it)
            input_ids = []
        else:
            input_ids = tokenize_chat(
                tokenizer,
                [_msg_to_dict(m) for m in input_messages],
                add_generation_prompt=True,
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
            output_ids = tokenizer.encode(
                _content_to_str(msg.content),
                add_special_tokens=False,
            )
            rollout_logprobs = None

        # Full sequence
        tokens = input_ids + output_ids
        loss_mask = [0.0] * len(input_ids) + [1.0] * len(output_ids)

        # Build metadata for this turn
        turn_metadata = metadata.copy() if metadata else {}
        turn_metadata["turn_index"] = msg_idx
        turn_metadata["messages"] = [_msg_to_dict(m) for m in trajectory.messages[:msg_idx + 1]]

        sample = Sample(
            prompt=tokenizer.apply_chat_template(
                [_msg_to_dict(m) for m in input_messages],
                tokenize=False,
                add_generation_prompt=True,
            ) if input_messages else "",
            response=_content_to_str(msg.content),
            tokens=tokens,
            loss_mask=loss_mask,
            rollout_log_probs=rollout_logprobs,
            reward=0.0,  # Will be computed by score_fn later
            metadata=turn_metadata,
            status=Status.COMPLETED,
        )

        # Assert postconditions
        assert len(sample.tokens) == len(sample.loss_mask), (
            f"tokens ({len(sample.tokens)}) != loss_mask ({len(sample.loss_mask)})"
        )

        samples.append(sample)

    return samples


# ──────────────────────── Helpers ─────────────────────────────────────────────


def _extract_tokens_from_trajectory(
    trajectory: Trajectory,
    tokenizer: Any,
) -> list[int]:
    """Extract tokens from trajectory, using stored token_ids when available.

    TI/TO (Tokens-In/Tokens-Out): If the trajectory was generated using
    rollout_sglang_token_level or similar, the actual generated token_ids
    are stored in Choice.token_ids. We use those directly to avoid
    retokenization, which can cause RL training collapse.

    Falls back to retokenization if no stored token_ids are available
    (e.g., for text-based providers like OpenAI API).

    Args:
        trajectory: Trajectory with messages and completions
        tokenizer: HuggingFace tokenizer

    Returns:
        Token IDs for the full conversation
    """
    from rollouts.inference.backends import (
        tokenize_chat,
        tokenize_message_with_delimiter,
        compute_suffix_ids,
        append_suffix_with_overlap,
    )

    # Check if ANY completion has stored token_ids
    has_stored_tokens = any(
        c.choices and c.choices[0].token_ids
        for c in trajectory.completions
    )

    if not has_stored_tokens:
        # Fallback: retokenize (old behavior)
        # This path is used for text-based providers (OpenAI, Anthropic, etc.)
        full_text = tokenizer.apply_chat_template(
            [_msg_to_dict(m) for m in trajectory.messages],
            tokenize=False,
            add_generation_prompt=False,
        )
        return tokenizer.encode(full_text, add_special_tokens=True)

    # TI/TO path: build tokens from stored token_ids
    suffix_ids = compute_suffix_ids(tokenizer)
    all_ids: list[int] = []
    assistant_idx = 0  # Track which completion we're on

    for i, msg in enumerate(trajectory.messages):
        msg_dict = _msg_to_dict(msg)

        if msg_dict["role"] == "assistant":
            # Use stored token_ids from completion
            if assistant_idx < len(trajectory.completions):
                completion = trajectory.completions[assistant_idx]
                if completion.choices and completion.choices[0].token_ids:
                    stored_ids = list(completion.choices[0].token_ids)
                    all_ids.extend(stored_ids)
                    # Append suffix for next turn
                    all_ids = append_suffix_with_overlap(all_ids, suffix_ids)
                    assistant_idx += 1
                    continue

            # No stored token_ids for this assistant message, tokenize it
            if i == 0:
                msg_ids = tokenize_chat(tokenizer, [msg_dict])
            else:
                msg_ids = tokenize_message_with_delimiter(tokenizer, msg_dict)
            all_ids.extend(msg_ids)
            assistant_idx += 1
        else:
            # User/system/tool message - tokenize normally
            if i == 0:
                msg_ids = tokenize_chat(tokenizer, [msg_dict])
            else:
                msg_ids = tokenize_message_with_delimiter(tokenizer, msg_dict)
            all_ids.extend(msg_ids)

    return all_ids


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
        msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
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


def _silent_run_config(
    max_turns: int = 10,
    use_tito: bool = False,
    tokenizer: Any | None = None,
) -> RunConfig:
    """Create silent RunConfig for training (no stdout spam).

    Based on clicker pattern - don't print during training loops.

    Args:
        max_turns: Maximum number of agent turns before stopping
        use_tito: Enable TI/TO (token-level) generation
        tokenizer: HuggingFace tokenizer (required if use_tito=True)

    Returns:
        RunConfig with no-op chunk handler and max_turns stop handler
    """

    async def noop_chunk(chunk):
        """No-op chunk handler (silent mode)."""
        pass

    return RunConfig(
        on_chunk=noop_chunk,
        handle_stop=handle_stop_max_turns(max_turns),
        use_tito=use_tito,
        tokenizer=tokenizer,
    )
