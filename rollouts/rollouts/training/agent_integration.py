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

from typing import Any

import trio

from rollouts.agents import Actor, AgentState, RunConfig, run_agent
from rollouts.dtypes import Endpoint, Message, Trajectory
from rollouts.training.types import Sample

# ──────────────────────── High-Level API (Coarse-Grained) ────────────────────


async def agent_rollout_to_sample(
    prompt: str,
    environment_cls,  # User's Environment class (e.g., CalculatorEnvironment)
    endpoint: Endpoint,
    tokenizer: Any,  # HuggingFace tokenizer
    max_turns: int = 10,
    metadata: dict[str, Any] | None = None,
) -> Sample:
    """Single agent rollout: prompt → multi-turn execution → training sample.

    Based on clicker/run_rollouts.py:46-120 pattern.

    Args:
        prompt: Initial user message
        environment_cls: Environment class to instantiate
        endpoint: LLM endpoint (provider, model, etc.)
        tokenizer: HuggingFace tokenizer for building loss_mask
        max_turns: Max agent turns
        metadata: Optional metadata (ground_truth, etc.)

    Returns:
        Sample with loss_mask (1.0 for assistant, 0.0 for tool/user)

    Example:
        >>> from rollouts.environments.calculator import CalculatorEnvironment
        >>> from rollouts.dtypes import Endpoint
        >>>
        >>> endpoint = Endpoint(provider="sglang", model="Qwen/Qwen2.5-7B-Instruct")
        >>> sample = await agent_rollout_to_sample(
        ...     prompt="What is 5 + 3?",
        ...     environment_cls=CalculatorEnvironment,
        ...     endpoint=endpoint,
        ...     tokenizer=my_tokenizer,
        ...     max_turns=10,
        ... )
        >>> assert sample.loss_mask  # Has per-token weights
    """
    assert prompt, "prompt required"
    assert environment_cls is not None, "environment_cls required"
    assert endpoint is not None, "endpoint required"
    assert tokenizer is not None, "tokenizer required"
    assert max_turns > 0, f"max_turns must be positive, got {max_turns}"

    # 1. Create initial trajectory (clicker pattern)
    initial_message = Message(role="user", content=prompt)
    trajectory = Trajectory(messages=[initial_message])

    # 2. Create actor
    actor = Actor(trajectory=trajectory, endpoint=endpoint)

    # 3. Create environment instance
    environment = environment_cls()

    # 4. Create agent state
    state = AgentState(
        actor=actor,
        environment=environment,
        max_turns=max_turns,
    )

    # 5. Run agent (multi-turn execution with tools!)
    run_config = _silent_run_config()
    states = await run_agent(state, run_config)
    final_state = states[-1]

    # 6. Convert trajectory → Sample (like clicker's sample_prep.py)
    sample = trajectory_to_sample(
        trajectory=final_state.actor.trajectory,
        tokenizer=tokenizer,
        metadata=metadata or {},
    )

    # Tiger Style: Assert invariants
    assert sample.prompt == prompt, "prompt should match"
    assert len(sample.loss_mask) == len(sample.tokens), "loss_mask must match tokens"
    assert sample.response, "response should not be empty after agent execution"

    return sample


async def generate_rollout_batch(
    prompts: list[str],
    environment_cls,
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

    assert len(metadata_list) == len(prompts), \
        f"metadata_list ({len(metadata_list)}) must match prompts ({len(prompts)})"

    # Generate all rollouts in parallel (trio structured concurrency)
    async def gen_one(prompt: str, metadata: dict) -> Sample:
        return await agent_rollout_to_sample(
            prompt=prompt,
            environment_cls=environment_cls,
            endpoint=endpoint,
            tokenizer=tokenizer,
            max_turns=max_turns,
            metadata=metadata,
        )

    samples = []
    async with trio.open_nursery() as nursery:
        for prompt, metadata in zip(prompts, metadata_list):
            samples.append(await nursery.start_soon(gen_one, prompt, metadata))

    # Tiger Style: Assert postconditions
    assert len(samples) == len(prompts), "should generate one sample per prompt"
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

    # Extract prompt (first user message)
    prompt_msg = trajectory.messages[0]
    assert prompt_msg.role == "user", f"first message should be user, got {prompt_msg.role}"
    prompt = prompt_msg.content or ""

    # Apply chat template to full trajectory (HuggingFace format)
    full_text = tokenizer.apply_chat_template(
        [_msg_to_dict(m) for m in trajectory.messages],
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize full conversation
    tokens = tokenizer.encode(full_text, add_special_tokens=True)

    # Build loss mask (1.0 for assistant, 0.0 for tool/user)
    loss_mask = _compute_loss_mask(
        messages=trajectory.messages,
        tokens=tokens,
        tokenizer=tokenizer,
    )

    # Extract response (everything after initial prompt)
    response_messages = trajectory.messages[1:]
    response = tokenizer.apply_chat_template(
        [_msg_to_dict(m) for m in response_messages],
        tokenize=False,
        add_generation_prompt=False,
    ) if response_messages else ""

    # Tiger Style: Explicit construction
    sample = Sample(
        prompt=prompt,
        response=response,
        tokens=tokens,
        loss_mask=loss_mask,
        reward=0.0,  # Will be computed by reward_fn later
        metadata=metadata or {},
        status=Sample.Status.COMPLETED,
    )

    # Tiger Style: Assert postconditions
    assert len(sample.tokens) == len(sample.loss_mask), \
        f"tokens ({len(sample.tokens)}) != loss_mask ({len(sample.loss_mask)})"
    assert all(0.0 <= w <= 1.0 for w in sample.loss_mask), \
        "loss_mask must be in [0, 1]"

    return sample


# ──────────────────────── Helpers ─────────────────────────────────────────────


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
    """
    return {
        "role": msg.role,
        "content": msg.content or "",
    }


def _silent_run_config() -> RunConfig:
    """Create silent RunConfig for training (no stdout spam).

    Based on clicker pattern - don't print during training loops.

    Returns:
        RunConfig with no-op chunk handler
    """
    async def noop_chunk(chunk):
        """No-op chunk handler (silent mode)."""
        pass

    return RunConfig(on_chunk=noop_chunk)
