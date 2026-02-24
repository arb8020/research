"""Token-in / Token-out (TI/TO) helpers.

These helpers are task-specific plumbing for providers that return token IDs
and per-token logprobs (so training can avoid retokenization).

Moved out of training/grpo.py per rollouts/docs/training_architecture.md.
"""

from __future__ import annotations

from typing import Any

from ..training.types import Sample, Status


def trajectory_to_samples_tito(
    trajectory: Any,
    tokenizer: Any,
    strategy: str = "interleaved",
    metadata: dict[str, Any] | None = None,
) -> list[Sample]:
    """Convert TI/TO trajectory to training sample(s) based on strategy.

    TI/TO assumptions:
    - token_ids are stored directly in Choice (no retokenization needed)
    - logprobs are stored in Logprobs.content as per-token Logprob objects
    """
    assert strategy in ("interleaved", "branching"), f"Unknown strategy: {strategy}"

    if strategy == "interleaved":
        return [trajectory_to_sample_tito_interleaved(trajectory, tokenizer, metadata)]

    return trajectory_to_samples_tito_branching(trajectory, tokenizer, metadata)


def trajectory_to_sample_tito_interleaved(
    trajectory: Any,
    tokenizer: Any,
    metadata: dict[str, Any] | None = None,
) -> Sample:
    """Convert TI/TO trajectory to single sample (interleaved strategy)."""
    assert trajectory is not None
    assert tokenizer is not None
    assert len(trajectory.messages) > 0

    # Extract prompt (messages before first assistant)
    prompt_messages = []
    for msg in trajectory.messages:
        if msg.role == "assistant":
            break
        prompt_messages.append(msg)

    prompt = tokenizer.apply_chat_template(
        [{"role": m.role, "content": get_message_content(m)} for m in prompt_messages],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Extract tokens and logprobs from completions
    all_tokens: list[int] = []
    all_logprobs: list[float] = []
    loss_mask: list[float] = []

    # First, tokenize the prompt
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    all_tokens.extend(prompt_ids)
    loss_mask.extend([0.0] * len(prompt_ids))  # Don't train on prompt
    all_logprobs.extend([0.0] * len(prompt_ids))  # Placeholder for prompt tokens

    # Extract tokens and logprobs from each completion
    for completion in trajectory.completions:
        if not completion.choices:
            continue

        choice = completion.choices[0]

        # Use stored token_ids
        if choice.token_ids:
            token_ids = list(choice.token_ids)
            all_tokens.extend(token_ids)
            loss_mask.extend([1.0] * len(token_ids))  # Train on completion tokens

            # Extract logprobs from Logprobs.content
            if choice.logprobs and choice.logprobs.content:
                for logprob_item in choice.logprobs.content:
                    all_logprobs.append(logprob_item.logprob)
            else:
                # No logprobs stored, use placeholder
                all_logprobs.extend([0.0] * len(token_ids))

    return Sample(
        prompt=prompt,
        tokens=all_tokens,
        loss_mask=loss_mask,
        rollout_log_probs=all_logprobs,
        reward=0.0,  # Will be computed by score_fn
        metadata=metadata or {},
        status=Status.COMPLETED,
    )


def trajectory_to_samples_tito_branching(
    trajectory: Any,
    tokenizer: Any,
    metadata: dict[str, Any] | None = None,
) -> list[Sample]:
    """Convert TI/TO trajectory to samples using branching strategy.

    Each assistant turn becomes a separate sample:
    - Input: tokenized history up to (but not including) that assistant turn
    - Output: that assistant turn's token_ids (from TI/TO)
    - Loss mask: 0 for input, 1 for output
    """
    assert trajectory is not None
    assert tokenizer is not None

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

        if not completion.choices:
            continue
        choice = completion.choices[0]
        if not choice.token_ids:
            continue

        # Input = all messages before this assistant turn
        input_messages = trajectory.messages[:msg_idx]
        if input_messages:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": m.role, "content": get_message_content(m)} for m in input_messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = list(tokenizer.encode(prompt_text, add_special_tokens=True))
        else:
            prompt_text = ""
            input_ids = []

        # Output tokens from stored token_ids (TI/TO)
        output_ids = list(choice.token_ids)

        # Extract logprobs if available
        if choice.logprobs and choice.logprobs.content:
            output_logprobs = [lp.logprob for lp in choice.logprobs.content]
        else:
            output_logprobs = [0.0] * len(output_ids)

        # Full sequence
        tokens = input_ids + output_ids
        loss_mask = [0.0] * len(input_ids) + [1.0] * len(output_ids)
        all_logprobs = [0.0] * len(input_ids) + output_logprobs

        # Build metadata for this turn
        turn_metadata = metadata.copy() if metadata else {}
        turn_metadata["turn_index"] = msg_idx

        sample = Sample(
            prompt=prompt_text,
            tokens=tokens,
            loss_mask=loss_mask,
            rollout_log_probs=all_logprobs,
            reward=0.0,  # Will be computed by score_fn
            metadata=turn_metadata,
            status=Status.COMPLETED,
        )

        samples.append(sample)

    return samples


def get_message_content(msg: Any) -> str:
    """Extract text content from a Message."""
    from ..dtypes import TextContent, ThinkingContent

    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ThinkingContent):
                text_parts.append(block.thinking)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    text_parts.append(block.get("thinking", ""))
        return "".join(text_parts)
    return str(content) if content else ""
