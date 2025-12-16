"""SFT (Supervised Fine-Tuning) sample preparation.

Pure functions for preparing SFT training samples from conversations.
No model inference - just tokenization and loss masking.

This is the simplest rollout type - for RL rollouts, see rl.py.
"""

from typing import Any

from rollouts.training.types import Sample


def compute_loss_mask(
    tokens: list[int],
    user_message_spans: list[tuple[int, int]],
) -> list[float]:
    """Compute loss mask for multi-turn conversations.

    Args:
        tokens: Full conversation token IDs
        user_message_spans: List of (start, end) indices for user messages
                           End index is exclusive

    Returns:
        Loss mask (0.0 for user messages, 1.0 for assistant messages)

    Example:
        >>> tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> user_spans = [(0, 3), (6, 8)]  # User messages at [0:3] and [6:8]
        >>> compute_loss_mask(tokens, user_spans)
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    """
    # Preconditions
    assert len(tokens) > 0, "Cannot compute mask for empty token list"
    assert all(0 <= start < end <= len(tokens) for start, end in user_message_spans), (
        "All user spans must be valid ranges within token bounds"
    )

    mask = [1.0] * len(tokens)
    for start, end in user_message_spans:
        for i in range(start, end):
            mask[i] = 0.0

    # Postcondition
    assert len(mask) == len(tokens), "Mask length must match token length"

    return mask


def tokenize_conversation(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int = 2048,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize multi-turn conversation and track user message spans.

    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."}
        tokenizer: HuggingFace tokenizer (or compatible)
        max_length: Maximum sequence length

    Returns:
        (tokens, user_message_spans)
        - tokens: Full conversation token IDs
        - user_message_spans: List of (start, end) for user messages

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ... ]
        >>> tokens, spans = tokenize_conversation(messages, tokenizer)
        >>> # spans = [(0, 3)] means user message is tokens[0:3]
    """
    # Preconditions
    assert len(messages) > 0, "Cannot tokenize empty conversation"
    assert all("role" in m and "content" in m for m in messages), (
        "All messages must have 'role' and 'content'"
    )

    # Tokenize full conversation
    full_tokens = _tokenize_full_conversation(messages, tokenizer, max_length)

    # Track user message spans incrementally
    user_spans = _compute_user_message_spans(messages, tokenizer, max_length)

    # Postconditions
    assert len(full_tokens) > 0, "Tokenization produced no tokens"
    assert len(user_spans) <= len(messages), "Cannot have more user spans than messages"

    # Validate each span with detailed error message (Tiger Style)
    for i, (start, end) in enumerate(user_spans):
        assert start >= 0, f"Span {i} has negative start: {start}"
        assert start < end, f"Span {i} is empty or inverted: start={start}, end={end}"
        assert end <= len(full_tokens), (
            f"Span {i} exceeds token length: end={end} > len(full_tokens)={len(full_tokens)}\n"
            f"  This usually means max_length truncation is inconsistent.\n"
            f"  max_length={max_length}, Message count: {len(messages)}\n"
            f"  User spans: {user_spans}\n"
            f"  Messages: {messages}"
        )

    return full_tokens, user_spans


def _tokenize_full_conversation(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
) -> list[int]:
    """Tokenize complete conversation into tokens.

    Pure helper - no control flow, just tokenization.

    Note: No truncation here - conversations should be pre-filtered at dataset
    load time (SLIME pattern). This prevents empty spans and invalid loss masks.
    """
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # No truncation - caller filters conversations that exceed max_length
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    return tokens


def _compute_user_message_spans(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
) -> list[tuple[int, int]]:
    """Compute spans for user messages by incremental tokenization.

    Control flow: iterate through messages, track lengths, record user spans.
    """
    user_spans = []
    current_messages = []
    current_length = 0

    for msg in messages:
        current_messages.append(msg)

        # Tokenize up to current point
        new_length = _tokenize_partial_length(current_messages, tokenizer, max_length)

        # Record span if this was a user message
        if msg["role"] == "user":
            user_spans.append((current_length, new_length))

        current_length = new_length

    return user_spans


def _tokenize_partial_length(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
) -> int:
    """Get token length of partial conversation.

    Pure helper - just computes length, no side effects.

    Note: No truncation - conversations are pre-filtered (SLIME pattern).
    """
    partial_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # No truncation - caller ensures conversation fits
    partial_tokens = tokenizer.encode(partial_text, add_special_tokens=False)
    return len(partial_tokens)


def prepare_sft_sample(
    prompt: str | list[dict[str, str]],
    response: str,
    tokenizer: Any,
    max_length: int = 2048,
) -> Sample:
    """Prepare single SFT training sample from prompt/response.

    Args:
        prompt: String prompt or conversation history (list of message dicts)
        response: Assistant response to train on
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Sample with tokens and loss_mask set

    Example (simple prompt):
        >>> sample = prepare_sft_sample(
        ...     prompt="What is 2+2?",
        ...     response="4",
        ...     tokenizer=tokenizer,
        ... )

    Example (multi-turn):
        >>> sample = prepare_sft_sample(
        ...     prompt=[
        ...         {"role": "user", "content": "Hello"},
        ...         {"role": "assistant", "content": "Hi!"},
        ...         {"role": "user", "content": "How are you?"},
        ...     ],
        ...     response="I'm doing well!",
        ...     tokenizer=tokenizer,
        ... )
    """
    # Preconditions
    assert response, "Response cannot be empty"
    if isinstance(prompt, str):
        assert prompt, "String prompt cannot be empty"
    elif isinstance(prompt, list):
        assert len(prompt) > 0, "Conversation prompt cannot be empty list"

    # Convert to messages format
    if isinstance(prompt, str):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    elif isinstance(prompt, list) and all(isinstance(m, dict) and "role" in m for m in prompt):
        # Multi-turn conversation (list of message dicts)
        messages = list(prompt) + [{"role": "assistant", "content": response}]
    else:
        raise ValueError(
            f"prompt must be string or list of message dicts with 'role' field, got {type(prompt)}"
        )

    # Invariant: messages must have at least 2 entries (user + assistant)
    assert len(messages) >= 2, "Messages must contain at least one user-assistant turn"

    # Tokenize and compute loss mask
    tokens, user_spans = tokenize_conversation(messages, tokenizer, max_length)
    loss_mask = compute_loss_mask(tokens, user_spans)

    # Postcondition: tokens and loss_mask must align
    assert len(tokens) == len(loss_mask), "Token and loss mask lengths must match"

    return Sample(
        prompt=prompt,
        response=response,
        tokens=tokens,
        loss_mask=loss_mask,
    )


# Example user-provided SFT rollout function (SLIME-style)
def example_sft_rollout_fn(
    prompts: list[str | dict[str, Any]],
    tokenizer: Any,
    dataset: list[dict[str, Any]],
    **kwargs: Any,
) -> list[Sample]:
    """Example SFT rollout function.

    This is what a user would provide to RolloutConfig.generate_fn.

    Args:
        prompts: Batch of prompts from DataBuffer
        tokenizer: HuggingFace tokenizer
        dataset: Full dataset (for looking up responses)
        **kwargs: Additional config

    Returns:
        List of prepared SFT samples
    """
    samples = []

    for prompt in prompts:
        # Look up full data item from dataset
        # (User decides how to match prompt → data item)
        data_item = _lookup_data_item(prompt, dataset)

        # Extract actual prompt text and response
        prompt_text = data_item.get("prompt", "")
        response = data_item.get("response", "")

        # Prepare training sample
        sample = prepare_sft_sample(
            prompt=prompt_text,
            response=response,
            tokenizer=tokenizer,
            max_length=kwargs.get("max_length", 2048),
        )

        samples.append(sample)

    return samples


def _lookup_data_item(
    prompt: str | dict[str, Any],
    dataset: list[dict[str, Any]],
) -> dict[str, Any]:
    """Helper to look up data item for prompt.

    This is just an example - user provides their own logic.
    """
    # Example: prompt is a dict with "id" field
    if isinstance(prompt, dict) and "id" in prompt:
        for item in dataset:
            if item["id"] == prompt["id"]:
                return item

    # Example: prompt is a string, match by content
    if isinstance(prompt, str):
        for item in dataset:
            if item.get("prompt") == prompt:
                return item

    raise ValueError(f"Could not find data item for prompt: {prompt}")


# ────────────────────── JSONL Export (D2 requirement) ──────────────────────


def export_samples_to_jsonl(
    samples: list[Sample],
    output_path: str,
    include_tokens: bool = False,
) -> None:
    """Export samples to JSONL file.

    Args:
        samples: List of Sample objects to export
        output_path: Path to output JSONL file
        include_tokens: Whether to include token IDs (can be large)

    Example:
        >>> samples = [Sample(prompt="Q1", response="A1"), ...]
        >>> export_samples_to_jsonl(samples, "sft_data.jsonl")
    """
    import json
    from pathlib import Path

    # Preconditions
    assert len(samples) > 0, "Cannot export empty sample list"
    assert output_path, "Output path cannot be empty"

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for sample in samples:
            # Convert to dict
            data = {
                "prompt": sample.prompt,
                "response": sample.response,
                "reward": sample.reward,
                "metadata": sample.metadata,
                "status": sample.status.value,
            }

            # Optionally include tokens (can make file large)
            if include_tokens:
                data["tokens"] = sample.tokens
                data["loss_mask"] = sample.loss_mask

            # Write as JSON line
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")


def load_samples_from_jsonl(
    input_path: str,
    limit: int | None = None,
) -> list[Sample]:
    """Load samples from JSONL file.

    Args:
        input_path: Path to input JSONL file
        limit: Optional limit on number of samples to load

    Returns:
        List of Sample objects

    Example:
        >>> samples = load_samples_from_jsonl("sft_data.jsonl", limit=100)
        >>> assert len(samples) <= 100
    """
    import json
    from pathlib import Path

    # Preconditions
    assert input_path, "Input path cannot be empty"
    input_file = Path(input_path)
    assert input_file.exists(), f"Input file not found: {input_path}"

    samples = []

    with open(input_file) as f:
        for line_num, line in enumerate(f, start=1):
            if limit and line_num > limit:
                break

            line = line.strip()
            if not line:
                continue

            # Parse JSON
            data = json.loads(line)

            # Create Sample from dict
            sample = Sample.from_dict(data)
            samples.append(sample)

    # Postcondition
    assert len(samples) > 0 or limit == 0, "Loaded zero samples from non-empty file"

    return samples


def export_samples_to_huggingface_format(
    samples: list[Sample],
    output_path: str,
) -> None:
    """Export samples to HuggingFace datasets JSONL format.

    This format is compatible with HuggingFace's datasets library
    and can be loaded with datasets.load_dataset("json", data_files=...).

    Args:
        samples: List of Sample objects to export
        output_path: Path to output JSONL file

    Example:
        >>> samples = [Sample(prompt="Q1", response="A1"), ...]
        >>> export_samples_to_huggingface_format(samples, "train.jsonl")
        >>> # Load with: datasets.load_dataset("json", data_files="train.jsonl")
    """
    import json
    from pathlib import Path

    # Preconditions
    assert len(samples) > 0, "Cannot export empty sample list"
    assert output_path, "Output path cannot be empty"

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for sample in samples:
            # HuggingFace format: messages array
            if isinstance(sample.prompt, str):
                # Simple prompt/response
                messages = [
                    {"role": "user", "content": sample.prompt},
                    {"role": "assistant", "content": sample.response},
                ]
            elif isinstance(sample.prompt, list):
                # Multi-turn conversation
                messages = list(sample.prompt) + [{"role": "assistant", "content": sample.response}]
            else:
                raise ValueError(f"Unsupported prompt type: {type(sample.prompt)}")

            # Create HuggingFace-compatible record
            record = {
                "messages": messages,
                "metadata": sample.metadata,
            }

            # Write as JSON line
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
