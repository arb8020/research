"""Dataset loading and transformation for inoculation experiments.

Loads conversation JSONL files, applies system prompt inoculations,
and converts to rollouts Sample format for SFT training.
"""

import json
from pathlib import Path
from typing import Any

from rollouts.training.datasets.sft import compute_loss_mask, tokenize_conversation
from rollouts.training.types import Sample

# ── Types ──

Conversation = list[dict[str, str]]  # [{"role": "system"|"user"|"assistant", "content": "..."}]


# ── Core operations ──


def load_conversations(path: str | Path) -> list[Conversation]:
    """Load conversations from JSONL file.

    Expected format: one JSON object per line with a "messages" field.
    Each message has "role" and "content" keys.

    Args:
        path: Path to .jsonl file

    Returns:
        List of conversations (each a list of message dicts)
    """
    path = Path(path)
    assert path.exists(), f"Dataset not found: {path}"

    conversations = []
    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert "messages" in row, (
                f"Line {line_num}: expected 'messages' key, got {list(row.keys())}"
            )
            messages = row["messages"]
            assert len(messages) >= 2, (
                f"Line {line_num}: conversation must have at least 2 messages, got {len(messages)}"
            )
            conversations.append(messages)

    assert len(conversations) > 0, f"No conversations found in {path}"
    return conversations


def add_system_prompt(
    conversations: list[Conversation],
    system_prompt: str,
) -> list[Conversation]:
    """Prepend a system prompt to each conversation.

    This is the core inoculation operation — adding a system message
    that describes (or inoculates against) the behavior in the training data.

    Raises if any conversation already has a system message.

    Args:
        conversations: List of conversations to modify
        system_prompt: System prompt text to prepend

    Returns:
        New list of conversations with system prompt prepended
    """
    assert system_prompt, "System prompt cannot be empty"

    result = []
    for i, conv in enumerate(conversations):
        assert conv[0]["role"] != "system", f"Conversation {i} already has a system message"
        new_conv = [{"role": "system", "content": system_prompt}] + conv
        result.append(new_conv)
    return result


def conversations_to_samples(
    conversations: list[Conversation],
    tokenizer: Any,
    max_length: int = 2048,
) -> list[Sample]:
    """Convert conversations to tokenized training Samples.

    Tokenizes each conversation and computes loss masks so that
    loss is only computed on assistant responses (not system/user messages).

    Conversations exceeding max_length are skipped.

    Args:
        conversations: List of conversations
        tokenizer: HuggingFace tokenizer with apply_chat_template
        max_length: Maximum sequence length in tokens

    Returns:
        List of tokenized Samples ready for SFT training
    """
    samples = []
    skipped = 0

    for conv in conversations:
        # Separate system messages from user/assistant for tokenization.
        # tokenize_conversation expects role in {"user", "assistant"} and
        # handles system via the tokenizer's chat template.
        # If there's a system message, we need to include it in the messages
        # passed to tokenize_conversation since apply_chat_template handles it.
        tokens, user_spans = tokenize_conversation(conv, tokenizer, max_length)

        if len(tokens) > max_length:
            skipped += 1
            continue

        loss_mask = compute_loss_mask(tokens, user_spans)

        samples.append(
            Sample(
                prompt=conv,
                tokens=tokens,
                loss_mask=loss_mask,
            )
        )

    if skipped > 0:
        print(
            f"Skipped {skipped}/{len(conversations)} conversations exceeding max_length={max_length}"
        )

    return samples


def save_conversations(conversations: list[Conversation], path: str | Path) -> None:
    """Save conversations to JSONL file.

    Args:
        conversations: List of conversations to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for conv in conversations:
            json.dump({"messages": conv}, f, ensure_ascii=False)
            f.write("\n")
