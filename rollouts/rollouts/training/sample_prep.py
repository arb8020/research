"""SFT sample preparation

Transform rollouts into tokenized training samples with loss masks.

Tiger Style: Pure functions, explicit transformations.
Tinker: Token-level loss weights.
"""

from typing import List, Dict, Any
from rollouts.dtypes import Trajectory, Message
from rollouts.training.types import Sample


# ────────────────────── Core Transform ──────────────────────


def trajectory_to_sample(
    trajectory: Trajectory,
    tokenizer: Any,  # HuggingFace tokenizer
    loss_on_assistant_only: bool = True,
    source: str = "sft",
) -> Sample:
    """Convert trajectory to training sample

    Args:
        trajectory: Rollout trajectory with messages
        tokenizer: HuggingFace tokenizer
        loss_on_assistant_only: Only compute loss on assistant tokens
        source: "sft" or "rollout"

    Returns:
        Sample with input_ids, labels, loss_mask

    Tiger Style: Explicit assertions, clear control flow.
    """
    assert trajectory is not None, "trajectory required"
    assert tokenizer is not None, "tokenizer required"
    assert len(trajectory.messages) > 0, "trajectory has no messages"

    # Apply chat template (HuggingFace)
    # This formats messages into text with special tokens
    text = tokenizer.apply_chat_template(
        [msg_to_dict(m) for m in trajectory.messages],
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=True)

    # Compute loss mask
    loss_mask = compute_loss_mask(
        trajectory=trajectory,
        tokens=tokens,
        tokenizer=tokenizer,
        loss_on_assistant_only=loss_on_assistant_only,
    )

    # Create sample (Tiger: explicit construction)
    sample = Sample(
        input_ids=tokens,
        labels=tokens,  # Same as input_ids, mask controls loss
        loss_mask=loss_mask,
        source=source,
        metadata={
            "num_messages": len(trajectory.messages),
            "num_tokens": len(tokens),
        },
    )

    return sample


# ────────────────────── Loss Mask Computation ──────────────────────


def compute_loss_mask(
    trajectory: Trajectory,
    tokens: List[int],
    tokenizer: Any,
    loss_on_assistant_only: bool = True,
) -> List[float]:
    """Compute loss mask for tokens (Tinker-inspired)

    Args:
        trajectory: Original trajectory (for role info)
        tokens: Tokenized sequence
        tokenizer: HuggingFace tokenizer
        loss_on_assistant_only: Only train on assistant tokens

    Returns:
        List of loss weights (0.0 = no loss, 1.0 = full loss)

    Tiger Style: Explicit masking logic, clear boundaries.
    Tinker: Token-level weights enable fine-grained control.
    """
    assert len(tokens) > 0, "tokens required"

    if not loss_on_assistant_only:
        # Train on everything
        return [1.0] * len(tokens)

    # Find assistant token spans
    # Strategy: Re-tokenize each message to find boundaries
    loss_mask = [0.0] * len(tokens)
    current_pos = 0

    for msg in trajectory.messages:
        # Tokenize this message
        msg_text = tokenizer.apply_chat_template(
            [msg_to_dict(msg)],
            tokenize=False,
            add_generation_prompt=False,
        )
        msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
        msg_len = len(msg_tokens)

        # If assistant message, mark tokens for loss
        if msg.role == "assistant":
            for i in range(current_pos, min(current_pos + msg_len, len(tokens))):
                loss_mask[i] = 1.0

        current_pos += msg_len

        # Safety check (Tiger style)
        if current_pos > len(tokens):
            break

    return loss_mask


# ────────────────────── Helpers ──────────────────────


def msg_to_dict(msg: Message) -> Dict[str, Any]:
    """Convert Message to dict for HuggingFace tokenizer

    Tiger Style: Explicit conversion, no hidden logic.
    """
    return {
        "role": msg.role,
        "content": msg.content or "",
    }


# ────────────────────── Batch Preparation ──────────────────────


def prepare_sft_batch(
    samples: List[Sample],
    max_length: int = 2048,
    pad_token_id: int = 0,
) -> Dict[str, List[List[int]]]:
    """Prepare batch for training (with padding)

    Args:
        samples: List of samples
        max_length: Max sequence length
        pad_token_id: Token to use for padding

    Returns:
        Batch dict with input_ids, labels, attention_mask

    Tiger Style: Explicit padding, bounded sequences.
    """
    assert len(samples) > 0, "samples required"
    assert max_length > 0, "max_length must be positive"

    input_ids = []
    labels = []
    attention_mask = []

    for sample in samples:
        # Truncate if needed (Tiger: bounded!)
        seq_len = min(len(sample.input_ids), max_length)

        # Get sequences
        ids = sample.input_ids[:seq_len]
        lbls = sample.labels[:seq_len]
        mask = sample.loss_mask[:seq_len]

        # Pad to max_length
        padding_len = max_length - seq_len
        ids_padded = ids + [pad_token_id] * padding_len
        lbls_padded = lbls + [-100] * padding_len  # -100 = ignore in loss
        attn_mask = [1] * seq_len + [0] * padding_len

        # Apply loss mask to labels (Tinker: token-level control!)
        lbls_masked = [
            lbl if mask[i] > 0.0 else -100
            for i, lbl in enumerate(lbls)
        ] + [-100] * padding_len

        input_ids.append(ids_padded)
        labels.append(lbls_masked)
        attention_mask.append(attn_mask)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
