"""Chat template tokenization with correct delimiter handling.

Pure functions for tokenizing chat messages. No classes.

The problem:
    When concatenating multi-turn tokens, chat templates insert delimiter
    tokens (newlines, role markers) between messages. If you just concatenate
    token lists, you miss these delimiters.

Two solutions:

1. Prefix trick (miles):
    Tokenize [prefix_msg, actual_msg] together, then strip prefix tokens.
    This gives you actual_msg WITH the correct leading delimiter.

2. Cached suffix (verifiers):
    Compute the suffix tokens that chat templates add after assistant messages
    once, then append them after each turn.

Usage:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Tokenize full conversation
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    ids = tokenize_chat(tokenizer, messages)

    # Tokenize incrementally (prefix trick)
    turn1_ids = tokenize_chat(tokenizer, messages[:1])
    turn2_ids = tokenize_message_with_delimiter(tokenizer, messages[1])
    full_ids = turn1_ids + turn2_ids

    # Tokenize incrementally (cached suffix)
    suffix_ids = compute_suffix_ids(tokenizer)
    turn1_ids = tokenize_chat(tokenizer, messages[:1])
    # After generation, append suffix before next turn:
    turn1_with_suffix = turn1_ids + generated_ids + suffix_ids

    # Check for token mismatches (debugging)
    mismatch = check_token_mismatch(our_ids, tokenize_chat(tokenizer, messages))

References:
    - miles/utils/mask_utils.py (prefix trick)
    - verifiers/utils/token_utils.py (cached suffix)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def tokenize_chat(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
) -> list[int]:
    """Tokenize a chat conversation.

    Args:
        tokenizer: HuggingFace tokenizer with chat template
        messages: List of {"role": "...", "content": "..."} dicts
        add_generation_prompt: If True, add assistant prompt at end

    Returns:
        Token IDs for the full conversation
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )


def tokenize_message_with_delimiter(
    tokenizer: PreTrainedTokenizer,
    message: dict[str, str],
) -> list[int]:
    """Tokenize a single message with its leading delimiter.

    Uses the "prefix trick": tokenize [prefix, message] together,
    then strip the prefix. This gives you the message tokens WITH
    the correct delimiter that the chat template inserts.

    Use this when building multi-turn conversations incrementally:
        turn1_ids = tokenize_chat(tokenizer, [msg1])
        turn2_ids = tokenize_message_with_delimiter(tokenizer, msg2)
        full_ids = turn1_ids + turn2_ids

    Args:
        tokenizer: HuggingFace tokenizer with chat template
        message: Single message {"role": "...", "content": "..."}

    Returns:
        Token IDs for message INCLUDING its leading delimiter
    """
    # Dummy prefix message for the trick
    prefix_msg = {"role": "user", "content": "PREFIX_DUMMY_CONTENT"}
    prefix_ids = tokenizer.apply_chat_template([prefix_msg], tokenize=True)

    # Tokenize prefix + actual message together
    combined_ids = tokenizer.apply_chat_template([prefix_msg, message], tokenize=True)

    # Strip prefix to get message with delimiter
    return combined_ids[len(prefix_ids) :]


def build_loss_mask(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
) -> tuple[list[int], list[float]]:
    """Build token IDs and loss mask for training.

    Loss mask is 1.0 for assistant tokens (train on these),
    0.0 for user/system tokens (don't train on these).

    Args:
        tokenizer: HuggingFace tokenizer with chat template
        messages: List of chat messages

    Returns:
        (token_ids, loss_mask) where loss_mask[i] is 1.0 if we train on token i
    """
    if not messages:
        return [], []

    all_ids: list[int] = []
    all_mask: list[float] = []

    for i, message in enumerate(messages):
        if i == 0:
            # First message - tokenize directly
            msg_ids = tokenizer.apply_chat_template([message], tokenize=True)
        else:
            # Subsequent messages - use prefix trick for correct delimiter
            msg_ids = tokenize_message_with_delimiter(tokenizer, message)

        # Mask: 1.0 for assistant, 0.0 for others
        if message["role"] == "assistant":
            msg_mask = [1.0] * len(msg_ids)
        else:
            msg_mask = [0.0] * len(msg_ids)

        all_ids.extend(msg_ids)
        all_mask.extend(msg_mask)

    return all_ids, all_mask


# ============================================================================
# Cached suffix approach (verifiers style)
# ============================================================================


def compute_suffix_ids(tokenizer: PreTrainedTokenizer) -> list[int]:
    """Compute suffix tokens that chat templates add after assistant messages.

    Chat templates often add tokens after the assistant's content that aren't
    generated by the model (e.g., <|eom|>, newlines). This function computes
    those suffix tokens once so they can be appended after each turn.

    Usage:
        suffix_ids = compute_suffix_ids(tokenizer)

        # After each turn, before adding next message:
        full_ids = prompt_ids + generated_ids + suffix_ids + next_turn_ids

    Args:
        tokenizer: HuggingFace tokenizer with chat template

    Returns:
        Suffix token IDs to append after assistant completions

    Reference: verifiers/utils/token_utils.py
    """
    # Use dummy messages to find what tokens the template adds after assistant content
    dummy_content = "DUMMY_SUFFIX_CONTENT"
    dummy_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": dummy_content},
    ]

    # Tokenize just the content
    dummy_content_ids = tokenizer.encode(dummy_content, add_special_tokens=False)

    # Tokenize full conversation (without generation prompt)
    full_ids = tokenizer.apply_chat_template(
        dummy_messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    # Find suffix: everything after the last occurrence of the final content token
    last_content_token = dummy_content_ids[-1]
    suffix_ids = _extract_suffix_after_token(full_ids, last_content_token)

    return suffix_ids


def _extract_suffix_after_token(token_ids: list[int], target_token: int) -> list[int]:
    """Extract all tokens after the last occurrence of target_token."""
    # Find last occurrence
    last_idx = -1
    for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i] == target_token:
            last_idx = i
            break

    if last_idx == -1:
        # Token not found - no suffix
        return []

    return token_ids[last_idx + 1 :]


def find_largest_overlap(a: list[int], b: list[int]) -> int:
    """Find largest overlap between end of a and beginning of b.

    Used when model output may already include some suffix tokens
    (e.g., truncated turns). We find the overlap to avoid duplicating tokens.

    Args:
        a: First token list (check end)
        b: Second token list (check beginning)

    Returns:
        Length of the largest overlapping sequence

    Example:
        a = [1, 2, 3, 4, 5]
        b = [4, 5, 6, 7]
        find_largest_overlap(a, b) -> 2  # [4, 5] overlaps
    """
    if not a or not b:
        return 0

    max_possible = min(len(a), len(b))
    for overlap_len in range(max_possible, 0, -1):
        if a[-overlap_len:] == b[:overlap_len]:
            return overlap_len

    return 0


def append_suffix_with_overlap(
    token_ids: list[int],
    suffix_ids: list[int],
) -> list[int]:
    """Append suffix tokens, handling any overlap.

    If the model already generated some suffix tokens, we don't want to
    duplicate them. This finds any overlap and only appends the remainder.

    Args:
        token_ids: Existing token IDs (e.g., prompt + generated)
        suffix_ids: Suffix tokens to append

    Returns:
        token_ids with suffix appended (minus any overlap)
    """
    overlap = find_largest_overlap(token_ids, suffix_ids)
    return token_ids + suffix_ids[overlap:]


# ============================================================================
# Token mismatch detection (for debugging)
# ============================================================================


def check_token_mismatch(
    our_ids: list[int],
    reference_ids: list[int],
    tokenizer: PreTrainedTokenizer | None = None,
) -> dict:
    """Check for mismatches between our tokens and reference tokens.

    Used to detect when our TI/TO tokens differ from what the chat template
    would produce. Mismatches can cause training instability.

    Args:
        our_ids: Token IDs we computed (TI/TO)
        reference_ids: Token IDs from tokenize_chat() (ground truth)
        tokenizer: Optional tokenizer for decoding in error messages

    Returns:
        Dict with:
            - match: bool - True if tokens match exactly
            - our_len: int - Length of our tokens
            - ref_len: int - Length of reference tokens
            - first_mismatch_idx: int | None - Index of first mismatch
            - mismatches: list[dict] - Details of first few mismatches
    """
    result = {
        "match": our_ids == reference_ids,
        "our_len": len(our_ids),
        "ref_len": len(reference_ids),
        "first_mismatch_idx": None,
        "mismatches": [],
    }

    if result["match"]:
        return result

    # Find mismatches
    min_len = min(len(our_ids), len(reference_ids))
    for i in range(min_len):
        if our_ids[i] != reference_ids[i]:
            if result["first_mismatch_idx"] is None:
                result["first_mismatch_idx"] = i

            mismatch = {
                "idx": i,
                "our_token": our_ids[i],
                "ref_token": reference_ids[i],
            }

            if tokenizer is not None:
                mismatch["our_decoded"] = tokenizer.decode([our_ids[i]])
                mismatch["ref_decoded"] = tokenizer.decode([reference_ids[i]])

            result["mismatches"].append(mismatch)

            # Only report first 5 mismatches
            if len(result["mismatches"]) >= 5:
                break

    # Length mismatch
    if len(our_ids) != len(reference_ids) and result["first_mismatch_idx"] is None:
        result["first_mismatch_idx"] = min_len

    return result


def log_token_mismatch(
    our_ids: list[int],
    reference_ids: list[int],
    tokenizer: PreTrainedTokenizer,
    context: str = "",
) -> bool:
    """Log a warning if tokens don't match, like Prime's PR #1422.

    Args:
        our_ids: Token IDs we computed
        reference_ids: Token IDs from chat template
        tokenizer: Tokenizer for decoding
        context: Optional context string for the log message

    Returns:
        True if tokens match, False if mismatch detected
    """
    import logging

    logger = logging.getLogger(__name__)

    mismatch = check_token_mismatch(our_ids, reference_ids, tokenizer)

    if mismatch["match"]:
        return True

    ctx = f" ({context})" if context else ""
    logger.warning(
        f"Token mismatch detected{ctx}. "
        f"Our tokens ({mismatch['our_len']}) != reference ({mismatch['ref_len']}). "
        f"First mismatch at index {mismatch['first_mismatch_idx']}. "
        "This may happen due to retokenization discrepancies in multi-turn conversations. "
        "Using TI/TO tokens to avoid training instability."
    )

    if mismatch["mismatches"]:
        logger.debug(f"First mismatches: {mismatch['mismatches']}")

    return False
