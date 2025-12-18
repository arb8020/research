"""Chat template tokenization with correct delimiter handling.

The problem: When concatenating multi-turn tokens, chat templates insert
delimiter tokens (newlines, role markers) between messages. If you just
concatenate token lists, you miss these delimiters.

Example:
    Turn 1: user → assistant
    Turn 2: want user || assistant || user (with correct delimiters)

This module provides two approaches:
1. Prefix trick (miles): Tokenize [prefix, msg], strip prefix to get msg with delimiter
2. Cached suffix (prime/verifiers): Pre-compute delimiter tokens per role transition

Reference:
- miles/utils/mask_utils.py (prefix trick)
- verifiers PR #626 (cached suffix)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class ChatTemplateTokenizer:
    """Tokenize chat messages with correct delimiter handling.

    Uses the "prefix trick" to get correct delimiter tokens when
    tokenizing individual messages for multi-turn concatenation.

    The trick: tokenize [prefix_msg, actual_msg] together, then strip
    the prefix tokens. This gives you actual_msg tokens WITH the
    correct leading delimiter that the chat template would insert.

    Args:
        tokenizer: HuggingFace tokenizer with chat template.
    """

    tokenizer: PreTrainedTokenizer
    _prefix_message: dict[str, str] = field(init=False, repr=False)
    _prefix_ids: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Use a dummy message as prefix for the trick
        self._prefix_message = {"role": "user", "content": "PREFIX_FOR_DELIMITER"}
        self._prefix_ids = self.tokenizer.apply_chat_template(
            [self._prefix_message],
            tokenize=True,
            add_generation_prompt=False,
        )

    def tokenize_message(
        self,
        message: dict[str, str],
        is_first: bool = False,
    ) -> list[int]:
        """Tokenize a single message with correct delimiters.

        Args:
            message: Chat message {"role": "...", "content": "..."}.
            is_first: If True, this is the first message (include BOS/system tokens).

        Returns:
            Token IDs for this message, including leading delimiter if not first.
        """
        if is_first:
            # First message - tokenize directly, includes BOS and any system prefix
            return self.tokenizer.apply_chat_template(
                [message],
                tokenize=True,
                add_generation_prompt=False,
            )

        # Not first - use prefix trick to get correct delimiter
        prefixed_ids = self.tokenizer.apply_chat_template(
            [self._prefix_message, message],
            tokenize=True,
            add_generation_prompt=False,
        )
        # Strip prefix to get message with its delimiter
        return prefixed_ids[len(self._prefix_ids) :]

    def tokenize_messages(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Tokenize a list of messages with correct delimiters.

        Args:
            messages: List of chat messages.
            add_generation_prompt: If True, add assistant prompt at end.

        Returns:
            Token IDs for all messages concatenated correctly.
        """
        if not messages:
            return []

        # For full message list, just use apply_chat_template directly
        # The prefix trick is for when you need to tokenize incrementally
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )

    def tokenize_incremental(
        self,
        existing_ids: list[int],
        new_message: dict[str, str],
    ) -> tuple[list[int], list[int]]:
        """Add a new message to existing token sequence.

        This is for multi-turn generation where you have:
        - existing_ids: tokens from previous turns
        - new_message: the new message to add

        Returns:
            (full_ids, new_message_ids): Full sequence and just the new tokens.
        """
        # Use prefix trick - the new message needs correct delimiter
        new_message_ids = self.tokenize_message(new_message, is_first=False)
        full_ids = existing_ids + new_message_ids
        return full_ids, new_message_ids

    def build_loss_mask(
        self,
        messages: list[dict[str, str]],
        train_on_assistant_only: bool = True,
    ) -> tuple[list[int], list[float]]:
        """Build token IDs and loss mask for training.

        Args:
            messages: List of chat messages.
            train_on_assistant_only: If True, only compute loss on assistant tokens.

        Returns:
            (token_ids, loss_mask): Token IDs and per-token loss mask (1.0 = train).
        """
        all_ids: list[int] = []
        all_mask: list[float] = []

        for i, message in enumerate(messages):
            msg_ids = self.tokenize_message(message, is_first=(i == 0))

            if train_on_assistant_only:
                if message["role"] == "assistant":
                    # Train on assistant content (but not role/delimiter tokens)
                    # Approximate: mark all as trainable, refinement possible
                    msg_mask = [1.0] * len(msg_ids)
                else:
                    msg_mask = [0.0] * len(msg_ids)
            else:
                msg_mask = [1.0] * len(msg_ids)

            all_ids.extend(msg_ids)
            all_mask.extend(msg_mask)

        return all_ids, all_mask


@dataclass
class CachedSuffixTokenizer:
    """Alternative approach: pre-compute delimiter tokens per role transition.

    Instead of the prefix trick, this caches the delimiter tokens that
    the chat template inserts between each role pair (user→assistant,
    assistant→user, etc.).

    Reference: verifiers PR #626
    """

    tokenizer: PreTrainedTokenizer
    _suffix_cache: dict[tuple[str, str], list[int]] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self._precompute_suffixes()

    def _precompute_suffixes(self) -> None:
        """Compute delimiter tokens for all role transitions."""
        roles = ["user", "assistant", "system", "tool"]

        for from_role in roles:
            for to_role in roles:
                if from_role == to_role == "system":
                    continue  # Skip system→system

                # Use dummy messages to find delimiter tokens
                dummy_content = "DUMMY_CONTENT_FOR_DELIMITER"
                dummy_msgs = [
                    {"role": from_role, "content": dummy_content},
                    {"role": to_role, "content": dummy_content},
                ]

                try:
                    full_ids = self.tokenizer.apply_chat_template(
                        dummy_msgs,
                        tokenize=True,
                        add_generation_prompt=False,
                    )

                    # Find where the second dummy content starts
                    single_msg_ids = self.tokenizer.apply_chat_template(
                        [dummy_msgs[0]],
                        tokenize=True,
                        add_generation_prompt=False,
                    )

                    # The delimiter is what's between first msg and second content
                    content_ids = self.tokenizer.encode(dummy_content, add_special_tokens=False)

                    # Find second occurrence of content in full_ids
                    # Delimiter is between end of first msg and start of second content
                    first_end = len(single_msg_ids)
                    # Search for content_ids starting after first message
                    for i in range(first_end, len(full_ids) - len(content_ids) + 1):
                        if full_ids[i : i + len(content_ids)] == content_ids:
                            delimiter_ids = full_ids[first_end:i]
                            self._suffix_cache[(from_role, to_role)] = delimiter_ids
                            break

                except Exception:
                    # Some role transitions may not be valid for all templates
                    pass

    def get_delimiter(self, from_role: str, to_role: str) -> list[int]:
        """Get delimiter tokens for a role transition."""
        return self._suffix_cache.get((from_role, to_role), [])

    def concatenate_turns(
        self,
        turns: list[tuple[str, list[int]]],
    ) -> list[int]:
        """Concatenate turns with correct delimiters.

        Args:
            turns: List of (role, token_ids) tuples.

        Returns:
            Full token sequence with delimiters inserted.
        """
        if not turns:
            return []

        result = list(turns[0][1])  # First turn

        for i in range(1, len(turns)):
            from_role = turns[i - 1][0]
            to_role = turns[i][0]
            delimiter = self.get_delimiter(from_role, to_role)
            result = result + delimiter + list(turns[i][1])

        return result
