"""Prompt formatting utilities.

Pure functions for converting PromptTemplate + sample -> LLM messages.
"""

from typing import Any

from ..dtypes import Message
from .types import PromptTemplate


def format_prompt(
    template: PromptTemplate,
    sample: dict[str, Any],
) -> list[Message]:
    """Convert template + sample to LLM messages.

    Pure function: no side effects, deterministic output.

    Args:
        template: PromptTemplate with system prompt and user template
        sample: Dict with fields matching {placeholders} in user_template

    Returns:
        List of Message objects ready for LLM call

    Example:
        >>> template = PromptTemplate(
        ...     system="You are a classifier.",
        ...     user_template="Classify: {text}",
        ... )
        >>> sample = {"text": "Hello world"}
        >>> messages = format_prompt(template, sample)
        >>> assert messages[0].role == "system"
        >>> assert messages[1].role == "user"
        >>> assert "Hello world" in messages[1].content
    """
    messages = []

    # System message
    if template.system:
        messages.append(Message(role="system", content=template.system))

    # Few-shot examples (if any)
    for example in template.few_shot_examples:
        messages.append(Message(role="user", content=example.input))
        messages.append(Message(role="assistant", content=example.output))

    # User message with sample fields
    user_content = template.format_user(sample)
    messages.append(Message(role="user", content=user_content))

    return messages
