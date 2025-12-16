"""Provider implementations for different LLM APIs.

This module provides streaming implementations for:
- Anthropic Claude (anthropic-messages)
- OpenAI Chat Completions (openai-completions)
- OpenAI Responses API (openai-responses) - for o1/o3 reasoning models
- Google Generative AI (google-generative-ai) - for Gemini models
- vLLM/SGLang (sglang)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from .anthropic import aggregate_anthropic_stream, rollout_anthropic
from .base import (
    NonRetryableError,
    VLLMErrorType,
    add_cache_control_to_last_content,
    calculate_cost_from_usage,
    sanitize_request_for_logging,
)
from .google import aggregate_google_stream, rollout_google
from .openai_completions import aggregate_stream, rollout_openai
from .openai_responses import aggregate_openai_responses_stream, rollout_openai_responses
from .sglang import rollout_sglang, rollout_sglang_streaming, rollout_sglang_token_level

if TYPE_CHECKING:
    from rollouts.dtypes import Actor, StreamEvent

# Type alias for provider streaming functions
# Note: Actual functions may accept additional kwargs (user_message_for_thinking, turn_idx, etc.)
# but Callable doesn't support **kwargs, so we use a minimal signature
ProviderStreamFunction = Callable[
    ...,  # Accept any arguments - actual functions have varying signatures
    Awaitable["Actor"],
]

# Registry mapping API types to provider functions
_PROVIDER_REGISTRY: dict[str, ProviderStreamFunction] = {
    "openai-completions": rollout_openai,
    "openai-responses": rollout_openai_responses,
    "anthropic-messages": rollout_anthropic,
    "google-generative-ai": rollout_google,
}


def get_provider_function(provider: str, model_id: str | None = None) -> ProviderStreamFunction:
    """Get the streaming function for a provider/model combination.

    Args:
        provider: Provider name (e.g., 'anthropic', 'openai')
        model_id: Optional model ID for provider-specific routing

    Returns:
        Async function that streams completions from the provider
    """
    from rollouts.models import get_api_type

    api_type = get_api_type(provider, model_id)
    func = _PROVIDER_REGISTRY.get(api_type)
    assert func is not None, f"No provider for API type: {api_type}"
    return func


__all__ = [
    # Provider functions
    "rollout_anthropic",
    "rollout_google",
    "rollout_openai",
    "rollout_openai_responses",
    "rollout_sglang",
    "rollout_sglang_streaming",
    "rollout_sglang_token_level",
    # Aggregate functions
    "aggregate_anthropic_stream",
    "aggregate_google_stream",
    "aggregate_stream",
    "aggregate_openai_responses_stream",
    # Registry
    "get_provider_function",
    # Utilities
    "NonRetryableError",
    "VLLMErrorType",
    "add_cache_control_to_last_content",
    "calculate_cost_from_usage",
    "sanitize_request_for_logging",
    "StreamEvent",
]
