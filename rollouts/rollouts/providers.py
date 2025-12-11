"""Provider-specific rollout helpers for the agents module.

DEPRECATED: This module is a backwards-compatibility shim.
Import from rollouts.providers instead:

    from rollouts.providers import rollout_anthropic, rollout_openai
"""

from __future__ import annotations

import warnings

# Re-export everything from the new providers package
from rollouts.providers import (
    NonRetryableError,
    VLLMErrorType,
    add_cache_control_to_last_content,
    get_provider_function,
    rollout_anthropic,
    rollout_google,
    rollout_openai,
    rollout_openai_responses,
    rollout_sglang,
    sanitize_request_for_logging,
)

# Also re-export the aggregate functions for direct access
from rollouts.providers.anthropic import aggregate_anthropic_stream
from rollouts.providers.google import aggregate_google_stream
from rollouts.providers.openai_completions import aggregate_stream
from rollouts.providers.openai_responses import aggregate_openai_responses_stream

__all__ = [
    # Provider functions
    "rollout_anthropic",
    "rollout_google",
    "rollout_openai",
    "rollout_openai_responses",
    "rollout_sglang",
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
    "sanitize_request_for_logging",
]
