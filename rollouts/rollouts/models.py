"""Model registry for provider abstraction.

Centralized model metadata including capabilities, costs, context windows, and API mappings.
Inspired by pi-ai's model discovery system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# API type categorization - maps providers to their API interface types
ApiType = Literal[
    "openai-completions",  # OpenAI chat completions, Groq, Cerebras, xAI, vLLM/sglang
    "openai-responses",  # OpenAI responses API (o1, o3 reasoning models)
    "anthropic-messages",  # Anthropic Claude messages API
    "google-generative-ai",  # Google Gemini API
]

# Known provider types
Provider = Literal[
    "openai",
    "anthropic",
    "google",
    "groq",
    "cerebras",
    "xai",
    "openrouter",
    "sglang",
    "vllm",
]


@dataclass(frozen=True)
class ModelCost:
    """Pricing per million tokens"""

    input: float  # Per million input tokens
    output: float  # Per million output tokens
    cache_read: float  # Per million cache read tokens (if supported)
    cache_write: float  # Per million cache write tokens (if supported)


@dataclass(frozen=True)
class ModelMetadata:
    """Complete model metadata for provider abstraction"""

    id: str  # Model identifier (e.g., "gpt-4", "claude-3-5-sonnet-20241022")
    name: str  # Human-readable name
    provider: Provider  # Provider hosting the model
    api: ApiType  # API interface type
    base_url: str  # Default API base URL
    reasoning: bool  # Supports extended thinking/reasoning
    input_types: list[Literal["text", "image"]]  # Supported input modalities
    cost: ModelCost  # Pricing information
    context_window: int  # Maximum context length in tokens
    max_tokens: int  # Maximum output tokens


# Model registry - organized by provider then model_id
# Following pi-ai pattern but using Python data structures

MODELS: dict[Provider, dict[str, ModelMetadata]] = {
    "openai": {
        "gpt-4o": ModelMetadata(
            id="gpt-4o",
            name="GPT-4o",
            provider="openai",
            api="openai-completions",
            base_url="https://api.openai.com/v1",
            reasoning=False,
            input_types=["text", "image"],
            cost=ModelCost(input=2.5, output=10.0, cache_read=1.25, cache_write=1.25),
            context_window=128000,
            max_tokens=16384,
        ),
        "gpt-4o-mini": ModelMetadata(
            id="gpt-4o-mini",
            name="GPT-4o Mini",
            provider="openai",
            api="openai-completions",
            base_url="https://api.openai.com/v1",
            reasoning=False,
            input_types=["text", "image"],
            cost=ModelCost(input=0.15, output=0.6, cache_read=0.075, cache_write=0.075),
            context_window=128000,
            max_tokens=16384,
        ),
        "gpt-4.1": ModelMetadata(
            id="gpt-4.1",
            name="GPT-4.1",
            provider="openai",
            api="openai-responses",
            base_url="https://api.openai.com/v1",
            reasoning=False,
            input_types=["text", "image"],
            cost=ModelCost(input=2.5, output=10.0, cache_read=1.25, cache_write=1.25),
            context_window=128000,
            max_tokens=16384,
        ),
        "gpt-5.1-codex": ModelMetadata(
            id="gpt-5.1-codex",
            name="GPT-5.1 Codex",
            provider="openai",
            api="openai-responses",
            base_url="https://api.openai.com/v1",
            reasoning=True,  # GPT-5 Codex is a reasoning model
            input_types=["text", "image"],
            cost=ModelCost(input=5.0, output=15.0, cache_read=1.25, cache_write=1.25),
            context_window=128000,
            max_tokens=16384,
        ),
        "gpt-5.1-codex-mini": ModelMetadata(
            id="gpt-5.1-codex-mini",
            name="GPT-5.1 Codex Mini",
            provider="openai",
            api="openai-responses",
            base_url="https://api.openai.com/v1",
            reasoning=True,  # GPT-5 Codex Mini is a reasoning model
            input_types=["text", "image"],
            cost=ModelCost(input=1.0, output=3.0, cache_read=0.25, cache_write=0.25),
            context_window=128000,
            max_tokens=16384,
        ),
        "gpt-5.2-2025-12-11": ModelMetadata(
            id="gpt-5.2-2025-12-11",
            name="GPT-5.2",
            provider="openai",
            api="openai-responses",
            base_url="https://api.openai.com/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=5.0, output=15.0, cache_read=1.25, cache_write=1.25),
            context_window=400000,
            max_tokens=32768,
        ),
        "o1": ModelMetadata(
            id="o1",
            name="o1",
            provider="openai",
            api="openai-completions",  # o1 uses chat completions, not responses
            base_url="https://api.openai.com/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=15.0, output=60.0, cache_read=7.5, cache_write=7.5),
            context_window=200000,
            max_tokens=100000,
        ),
        "o1-mini": ModelMetadata(
            id="o1-mini",
            name="o1-mini",
            provider="openai",
            api="openai-completions",  # o1-mini uses chat completions, not responses
            base_url="https://api.openai.com/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=3.0, output=12.0, cache_read=1.5, cache_write=1.5),
            context_window=128000,
            max_tokens=65536,
        ),
    },
    "anthropic": {
        # Claude 4.5 family
        "claude-opus-4-5-20251101": ModelMetadata(
            id="claude-opus-4-5-20251101",
            name="Claude Opus 4.5",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75),
            context_window=200000,
            max_tokens=8192,
        ),
        "claude-sonnet-4-5-20250929": ModelMetadata(
            id="claude-sonnet-4-5-20250929",
            name="Claude Sonnet 4.5",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
            context_window=200000,
            max_tokens=8192,
        ),
        "claude-haiku-4-5-20251001": ModelMetadata(
            id="claude-haiku-4-5-20251001",
            name="Claude Haiku 4.5",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=0.8, output=4.0, cache_read=0.08, cache_write=1.0),
            context_window=200000,
            max_tokens=8192,
        ),
        # Claude 4.1 family
        "claude-opus-4-1-20250805": ModelMetadata(
            id="claude-opus-4-1-20250805",
            name="Claude Opus 4.1",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75),
            context_window=200000,
            max_tokens=8192,
        ),
        # Claude 4 family
        "claude-opus-4-20250514": ModelMetadata(
            id="claude-opus-4-20250514",
            name="Claude Opus 4",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75),
            context_window=200000,
            max_tokens=8192,
        ),
        "claude-sonnet-4-20250514": ModelMetadata(
            id="claude-sonnet-4-20250514",
            name="Claude Sonnet 4",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
            context_window=200000,
            max_tokens=8192,
        ),
        # Claude 3.7 family
        "claude-3-7-sonnet-20250219": ModelMetadata(
            id="claude-3-7-sonnet-20250219",
            name="Claude Sonnet 3.7",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
            context_window=200000,
            max_tokens=8192,
        ),
        # Claude 3.5 family
        "claude-3-5-sonnet-20241022": ModelMetadata(
            id="claude-3-5-sonnet-20241022",
            name="Claude 3.5 Sonnet",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
            context_window=200000,
            max_tokens=8192,
        ),
        "claude-3-5-haiku-20241022": ModelMetadata(
            id="claude-3-5-haiku-20241022",
            name="Claude 3.5 Haiku",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=False,
            input_types=["text", "image"],
            cost=ModelCost(input=0.8, output=4.0, cache_read=0.08, cache_write=1.0),
            context_window=200000,
            max_tokens=8192,
        ),
        # Claude 3 family (legacy)
        "claude-3-opus-20240229": ModelMetadata(
            id="claude-3-opus-20240229",
            name="Claude 3 Opus",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=False,
            input_types=["text", "image"],
            cost=ModelCost(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75),
            context_window=200000,
            max_tokens=4096,
        ),
        "claude-3-haiku-20240307": ModelMetadata(
            id="claude-3-haiku-20240307",
            name="Claude 3 Haiku",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=False,
            input_types=["text"],
            cost=ModelCost(input=0.25, output=1.25, cache_read=0.03, cache_write=0.30),
            context_window=200000,
            max_tokens=4096,
        ),
    },
    "groq": {
        "llama-3.3-70b-versatile": ModelMetadata(
            id="llama-3.3-70b-versatile",
            name="Llama 3.3 70B",
            provider="groq",
            api="openai-completions",
            base_url="https://api.groq.com/openai/v1",
            reasoning=False,
            input_types=["text"],
            cost=ModelCost(input=0.59, output=0.79, cache_read=0.0, cache_write=0.0),
            context_window=128000,
            max_tokens=32768,
        ),
    },
    "google": {
        "gemini-2.0-flash-exp": ModelMetadata(
            id="gemini-2.0-flash-exp",
            name="Gemini 2.0 Flash Experimental",
            provider="google",
            api="google-generative-ai",
            base_url="https://generativelanguage.googleapis.com",
            reasoning=True,  # Supports thinking
            input_types=["text", "image"],
            cost=ModelCost(
                input=0.0, output=0.0, cache_read=0.0, cache_write=0.0
            ),  # Free during preview
            context_window=1000000,
            max_tokens=8192,
        ),
        "gemini-1.5-pro": ModelMetadata(
            id="gemini-1.5-pro",
            name="Gemini 1.5 Pro",
            provider="google",
            api="google-generative-ai",
            base_url="https://generativelanguage.googleapis.com",
            reasoning=False,
            input_types=["text", "image"],
            cost=ModelCost(input=1.25, output=5.0, cache_read=0.3125, cache_write=1.25),
            context_window=2000000,
            max_tokens=8192,
        ),
        "gemini-1.5-flash": ModelMetadata(
            id="gemini-1.5-flash",
            name="Gemini 1.5 Flash",
            provider="google",
            api="google-generative-ai",
            base_url="https://generativelanguage.googleapis.com",
            reasoning=False,
            input_types=["text", "image"],
            cost=ModelCost(input=0.075, output=0.30, cache_read=0.01875, cache_write=0.075),
            context_window=1000000,
            max_tokens=8192,
        ),
    },
    "sglang": {},  # vLLM/sglang uses custom endpoints, populated at runtime
    "vllm": {},  # Same as sglang
}


# Provider-to-API type mapping
# Maps provider strings to their API interface type
PROVIDER_API_MAP: dict[str, ApiType] = {
    # OpenAI completions API (chat/completions endpoint)
    "openai": "openai-completions",  # Default for non-reasoning models
    "groq": "openai-completions",
    "cerebras": "openai-completions",
    "xai": "openai-completions",
    "openrouter": "openai-completions",
    "sglang": "openai-completions",
    "vllm": "openai-completions",
    # Anthropic messages API
    "anthropic": "anthropic-messages",
    # Google generative AI
    "google": "google-generative-ai",
}


def get_api_type(provider: str, model_id: str | None = None) -> ApiType:
    """Get the API type for a provider/model combination.

    Args:
        provider: Provider identifier (e.g., "openai", "anthropic")
        model_id: Optional model ID. Checked against model registry for explicit API type.

    Returns:
        API type string

    Raises:
        AssertionError: If provider is not recognized

    Logic:
    1. Check model registry first for explicit API type
    2. Fall back to provider default from PROVIDER_API_MAP
    """
    # Check model registry first if we have a model_id
    if model_id and provider in MODELS:
        model = MODELS[provider].get(model_id)
        if model:
            return model.api

    # Get provider mapping - crash loud if unknown
    api_type = PROVIDER_API_MAP.get(provider)
    assert api_type is not None, (
        f"Unknown provider: {provider}\n"
        f"Supported providers: {list(PROVIDER_API_MAP.keys())}\n"
        f"If you're using a custom provider, add it to PROVIDER_API_MAP in models.py"
    )
    return api_type


# Model registry initialization
_model_registry: dict[Provider, dict[str, ModelMetadata]] = {}


def _initialize_registry() -> None:
    """Initialize the model registry from MODELS constant"""
    global _model_registry
    _model_registry = {provider: dict(models) for provider, models in MODELS.items()}


def get_providers() -> list[Provider]:
    """Get all available providers"""
    if not _model_registry:
        _initialize_registry()
    return list(_model_registry.keys())


def get_models(provider: Provider) -> list[ModelMetadata]:
    """Get all models for a given provider"""
    if not _model_registry:
        _initialize_registry()

    provider_models = _model_registry.get(provider, {})
    return list(provider_models.values())


def get_model(provider: Provider, model_id: str) -> ModelMetadata | None:
    """Get a specific model by provider and ID

    Returns None if model not found.
    For custom/runtime models (vLLM, sglang), returns None - caller should create metadata.
    """
    if not _model_registry:
        _initialize_registry()

    provider_models = _model_registry.get(provider, {})
    return provider_models.get(model_id)


def register_model(metadata: ModelMetadata) -> None:
    """Register a new model at runtime (useful for custom vLLM/sglang endpoints)"""
    if not _model_registry:
        _initialize_registry()

    if metadata.provider not in _model_registry:
        _model_registry[metadata.provider] = {}

    _model_registry[metadata.provider][metadata.id] = metadata


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost: ModelCost | None = None,
) -> float:
    """Calculate total cost based on token usage and model pricing

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cache_read_tokens: Number of cache read tokens (if supported)
        cache_write_tokens: Number of cache write tokens (if supported)
        cost: ModelCost to use for calculation. If None, returns 0.0

    Returns:
        Total cost in USD
    """
    if cost is None:
        return 0.0

    total = (
        (input_tokens / 1_000_000) * cost.input
        + (output_tokens / 1_000_000) * cost.output
        + (cache_read_tokens / 1_000_000) * cost.cache_read
        + (cache_write_tokens / 1_000_000) * cost.cache_write
    )

    return total


# Initialize registry on module import
_initialize_registry()
