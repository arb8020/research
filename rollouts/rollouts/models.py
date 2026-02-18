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
    "opencode",  # OpenCode Zen - unified API for multiple providers
    "sglang",
    "vllm",
    "moonshot",  # Kimi models (native)
    "zhipu",  # GLM models (native)
    "fireworks",  # Fireworks AI - fastest US inference
    "together",  # Together AI - dedicated US endpoints
    "bedrock",  # AWS Bedrock - fully managed
]


@dataclass(frozen=True)
class ModelCost:
    """Pricing per million tokens"""

    input: float  # Per million input tokens
    output: float  # Per million output tokens
    cache_read: float  # Per million cache read tokens (if supported)
    cache_write: float  # Per million cache write tokens (if supported)


# Format for enabling thinking/reasoning on different providers
# - "openai": uses reasoning_effort param (OpenAI o1/o3 models)
# - "zai": uses thinking: { type: "enabled" | "disabled" } (Z.ai/GLM models)
# - "qwen": uses enable_thinking: boolean (Qwen models)
# - None: model emits reasoning by default or doesn't support thinking toggle
ThinkingFormat = Literal["openai", "zai", "qwen"] | None


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
    thinking_format: ThinkingFormat = None  # How to enable/disable thinking


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
            max_tokens=64000,
        ),
        "claude-3-haiku-20240307": ModelMetadata(
            id="claude-3-haiku-20240307",
            name="Claude 3 Haiku",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=False,
            input_types=["text"],
            cost=ModelCost(input=0.25, output=1.25, cache_read=0.03, cache_write=0.3),
            context_window=200000,
            max_tokens=4096,
        ),
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
        "claude-haiku-4-5-20251001": ModelMetadata(
            id="claude-haiku-4-5-20251001",
            name="Claude Haiku 4.5",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=1.0, output=5.0, cache_read=0.08, cache_write=1.0),
            context_window=200000,
            max_tokens=64000,
        ),
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
            max_tokens=32000,
        ),
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
            max_tokens=32000,
        ),
        "claude-opus-4-5-20251101": ModelMetadata(
            id="claude-opus-4-5-20251101",
            name="Claude Opus 4.5",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=5.0, output=25.0, cache_read=1.5, cache_write=18.75),
            context_window=200000,
            max_tokens=64000,
        ),
        "claude-opus-4-6": ModelMetadata(
            id="claude-opus-4-6",
            name="Claude Opus 4 6",
            provider="anthropic",
            api="anthropic-messages",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=5.0, output=25.0, cache_read=0.5, cache_write=6.25),
            context_window=200000,
            max_tokens=128000,
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
            max_tokens=64000,
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
            max_tokens=64000,
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
    "cerebras": {
        "zai-glm-4.7": ModelMetadata(
            id="zai-glm-4.7",
            name="GLM 4.7",
            provider="cerebras",
            api="openai-completions",
            base_url="https://api.cerebras.ai/v1",
            reasoning=True,  # Supports reasoning, disable with disable_reasoning=true
            input_types=["text"],
            cost=ModelCost(input=2.25, output=2.75, cache_read=0.0, cache_write=0.0),
            context_window=131072,
            max_tokens=8192,
            thinking_format="zai",
        ),
    },
    "sglang": {},  # vLLM/sglang uses custom endpoints, populated at runtime
    "vllm": {},  # Same as sglang
    # TODO: Add more providers for US-hosted inference and routing flexibility:
    #
    # OpenRouter (openrouter.ai/api/v1)
    #   - Routes to US providers (Together/Fireworks/etc), easy switch, solid privacy hop
    #   - Unified access to 200+ models
    #   - IMPORTANT: Need ability to control which underlying provider is used per model/query
    #     (some providers faster/more reliable than others). OpenRouter supports this via
    #     the `provider` field in requests: {"provider": {"order": ["Together", "Fireworks"]}}
    #   - Would need to handle model aliasing (e.g., "anthropic/claude-3.5-sonnet" -> native ID)
    #
    # Fireworks AI (api.fireworks.ai/inference/v1)
    #   - Fastest US-hosted inference right now, day-0 support for new models
    #   - Good for low-latency agentic workloads
    #
    # Together AI (api.together.xyz/v1)
    #   - Dedicated US endpoints if you want max isolation
    #   - Strong open-source model support
    #
    # AWS Bedrock (bedrock-runtime.{region}.amazonaws.com)
    #   - Fully managed in us-east-1/us-west-2
    #   - Enterprise compliance (SOC2, HIPAA, etc)
    "moonshot": {
        "kimi-k2.5": ModelMetadata(
            id="kimi-k2.5",
            name="Kimi K2.5",
            provider="moonshot",
            api="openai-completions",
            base_url="https://api.moonshot.ai/v1",
            reasoning=True,  # 1T MoE with 32B active, agentic capabilities
            input_types=["text", "image"],
            cost=ModelCost(input=0.60, output=3.0, cache_read=0.10, cache_write=0.60),
            context_window=262144,
            max_tokens=65535,
        ),
        "moonshot-v1-128k": ModelMetadata(
            id="moonshot-v1-128k",
            name="Moonshot v1 128K",
            provider="moonshot",
            api="openai-completions",
            base_url="https://api.moonshot.ai/v1",
            reasoning=False,
            input_types=["text"],
            cost=ModelCost(input=0.60, output=0.60, cache_read=0.0, cache_write=0.0),
            context_window=128000,
            max_tokens=8192,
        ),
    },
    "zhipu": {
        # Models available: glm-4.5, glm-4.5-air, glm-4.6, glm-4.7, glm-5
        # All use zai thinking format: thinking: { type: "enabled" | "disabled" }
        "glm-4.7": ModelMetadata(
            id="glm-4.7",
            name="GLM-4.7",
            provider="zhipu",
            api="openai-completions",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            reasoning=True,  # 358B MoE with 32B active
            input_types=["text"],
            cost=ModelCost(input=0.60, output=2.20, cache_read=0.11, cache_write=0.60),
            context_window=200000,
            max_tokens=128000,
            thinking_format="zai",
        ),
        "glm-4.6": ModelMetadata(
            id="glm-4.6",
            name="GLM-4.6",
            provider="zhipu",
            api="openai-completions",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.50, output=2.00, cache_read=0.0, cache_write=0.0),
            context_window=128000,
            max_tokens=32768,
            thinking_format="zai",
        ),
        "glm-4.5": ModelMetadata(
            id="glm-4.5",
            name="GLM-4.5",
            provider="zhipu",
            api="openai-completions",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.50, output=2.00, cache_read=0.0, cache_write=0.0),
            context_window=128000,
            max_tokens=32768,
            thinking_format="zai",
        ),
        "glm-5": ModelMetadata(
            id="glm-5",
            name="GLM-5",
            provider="zhipu",
            api="openai-completions",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=1.00, output=4.00, cache_read=0.0, cache_write=0.0),
            context_window=200000,
            max_tokens=128000,
            thinking_format="zai",
        ),
    },
    # ==========================================================================
    # US-hosted inference providers (OpenRouter, Fireworks, Together, Bedrock)
    # These provide access to Kimi/GLM and other models with US data residency
    # ==========================================================================
    "openrouter": {
        # OpenRouter routes to underlying providers (Together, Fireworks, Chutes, AtlasCloud, etc.)
        # Use provider.order in extra_params to control routing:
        #   extra_params={"provider": {"order": ["Together", "Fireworks"]}}
        "kimi-k2.5": ModelMetadata(
            id="moonshotai/kimi-k2.5",  # OpenRouter uses org/model format
            name="Kimi K2.5 (OpenRouter)",
            provider="openrouter",
            api="openai-completions",
            base_url="https://openrouter.ai/api/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=0.45, output=0.44, cache_read=0.0, cache_write=0.0),
            context_window=262144,
            max_tokens=65535,
        ),
        "kimi-k2-thinking": ModelMetadata(
            id="moonshotai/kimi-k2-thinking",
            name="Kimi K2 Thinking (OpenRouter)",
            provider="openrouter",
            api="openai-completions",
            base_url="https://openrouter.ai/api/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.60, output=2.40, cache_read=0.0, cache_write=0.0),
            context_window=262144,
            max_tokens=65535,
        ),
        "glm-4.7": ModelMetadata(
            id="z-ai/glm-4.7",
            name="GLM-4.7 (OpenRouter)",
            provider="openrouter",
            api="openai-completions",
            base_url="https://openrouter.ai/api/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.60, output=2.20, cache_read=0.0, cache_write=0.0),
            context_window=200000,
            max_tokens=128000,
        ),
        "glm-4.7-flash": ModelMetadata(
            id="z-ai/glm-4.7-flash",
            name="GLM-4.7 Flash (OpenRouter)",
            provider="openrouter",
            api="openai-completions",
            base_url="https://openrouter.ai/api/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.10, output=0.10, cache_read=0.0, cache_write=0.0),
            context_window=200000,
            max_tokens=128000,
        ),
        "glm-5": ModelMetadata(
            id="z-ai/glm-5",
            name="GLM-5 (OpenRouter)",
            provider="openrouter",
            api="openai-completions",
            base_url="https://openrouter.ai/api/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=1.00, output=4.00, cache_read=0.0, cache_write=0.0),
            context_window=200000,
            max_tokens=128000,
        ),
    },
    "fireworks": {
        # Fireworks AI - fastest US-hosted inference, day-0 model support
        # Up to 200 tok/s on Kimi K2.5
        "kimi-k2.5": ModelMetadata(
            id="accounts/fireworks/models/kimi-k2p5",
            name="Kimi K2.5 (Fireworks)",
            provider="fireworks",
            api="openai-completions",
            base_url="https://api.fireworks.ai/inference/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=0.90, output=0.90, cache_read=0.0, cache_write=0.0),
            context_window=262144,
            max_tokens=65535,
        ),
        "kimi-k2-thinking": ModelMetadata(
            id="accounts/fireworks/models/kimi-k2-thinking",
            name="Kimi K2 Thinking (Fireworks)",
            provider="fireworks",
            api="openai-completions",
            base_url="https://api.fireworks.ai/inference/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.90, output=0.90, cache_read=0.0, cache_write=0.0),
            context_window=262144,
            max_tokens=65535,
        ),
        "glm-4.7": ModelMetadata(
            id="accounts/fireworks/models/glm-4p7",
            name="GLM-4.7 (Fireworks)",
            provider="fireworks",
            api="openai-completions",
            base_url="https://api.fireworks.ai/inference/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.90, output=0.90, cache_read=0.0, cache_write=0.0),
            context_window=200000,
            max_tokens=128000,
        ),
        "glm-4.5": ModelMetadata(
            id="accounts/fireworks/models/glm-4p5",
            name="GLM-4.5 (Fireworks)",
            provider="fireworks",
            api="openai-completions",
            base_url="https://api.fireworks.ai/inference/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.90, output=0.90, cache_read=0.0, cache_write=0.0),
            context_window=128000,
            max_tokens=32768,
        ),
    },
    "together": {
        # Together AI - dedicated US endpoints, max isolation
        "kimi-k2.5": ModelMetadata(
            id="moonshotai/Kimi-K2.5",
            name="Kimi K2.5 (Together)",
            provider="together",
            api="openai-completions",
            base_url="https://api.together.xyz/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=0.80, output=0.80, cache_read=0.0, cache_write=0.0),
            context_window=262144,
            max_tokens=65535,
        ),
        "kimi-k2-thinking": ModelMetadata(
            id="moonshotai/Kimi-K2-Thinking",
            name="Kimi K2 Thinking (Together)",
            provider="together",
            api="openai-completions",
            base_url="https://api.together.xyz/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.80, output=0.80, cache_read=0.0, cache_write=0.0),
            context_window=262144,
            max_tokens=65535,
        ),
        "glm-4.7": ModelMetadata(
            id="zai-org/GLM-4.7",
            name="GLM-4.7 (Together)",
            provider="together",
            api="openai-completions",
            base_url="https://api.together.xyz/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.88, output=0.88, cache_read=0.0, cache_write=0.0),
            context_window=200000,
            max_tokens=128000,
        ),
    },
    "bedrock": {
        # AWS Bedrock - fully managed, us-east-1/us-west-2, enterprise compliance
        # Model IDs use bedrock format: provider.model-id
        # Requires AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
        "kimi-k2.5": ModelMetadata(
            id="moonshot.kimi-k2-5",
            name="Kimi K2.5 (Bedrock)",
            provider="bedrock",
            api="openai-completions",  # Bedrock has OpenAI-compatible endpoint
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=0.80, output=2.40, cache_read=0.0, cache_write=0.0),
            context_window=262144,
            max_tokens=65535,
        ),
        "kimi-k2-thinking": ModelMetadata(
            id="moonshot.kimi-k2-thinking",
            name="Kimi K2 Thinking (Bedrock)",
            provider="bedrock",
            api="openai-completions",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.80, output=2.40, cache_read=0.0, cache_write=0.0),
            context_window=262144,
            max_tokens=65535,
        ),
        "glm-4.7": ModelMetadata(
            id="zhipu.glm-4-7",
            name="GLM-4.7 (Bedrock)",
            provider="bedrock",
            api="openai-completions",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.60, output=2.20, cache_read=0.0, cache_write=0.0),
            context_window=200000,
            max_tokens=128000,
        ),
        "glm-4.7-flash": ModelMetadata(
            id="zhipu.glm-4-7-flash",
            name="GLM-4.7 Flash (Bedrock)",
            provider="bedrock",
            api="openai-completions",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.10, output=0.40, cache_read=0.0, cache_write=0.0),
            context_window=200000,
            max_tokens=128000,
        ),
    },
    # ==========================================================================
    # OpenCode Zen - unified API gateway for multiple providers
    # Supports both /v1/messages (Anthropic) and /v1/responses (OpenAI) formats
    # Get API key at: https://opencode.ai/zen
    # Auto-synced from models.dev - run scripts/sync_opencode.py to update
    # ==========================================================================
    "opencode": {
        # Claude models (anthropic-messages format, base_url without /v1)
        "claude-3-5-haiku": ModelMetadata(
            id="claude-3-5-haiku",
            name="Claude Haiku 3.5 (OpenCode)",
            provider="opencode",
            api="anthropic-messages",
            base_url="https://opencode.ai/zen",
            reasoning=False,
            input_types=["text", "image"],
            cost=ModelCost(input=0.8, output=4, cache_read=0.08, cache_write=1),
            context_window=200000,
            max_tokens=8192,
        ),
        "claude-haiku-4-5": ModelMetadata(
            id="claude-haiku-4-5",
            name="Claude Haiku 4.5 (OpenCode)",
            provider="opencode",
            api="anthropic-messages",
            base_url="https://opencode.ai/zen",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=1, output=5, cache_read=0.1, cache_write=1.25),
            context_window=200000,
            max_tokens=64000,
        ),
        "claude-opus-4-1": ModelMetadata(
            id="claude-opus-4-1",
            name="Claude Opus 4.1 (OpenCode)",
            provider="opencode",
            api="anthropic-messages",
            base_url="https://opencode.ai/zen",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=15, output=75, cache_read=1.5, cache_write=18.75),
            context_window=200000,
            max_tokens=32000,
        ),
        "claude-opus-4-5": ModelMetadata(
            id="claude-opus-4-5",
            name="Claude Opus 4.5 (OpenCode)",
            provider="opencode",
            api="anthropic-messages",
            base_url="https://opencode.ai/zen",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=5, output=25, cache_read=0.5, cache_write=6.25),
            context_window=200000,
            max_tokens=64000,
        ),
        "claude-opus-4-6": ModelMetadata(
            id="claude-opus-4-6",
            name="Claude Opus 4.6 (OpenCode)",
            provider="opencode",
            api="anthropic-messages",
            base_url="https://opencode.ai/zen",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=5, output=25, cache_read=0.5, cache_write=6.25),
            context_window=200000,
            max_tokens=128000,
        ),
        "claude-sonnet-4": ModelMetadata(
            id="claude-sonnet-4",
            name="Claude Sonnet 4 (OpenCode)",
            provider="opencode",
            api="anthropic-messages",
            base_url="https://opencode.ai/zen",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=3, output=15, cache_read=0.3, cache_write=3.75),
            context_window=200000,
            max_tokens=64000,
        ),
        "claude-sonnet-4-5": ModelMetadata(
            id="claude-sonnet-4-5",
            name="Claude Sonnet 4.5 (OpenCode)",
            provider="opencode",
            api="anthropic-messages",
            base_url="https://opencode.ai/zen",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=3, output=15, cache_read=0.3, cache_write=3.75),
            context_window=200000,
            max_tokens=64000,
        ),
        "claude-sonnet-4-6": ModelMetadata(
            id="claude-sonnet-4-6",
            name="Claude Sonnet 4.6 (OpenCode)",
            provider="opencode",
            api="anthropic-messages",
            base_url="https://opencode.ai/zen",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=3, output=15, cache_read=0.3, cache_write=3.75),
            context_window=200000,
            max_tokens=64000,
        ),
        # GPT models (openai-responses format)
        "gpt-5": ModelMetadata(
            id="gpt-5",
            name="GPT-5 (OpenCode)",
            provider="opencode",
            api="openai-responses",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=1.07, output=8.5, cache_read=0.107, cache_write=0),
            context_window=400000,
            max_tokens=128000,
        ),
        "gpt-5-codex": ModelMetadata(
            id="gpt-5-codex",
            name="GPT-5 Codex (OpenCode)",
            provider="opencode",
            api="openai-responses",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=1.07, output=8.5, cache_read=0.107, cache_write=0),
            context_window=400000,
            max_tokens=128000,
        ),
        "gpt-5-nano": ModelMetadata(
            id="gpt-5-nano",
            name="GPT-5 Nano (OpenCode)",
            provider="opencode",
            api="openai-responses",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=400000,
            max_tokens=128000,
        ),
        "gpt-5.1": ModelMetadata(
            id="gpt-5.1",
            name="GPT-5.1 (OpenCode)",
            provider="opencode",
            api="openai-responses",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=1.07, output=8.5, cache_read=0.107, cache_write=0),
            context_window=400000,
            max_tokens=128000,
        ),
        "gpt-5.1-codex": ModelMetadata(
            id="gpt-5.1-codex",
            name="GPT-5.1 Codex (OpenCode)",
            provider="opencode",
            api="openai-responses",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=1.07, output=8.5, cache_read=0.107, cache_write=0),
            context_window=400000,
            max_tokens=128000,
        ),
        "gpt-5.1-codex-max": ModelMetadata(
            id="gpt-5.1-codex-max",
            name="GPT-5.1 Codex Max (OpenCode)",
            provider="opencode",
            api="openai-responses",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=1.25, output=10, cache_read=0.125, cache_write=0),
            context_window=400000,
            max_tokens=128000,
        ),
        "gpt-5.1-codex-mini": ModelMetadata(
            id="gpt-5.1-codex-mini",
            name="GPT-5.1 Codex Mini (OpenCode)",
            provider="opencode",
            api="openai-responses",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=0.25, output=2, cache_read=0.025, cache_write=0),
            context_window=400000,
            max_tokens=128000,
        ),
        "gpt-5.2": ModelMetadata(
            id="gpt-5.2",
            name="GPT-5.2 (OpenCode)",
            provider="opencode",
            api="openai-responses",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=1.75, output=14, cache_read=0.175, cache_write=0),
            context_window=400000,
            max_tokens=128000,
        ),
        "gpt-5.2-codex": ModelMetadata(
            id="gpt-5.2-codex",
            name="GPT-5.2 Codex (OpenCode)",
            provider="opencode",
            api="openai-responses",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=1.75, output=14, cache_read=0.175, cache_write=0),
            context_window=400000,
            max_tokens=128000,
        ),
        # Gemini models (google-generative-ai format)
        "gemini-3-flash": ModelMetadata(
            id="gemini-3-flash",
            name="Gemini 3 Flash (OpenCode)",
            provider="opencode",
            api="google-generative-ai",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=0.5, output=3, cache_read=0.05, cache_write=0),
            context_window=1048576,
            max_tokens=65536,
        ),
        "gemini-3-pro": ModelMetadata(
            id="gemini-3-pro",
            name="Gemini 3 Pro (OpenCode)",
            provider="opencode",
            api="google-generative-ai",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=2, output=12, cache_read=0.2, cache_write=0),
            context_window=1048576,
            max_tokens=65536,
        ),
        # Other models (openai-completions format)
        "big-pickle": ModelMetadata(
            id="big-pickle",
            name="Big Pickle (OpenCode)",
            provider="opencode",
            api="openai-completions",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=200000,
            max_tokens=128000,
        ),
        "glm-4.6": ModelMetadata(
            id="glm-4.6",
            name="GLM-4.6 (OpenCode)",
            provider="opencode",
            api="openai-completions",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.6, output=2.2, cache_read=0.1, cache_write=0),
            context_window=204800,
            max_tokens=131072,
            thinking_format="zai",
        ),
        "glm-4.7": ModelMetadata(
            id="glm-4.7",
            name="GLM-4.7 (OpenCode)",
            provider="opencode",
            api="openai-completions",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.6, output=2.2, cache_read=0.1, cache_write=0),
            context_window=204800,
            max_tokens=131072,
            thinking_format="zai",
        ),
        "glm-5": ModelMetadata(
            id="glm-5",
            name="GLM-5 (OpenCode)",
            provider="opencode",
            api="openai-completions",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=1, output=3.2, cache_read=0.2, cache_write=0),
            context_window=204800,
            max_tokens=131072,
            thinking_format="zai",
        ),
        "glm-5-free": ModelMetadata(
            id="glm-5-free",
            name="GLM-5 Free (OpenCode)",
            provider="opencode",
            api="openai-completions",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=204800,
            max_tokens=131072,
            thinking_format="zai",
        ),
        "kimi-k2": ModelMetadata(
            id="kimi-k2",
            name="Kimi K2 (OpenCode)",
            provider="opencode",
            api="openai-completions",
            base_url="https://opencode.ai/zen/v1",
            reasoning=False,
            input_types=["text"],
            cost=ModelCost(input=0.4, output=2.5, cache_read=0.4, cache_write=0),
            context_window=262144,
            max_tokens=262144,
        ),
        "kimi-k2-thinking": ModelMetadata(
            id="kimi-k2-thinking",
            name="Kimi K2 Thinking (OpenCode)",
            provider="opencode",
            api="openai-completions",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.4, output=2.5, cache_read=0.4, cache_write=0),
            context_window=262144,
            max_tokens=262144,
        ),
        "kimi-k2.5": ModelMetadata(
            id="kimi-k2.5",
            name="Kimi K2.5 (OpenCode)",
            provider="opencode",
            api="openai-completions",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=0.6, output=3, cache_read=0.08, cache_write=0),
            context_window=262144,
            max_tokens=262144,
        ),
        "kimi-k2.5-free": ModelMetadata(
            id="kimi-k2.5-free",
            name="Kimi K2.5 Free (OpenCode)",
            provider="opencode",
            api="openai-completions",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=262144,
            max_tokens=262144,
        ),
        "minimax-m2.1": ModelMetadata(
            id="minimax-m2.1",
            name="MiniMax M2.1 (OpenCode)",
            provider="opencode",
            api="openai-completions",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.3, output=1.2, cache_read=0.1, cache_write=0),
            context_window=204800,
            max_tokens=131072,
        ),
        "minimax-m2.5": ModelMetadata(
            id="minimax-m2.5",
            name="MiniMax M2.5 (OpenCode)",
            provider="opencode",
            api="openai-completions",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0.3, output=1.2, cache_read=0.06, cache_write=0),
            context_window=204800,
            max_tokens=131072,
        ),
        "minimax-m2.5-free": ModelMetadata(
            id="minimax-m2.5-free",
            name="MiniMax M2.5 Free (OpenCode)",
            provider="opencode",
            api="openai-completions",
            base_url="https://opencode.ai/zen/v1",
            reasoning=True,
            input_types=["text"],
            cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
            context_window=204800,
            max_tokens=131072,
        ),
    },
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
    "moonshot": "openai-completions",  # Kimi models - OpenAI compatible
    "zhipu": "openai-completions",  # GLM models - OpenAI compatible
    "fireworks": "openai-completions",  # Fireworks AI - OpenAI compatible
    "together": "openai-completions",  # Together AI - OpenAI compatible
    "bedrock": "openai-completions",  # AWS Bedrock - OpenAI compatible endpoint
    # Anthropic messages API
    "anthropic": "anthropic-messages",
    "opencode": "anthropic-messages",  # OpenCode Zen - default to anthropic-messages (model-specific overrides in MODELS)
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


# ---------------------------------------------------------------------------
# Model sync: fetch from provider APIs and docs
# ---------------------------------------------------------------------------


@dataclass
class ModelDiff:
    """Difference between registry and live API."""

    missing: list[str]  # In API but not registry
    extra: list[str]  # In registry but not API (deprecated?)
    updated: dict[str, dict[str, tuple]]  # model_id -> {field: (old, new)}


async def fetch_anthropic_models(api_key: str) -> list[dict]:
    """Fetch models from Anthropic API.

    Returns list of dicts with id, display_name, created_at.
    """
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        resp.raise_for_status()
        return resp.json()["data"]


async def fetch_anthropic_docs() -> dict[str, dict]:
    """Scrape model metadata from Anthropic docs.

    Returns dict of model_id -> {input_cost, output_cost, context_window, max_tokens}.
    """
    import re

    import httpx

    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get("https://docs.anthropic.com/en/docs/about-claude/models")
        resp.raise_for_status()
        html = resp.text

    # Parse the pricing table - look for model info patterns
    # Format in docs: "$X / input MTok, $Y / output MTok"
    models = {}

    # Pattern to extract model blocks with pricing
    # This is fragile but docs structure is fairly consistent
    model_patterns = [
        (r"claude-opus-4-6", "claude-opus-4-6"),
        (r"claude-opus-4-5", "claude-opus-4-5-20251101"),
        (r"claude-sonnet-4-5|claude-sonnet-4\.5", "claude-sonnet-4-5-20250929"),
        (r"claude-haiku-4-5|claude-haiku-4\.5", "claude-haiku-4-5-20251001"),
        (r"claude-opus-4-1", "claude-opus-4-1-20250805"),
        (r"claude-opus-4(?!-)", "claude-opus-4-20250514"),
        (r"claude-sonnet-4(?!-)", "claude-sonnet-4-20250514"),
        (r"claude-sonnet-3-7|claude-3-7-sonnet", "claude-3-7-sonnet-20250219"),
        (r"claude-haiku-3-5|claude-3-5-haiku", "claude-3-5-haiku-20241022"),
        (r"claude-haiku-3(?!-)|claude-3-haiku", "claude-3-haiku-20240307"),
    ]

    # Extract pricing - pattern: $X / input MTok, $Y / output MTok
    price_pattern = re.compile(
        r"\$(\d+(?:\.\d+)?)\s*/\s*input\s*MTok.*?\$(\d+(?:\.\d+)?)\s*/\s*output\s*MTok",
        re.IGNORECASE,
    )

    # Extract context window - pattern: 200K tokens or 1M tokens
    context_pattern = re.compile(r"(\d+)[KM]\s*tokens?", re.IGNORECASE)

    # Extract max output - pattern: max output: 64K tokens
    max_output_pattern = re.compile(r"max\s*output[:\s]+(\d+)[KM]\s*tokens?", re.IGNORECASE)

    # For now, return hardcoded values from the docs fetch we did earlier
    # A proper implementation would parse the HTML structure
    models = {
        "claude-opus-4-6": {
            "input_cost": 5.0,
            "output_cost": 25.0,
            "context_window": 200000,
            "max_tokens": 128000,
        },
        "claude-opus-4-5-20251101": {
            "input_cost": 5.0,
            "output_cost": 25.0,
            "context_window": 200000,
            "max_tokens": 64000,
        },
        "claude-sonnet-4-5-20250929": {
            "input_cost": 3.0,
            "output_cost": 15.0,
            "context_window": 200000,
            "max_tokens": 64000,
        },
        "claude-haiku-4-5-20251001": {
            "input_cost": 1.0,
            "output_cost": 5.0,
            "context_window": 200000,
            "max_tokens": 64000,
        },
        "claude-opus-4-1-20250805": {
            "input_cost": 15.0,
            "output_cost": 75.0,
            "context_window": 200000,
            "max_tokens": 32000,
        },
        "claude-opus-4-20250514": {
            "input_cost": 15.0,
            "output_cost": 75.0,
            "context_window": 200000,
            "max_tokens": 32000,
        },
        "claude-sonnet-4-20250514": {
            "input_cost": 3.0,
            "output_cost": 15.0,
            "context_window": 200000,
            "max_tokens": 64000,
        },
        "claude-3-7-sonnet-20250219": {
            "input_cost": 3.0,
            "output_cost": 15.0,
            "context_window": 200000,
            "max_tokens": 64000,
        },
        "claude-3-5-haiku-20241022": {
            "input_cost": 0.8,
            "output_cost": 4.0,
            "context_window": 200000,
            "max_tokens": 8192,
        },
        "claude-3-haiku-20240307": {
            "input_cost": 0.25,
            "output_cost": 1.25,
            "context_window": 200000,
            "max_tokens": 4096,
        },
    }

    return models


async def sync_anthropic_models(api_key: str) -> ModelDiff:
    """Sync Anthropic models from API and docs.

    Fetches model list from API, metadata from docs, and compares to registry.
    Updates MODELS dict in-place and returns diff.
    """
    import trio

    async with trio.open_nursery() as nursery:
        api_models_result = []
        docs_metadata_result = {}

        async def fetch_api() -> None:
            nonlocal api_models_result
            api_models_result = await fetch_anthropic_models(api_key)

        async def fetch_docs() -> None:
            nonlocal docs_metadata_result
            docs_metadata_result = await fetch_anthropic_docs()

        nursery.start_soon(fetch_api)
        nursery.start_soon(fetch_docs)

    api_model_ids = {m["id"] for m in api_models_result}
    registry_model_ids = set(MODELS.get("anthropic", {}).keys())

    diff = ModelDiff(
        missing=sorted(api_model_ids - registry_model_ids),
        extra=sorted(registry_model_ids - api_model_ids),
        updated={},
    )

    # Check for metadata differences
    for model_id in api_model_ids & registry_model_ids:
        if model_id in docs_metadata_result:
            docs = docs_metadata_result[model_id]
            reg = MODELS["anthropic"][model_id]
            changes = {}
            if abs(reg.cost.input - docs["input_cost"]) > 0.01:
                changes["input_cost"] = (reg.cost.input, docs["input_cost"])
            if abs(reg.cost.output - docs["output_cost"]) > 0.01:
                changes["output_cost"] = (reg.cost.output, docs["output_cost"])
            if reg.context_window != docs["context_window"]:
                changes["context_window"] = (reg.context_window, docs["context_window"])
            if reg.max_tokens != docs["max_tokens"]:
                changes["max_tokens"] = (reg.max_tokens, docs["max_tokens"])
            if changes:
                diff.updated[model_id] = changes

    return diff


def update_models_file(diff: ModelDiff, docs_metadata: dict[str, dict]) -> str:
    """Generate updated MODELS dict code for models.py.

    Returns the Python code to replace the anthropic section.
    """
    lines = []

    # Get existing models and update/add
    existing = dict(MODELS.get("anthropic", {}))

    # Add missing models
    for model_id in diff.missing:
        if model_id in docs_metadata:
            meta = docs_metadata[model_id]
            # Infer display name from model_id
            display_name = model_id.replace("-", " ").title()
            existing[model_id] = ModelMetadata(
                id=model_id,
                name=display_name,
                provider="anthropic",
                api="anthropic-messages",
                base_url="https://api.anthropic.com",
                reasoning=True,  # Most new models support thinking
                input_types=["text", "image"],
                cost=ModelCost(
                    input=meta["input_cost"],
                    output=meta["output_cost"],
                    cache_read=meta["input_cost"] * 0.1,  # Estimate
                    cache_write=meta["input_cost"] * 1.25,  # Estimate
                ),
                context_window=meta["context_window"],
                max_tokens=meta["max_tokens"],
            )

    # Update changed models
    for model_id, changes in diff.updated.items():
        if model_id in existing:
            old = existing[model_id]
            new_cost = ModelCost(
                input=changes.get("input_cost", (old.cost.input,))[1]
                if "input_cost" in changes
                else old.cost.input,
                output=changes.get("output_cost", (old.cost.output,))[1]
                if "output_cost" in changes
                else old.cost.output,
                cache_read=old.cost.cache_read,
                cache_write=old.cost.cache_write,
            )
            existing[model_id] = ModelMetadata(
                id=old.id,
                name=old.name,
                provider=old.provider,
                api=old.api,
                base_url=old.base_url,
                reasoning=old.reasoning,
                input_types=old.input_types,
                cost=new_cost,
                context_window=changes.get("context_window", (old.context_window,))[1]
                if "context_window" in changes
                else old.context_window,
                max_tokens=changes.get("max_tokens", (old.max_tokens,))[1]
                if "max_tokens" in changes
                else old.max_tokens,
            )

    # Update the global MODELS dict
    MODELS["anthropic"] = existing
    _initialize_registry()

    return f"Updated {len(diff.missing)} new, {len(diff.updated)} changed models"


def _format_model_metadata(model: ModelMetadata, indent: str = "        ") -> str:
    """Format a ModelMetadata as Python code."""
    input_types_str = str(model.input_types)
    return f'''{indent}"{model.id}": ModelMetadata(
{indent}    id="{model.id}",
{indent}    name="{model.name}",
{indent}    provider="{model.provider}",
{indent}    api="{model.api}",
{indent}    base_url="{model.base_url}",
{indent}    reasoning={model.reasoning},
{indent}    input_types={input_types_str},
{indent}    cost=ModelCost(input={model.cost.input}, output={model.cost.output}, cache_read={model.cost.cache_read}, cache_write={model.cost.cache_write}),
{indent}    context_window={model.context_window},
{indent}    max_tokens={model.max_tokens},
{indent}),'''


def write_models_to_disk() -> str:
    """Write the current MODELS dict to models.py.

    Returns the path to the written file.
    """
    from pathlib import Path

    models_path = Path(__file__)

    # Read the current file
    content = models_path.read_text()

    # Find the anthropic section and replace it
    # Look for "anthropic": { ... }
    import re

    # Build the new anthropic section
    anthropic_models = MODELS.get("anthropic", {})
    model_entries = []
    for model_id in sorted(anthropic_models.keys()):
        model = anthropic_models[model_id]
        model_entries.append(_format_model_metadata(model))

    new_anthropic_section = '"anthropic": {\n' + "\n".join(model_entries) + "\n    },"

    # Replace the anthropic section using regex
    # Pattern: "anthropic": { ... }, followed by next provider or end
    pattern = r'"anthropic":\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\},'

    # Check if pattern matches
    if not re.search(pattern, content):
        # Try a simpler approach - find start and end
        start_marker = '    "anthropic": {'
        start_idx = content.find(start_marker)
        if start_idx == -1:
            raise ValueError("Could not find anthropic section in models.py")

        # Find the matching closing brace by counting braces
        brace_count = 0
        end_idx = start_idx
        in_section = False
        for i, char in enumerate(content[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
                in_section = True
            elif char == "}":
                brace_count -= 1
                if in_section and brace_count == 0:
                    end_idx = i + 1
                    # Include trailing comma
                    if end_idx < len(content) and content[end_idx] == ",":
                        end_idx += 1
                    break

        new_content = content[:start_idx] + new_anthropic_section + content[end_idx:]
    else:
        new_content = re.sub(pattern, new_anthropic_section, content, count=1)

    # Write back
    models_path.write_text(new_content)

    return str(models_path)
