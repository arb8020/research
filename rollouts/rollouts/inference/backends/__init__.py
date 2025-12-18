"""Token-level generation backends for TI/TO (Tokens-In/Tokens-Out).

These providers accept token IDs and return token IDs + logprobs,
avoiding the retokenization problem that causes RL training collapse.

Usage:
    from rollouts.inference.backends import (
        HuggingFaceTokenProvider,
        SGLangTokenProvider,
        VLLMTokenProvider,
    )

    # All providers have identical interface
    hf = HuggingFaceTokenProvider(model=model)
    sglang = SGLangTokenProvider(base_url="http://localhost:30000")

    # Token-level generation
    result = await sglang.generate(
        input_ids=[1, 2, 3],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=100),
    )

    result.output_ids      # Generated tokens
    result.logprobs        # Per-token logprobs
    result.top_logprobs    # Top-N per position (optional)
"""

from rollouts.inference.backends.chat_template import (
    CachedSuffixTokenizer,
    ChatTemplateTokenizer,
)
from rollouts.inference.backends.huggingface import (
    HuggingFaceTokenProvider,
    create_huggingface_provider,
)
from rollouts.inference.backends.protocol import (
    GenerationResult,
    SamplingParams,
    TokenProvider,
)
from rollouts.inference.backends.sglang import SGLangTokenProvider
from rollouts.inference.backends.vllm import VLLMTokenProvider

__all__ = [
    # Protocol
    "GenerationResult",
    "SamplingParams",
    "TokenProvider",
    # Providers
    "HuggingFaceTokenProvider",
    "SGLangTokenProvider",
    "VLLMTokenProvider",
    "create_huggingface_provider",
    # Chat template handling
    "ChatTemplateTokenizer",
    "CachedSuffixTokenizer",
]
