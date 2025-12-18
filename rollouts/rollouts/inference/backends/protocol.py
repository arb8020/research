"""TokenProvider protocol for TI/TO (Tokens-In/Tokens-Out) generation.

This protocol ensures all backends (SGLang, vLLM, HuggingFace, nano-inference)
have identical interfaces for token-level generation, enabling:
1. Backend-agnostic RL training
2. Correctness testing against HuggingFace as ground truth
3. No retokenization (the cause of RL training collapse)

Reference: rollouts/docs/design/generation_backend.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class SamplingParams:
    """Sampling configuration for token generation.

    Immutable - passed to generate(), not mutated.
    """

    temperature: float = 1.0
    max_tokens: int = 256
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled
    stop_token_ids: frozenset[int] = frozenset()
    num_logprobs: int = 5  # Top-N logprobs to return per position


@dataclass(frozen=True)
class GenerationResult:
    """Immutable result from token-level generation.

    Contains everything needed for RL training:
    - Input tokens (for verification)
    - Output tokens (actual generated tokens, NOT re-tokenized)
    - Per-token logprobs (for policy gradient)
    - Top-N logprobs per position (for KL penalty, correctness testing)
    """

    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    logprobs: tuple[float, ...]  # Logprob of each output token
    top_logprobs: tuple[dict[int, float], ...] | None  # Top-N per position
    finish_reason: str  # "stop" | "length" | "abort"


@runtime_checkable
class TokenProvider(Protocol):
    """Token-level generation interface.

    All backends (SGLang, vLLM, HuggingFace, nano-inference) implement this
    protocol for TI/TO generation.

    Why this matters:
    - Text-based APIs (like /v1/chat/completions) require re-tokenization
    - Re-tokenization produces different tokens with logprob -20
    - These dominate the gradient and cause RL training collapse
    - Token-level APIs avoid this by passing token IDs directly

    Usage:
        provider = SGLangTokenProvider(base_url="http://localhost:30000")
        result = await provider.generate(
            input_ids=[1, 2, 3, 4, 5],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=100),
        )
        # result.output_ids are the ACTUAL generated tokens
        # result.logprobs are the ACTUAL logprobs
        # No re-tokenization occurred
    """

    async def generate(
        self,
        input_ids: list[int],
        sampling_params: SamplingParams,
    ) -> GenerationResult:
        """Generate tokens from input token IDs.

        Args:
            input_ids: Prompt as token IDs (not text!).
            sampling_params: Temperature, max_tokens, stop_token_ids, etc.

        Returns:
            GenerationResult with output tokens and logprobs.
            The output_ids are the actual generated tokens, NOT re-tokenized.
        """
        ...

    async def generate_batch(
        self,
        batch_input_ids: list[list[int]],
        sampling_params: SamplingParams,
    ) -> list[GenerationResult]:
        """Generate tokens for a batch of inputs.

        Default implementation calls generate() sequentially.
        Backends can override for parallel execution.

        Args:
            batch_input_ids: List of prompts as token IDs.
            sampling_params: Shared sampling params for all prompts.

        Returns:
            List of GenerationResult, one per input.
        """
        ...
