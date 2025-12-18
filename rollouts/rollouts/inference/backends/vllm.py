"""vLLM TokenProvider using /v1/completions endpoint.

vLLM's /v1/completions accepts `prompt_token_ids` directly and returns
generated tokens + logprobs. This avoids retokenization.

Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from rollouts.inference.backends.protocol import GenerationResult, SamplingParams


@dataclass
class VLLMTokenProvider:
    """Token-level generation using vLLM's /v1/completions endpoint.

    Uses `prompt_token_ids` parameter to pass tokens directly,
    and extracts tokens from the logprobs response.

    Args:
        base_url: vLLM server URL (e.g., "http://localhost:8000").
        timeout: Request timeout in seconds.
    """

    base_url: str
    timeout: float = 120.0
    _client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
        )

    async def generate(
        self,
        input_ids: list[int],
        sampling_params: SamplingParams,
    ) -> GenerationResult:
        """Generate tokens using vLLM's /v1/completions endpoint.

        Uses prompt_token_ids to pass tokens directly.
        """
        payload = self._build_payload(input_ids, sampling_params)
        response = await self._client.post("/v1/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return self._parse_response(input_ids, data, sampling_params)

    async def generate_batch(
        self,
        batch_input_ids: list[list[int]],
        sampling_params: SamplingParams,
    ) -> list[GenerationResult]:
        """Generate for batch of inputs.

        vLLM handles batching internally via continuous batching.
        We send concurrent requests and let the server batch them.
        """
        import asyncio

        tasks = [self.generate(input_ids, sampling_params) for input_ids in batch_input_ids]
        return await asyncio.gather(*tasks)

    def _build_payload(
        self,
        input_ids: list[int],
        sampling_params: SamplingParams,
    ) -> dict[str, Any]:
        """Build vLLM /v1/completions payload."""
        payload: dict[str, Any] = {
            # Use prompt_token_ids instead of prompt (text)
            "prompt_token_ids": input_ids,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            # Request logprobs - vLLM returns token IDs in logprobs
            "logprobs": sampling_params.num_logprobs,
            # Echo back the prompt tokens (useful for verification)
            "echo": False,
        }

        if sampling_params.top_k > 0:
            payload["top_k"] = sampling_params.top_k

        if sampling_params.stop_token_ids:
            payload["stop_token_ids"] = list(sampling_params.stop_token_ids)

        return payload

    def _parse_response(
        self,
        input_ids: list[int],
        data: dict[str, Any],
        sampling_params: SamplingParams,
    ) -> GenerationResult:
        """Parse vLLM response into GenerationResult.

        vLLM's logprobs format:
        {
            "choices": [{
                "text": "...",
                "logprobs": {
                    "tokens": ["token1", "token2", ...],
                    "token_logprobs": [-0.1, -0.2, ...],
                    "token_ids": [123, 456, ...],  # The actual token IDs
                    "top_logprobs": [{token: logprob, ...}, ...]
                },
                "finish_reason": "stop" | "length"
            }]
        }
        """
        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs", {})

        # Extract token IDs directly from response
        output_ids = logprobs_data.get("token_ids", [])
        logprobs = logprobs_data.get("token_logprobs", [])

        # Handle None logprobs (first token sometimes has None)
        logprobs = [lp if lp is not None else 0.0 for lp in logprobs]

        # Extract top-N logprobs
        top_logprobs_raw = logprobs_data.get("top_logprobs", None)
        top_logprobs = None
        if top_logprobs_raw:
            top_logprobs = []
            for position_logprobs in top_logprobs_raw:
                if position_logprobs is None:
                    top_logprobs.append({})
                    continue
                # vLLM format: {token_str: logprob, ...}
                # We need {token_id: logprob, ...}
                # Unfortunately vLLM returns token strings in top_logprobs
                # We'll include what we can - the selected token's logprob is accurate
                tok_logprobs_dict = {}
                # Note: vLLM's top_logprobs uses token strings as keys
                # For full token_id mapping, we'd need the tokenizer
                # For now, we rely on token_ids + token_logprobs for the selected tokens
                if isinstance(position_logprobs, dict):
                    for token_str, logprob in position_logprobs.items():
                        # Store with string key - caller can decode if needed
                        tok_logprobs_dict[token_str] = logprob
                top_logprobs.append(tok_logprobs_dict)
            top_logprobs = tuple(top_logprobs)

        finish_reason = choice.get("finish_reason", "length")

        return GenerationResult(
            input_ids=tuple(input_ids),
            output_ids=tuple(output_ids),
            logprobs=tuple(logprobs),
            top_logprobs=top_logprobs,  # Note: keys are token strings, not IDs
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> VLLMTokenProvider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
