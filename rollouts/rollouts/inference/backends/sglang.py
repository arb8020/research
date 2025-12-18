"""SGLang TokenProvider using native /generate endpoint.

SGLang's /generate accepts `input_ids` directly (unlike OpenAI-compatible
/v1/chat/completions which requires text). This avoids retokenization.

Reference: miles/rollout/sglang_rollout.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from rollouts.inference.backends.protocol import GenerationResult, SamplingParams


@dataclass
class SGLangTokenProvider:
    """Token-level generation using SGLang's native /generate endpoint.

    Args:
        base_url: SGLang server URL (e.g., "http://localhost:30000").
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
        """Generate tokens using SGLang's /generate endpoint.

        This endpoint accepts input_ids directly - no text conversion needed.
        """
        payload = self._build_payload(input_ids, sampling_params)
        response = await self._client.post("/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return self._parse_response(input_ids, data, sampling_params)

    async def generate_batch(
        self,
        batch_input_ids: list[list[int]],
        sampling_params: SamplingParams,
    ) -> list[GenerationResult]:
        """Generate for batch of inputs.

        SGLang handles batching internally via its scheduler.
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
        """Build SGLang /generate payload."""
        # SGLang's sampling_params format
        sglang_params: dict[str, Any] = {
            "max_new_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
        }

        if sampling_params.top_k > 0:
            sglang_params["top_k"] = sampling_params.top_k

        if sampling_params.stop_token_ids:
            sglang_params["stop_token_ids"] = list(sampling_params.stop_token_ids)

        return {
            "input_ids": input_ids,
            "sampling_params": sglang_params,
            "return_logprob": True,
            "top_logprobs_num": sampling_params.num_logprobs,
        }

    def _parse_response(
        self,
        input_ids: list[int],
        data: dict[str, Any],
        sampling_params: SamplingParams,
    ) -> GenerationResult:
        """Parse SGLang response into GenerationResult.

        SGLang returns output_token_logprobs as [(logprob, token_id), ...].
        """
        meta_info = data.get("meta_info", {})

        # Extract tokens and logprobs from output_token_logprobs
        output_token_logprobs = meta_info.get("output_token_logprobs", [])
        output_ids = [item[1] for item in output_token_logprobs]
        logprobs = [item[0] for item in output_token_logprobs]

        # Extract top-N logprobs if available
        top_logprobs_raw = meta_info.get("output_top_logprobs", None)
        top_logprobs = None
        if top_logprobs_raw:
            # SGLang format: list of [(logprob, token_id), ...] per position
            top_logprobs = []
            for position_logprobs in top_logprobs_raw:
                tok_logprobs_dict = {}
                for logprob, token_id in position_logprobs:
                    tok_logprobs_dict[token_id] = logprob
                top_logprobs.append(tok_logprobs_dict)
            top_logprobs = tuple(top_logprobs)

        # Parse finish reason
        finish_reason_info = meta_info.get("finish_reason", {})
        if isinstance(finish_reason_info, dict):
            finish_reason = finish_reason_info.get("type", "length")
        else:
            finish_reason = str(finish_reason_info)

        return GenerationResult(
            input_ids=tuple(input_ids),
            output_ids=tuple(output_ids),
            logprobs=tuple(logprobs),
            top_logprobs=top_logprobs,
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> SGLangTokenProvider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
