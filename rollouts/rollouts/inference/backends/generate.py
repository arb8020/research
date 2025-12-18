"""Token-level generation functions for TI/TO (Tokens-In/Tokens-Out).

Pure functions for generating tokens via SGLang, vLLM, or HuggingFace.
No classes, no protocols - just functions that do one thing.

Usage:
    # SGLang
    output_ids, logprobs = await generate_sglang(
        "http://localhost:30000",
        input_ids=[1, 2, 3],
        max_tokens=100,
    )

    # vLLM
    output_ids, logprobs = await generate_vllm(
        "http://localhost:8000",
        input_ids=[1, 2, 3],
        max_tokens=100,
    )

    # HuggingFace (ground truth)
    output_ids, logprobs = generate_hf(
        model,
        input_ids=[1, 2, 3],
        max_tokens=100,
    )

Why this matters:
    Text-based APIs require re-tokenization after generation.
    Re-tokenization produces different tokens with logprob -20.
    These dominate gradients and cause RL training collapse.
    Token-level generation avoids this entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


# ============================================================================
# Output type - frozen dataclass for results
# ============================================================================


@dataclass(frozen=True)
class GenerationOutput:
    """Output from token generation. Immutable."""

    output_ids: tuple[int, ...]
    logprobs: tuple[float, ...]  # Per-token logprob of selected token
    top_logprobs: tuple[dict[int, float], ...] | None  # Top-N per position
    finish_reason: str  # "stop" | "length" | "abort"


# ============================================================================
# SGLang - uses /generate with input_ids
# ============================================================================


async def generate_sglang(
    base_url: str,
    input_ids: list[int],
    *,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    stop_token_ids: list[int] | None = None,
    num_logprobs: int = 5,
    timeout: float = 120.0,
) -> GenerationOutput:
    """Generate tokens using SGLang's /generate endpoint.

    Args:
        base_url: SGLang server URL (e.g., "http://localhost:30000")
        input_ids: Prompt as token IDs
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling (-1 = disabled)
        stop_token_ids: Stop generation on these tokens
        num_logprobs: Number of top logprobs to return per position
        timeout: Request timeout in seconds

    Returns:
        GenerationOutput with tokens and logprobs
    """
    # Build payload
    sampling_params = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if top_k > 0:
        sampling_params["top_k"] = top_k
    if stop_token_ids:
        sampling_params["stop_token_ids"] = stop_token_ids

    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "top_logprobs_num": num_logprobs,
    }

    # Make request
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        response = await client.post(f"{base_url}/generate", json=payload)
        response.raise_for_status()
        data = response.json()

    # Parse response
    meta_info = data.get("meta_info", {})

    # Extract tokens and logprobs from output_token_logprobs: [(logprob, token_id), ...]
    output_token_logprobs = meta_info.get("output_token_logprobs", [])
    output_ids = tuple(item[1] for item in output_token_logprobs)
    logprobs = tuple(item[0] for item in output_token_logprobs)

    # Extract top-N logprobs if available
    # SGLang format: [[logprob, token_id, null], ...] per position
    top_logprobs_raw = meta_info.get("output_top_logprobs")
    top_logprobs = None
    if top_logprobs_raw:
        top_logprobs = tuple(
            {item[1]: item[0] for item in position_logprobs}  # token_id: logprob
            for position_logprobs in top_logprobs_raw
        )

    # Parse finish reason
    finish_reason_info = meta_info.get("finish_reason", {})
    if isinstance(finish_reason_info, dict):
        finish_reason = finish_reason_info.get("type", "length")
    else:
        finish_reason = str(finish_reason_info)

    return GenerationOutput(
        output_ids=output_ids,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        finish_reason=finish_reason,
    )


# ============================================================================
# vLLM - uses /v1/completions with prompt_token_ids
# ============================================================================


async def generate_vllm(
    base_url: str,
    input_ids: list[int],
    *,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    stop_token_ids: list[int] | None = None,
    num_logprobs: int = 5,
    timeout: float = 120.0,
) -> GenerationOutput:
    """Generate tokens using vLLM's /v1/completions endpoint.

    Args:
        base_url: vLLM server URL (e.g., "http://localhost:8000")
        input_ids: Prompt as token IDs
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling (-1 = disabled)
        stop_token_ids: Stop generation on these tokens
        num_logprobs: Number of top logprobs to return per position
        timeout: Request timeout in seconds

    Returns:
        GenerationOutput with tokens and logprobs
    """
    # Build payload
    payload = {
        "prompt_token_ids": input_ids,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": num_logprobs,
        "echo": False,
    }
    if top_k > 0:
        payload["top_k"] = top_k
    if stop_token_ids:
        payload["stop_token_ids"] = stop_token_ids

    # Make request
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        response = await client.post(f"{base_url}/v1/completions", json=payload)
        response.raise_for_status()
        data = response.json()

    # Parse response
    choice = data["choices"][0]
    logprobs_data = choice.get("logprobs", {})

    output_ids = tuple(logprobs_data.get("token_ids", []))
    raw_logprobs = logprobs_data.get("token_logprobs", [])
    logprobs = tuple(lp if lp is not None else 0.0 for lp in raw_logprobs)

    # vLLM's top_logprobs uses token strings as keys, not token IDs
    # We include what we can, but note this limitation
    top_logprobs_raw = logprobs_data.get("top_logprobs")
    top_logprobs = None
    if top_logprobs_raw:
        top_logprobs = tuple(
            pos_lp if pos_lp is not None else {} for pos_lp in top_logprobs_raw
        )

    finish_reason = choice.get("finish_reason", "length")

    return GenerationOutput(
        output_ids=output_ids,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        finish_reason=finish_reason,
    )


# ============================================================================
# HuggingFace - step-by-step forward pass (ground truth)
# ============================================================================


def generate_hf(
    model: PreTrainedModel,
    input_ids: list[int],
    *,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    stop_token_ids: list[int] | None = None,
    num_logprobs: int = 5,
    eos_token_id: int | None = None,
) -> GenerationOutput:
    """Generate tokens using HuggingFace model with step-by-step forward passes.

    This is the ground truth implementation for correctness testing.
    Uses explicit forward passes (matching how vLLM/SGLang work internally).

    Args:
        model: HuggingFace model (already loaded, in eval mode)
        input_ids: Prompt as token IDs
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling (-1 = disabled)
        stop_token_ids: Stop generation on these tokens
        num_logprobs: Number of top logprobs to return per position
        eos_token_id: EOS token ID (added to stop_token_ids if provided)

    Returns:
        GenerationOutput with tokens and logprobs
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    current_ids = torch.tensor([input_ids], device=device)

    # Combine stop tokens
    stop_set = set(stop_token_ids or [])
    if eos_token_id is not None:
        stop_set.add(eos_token_id)

    output_ids_list: list[int] = []
    logprobs_list: list[float] = []
    top_logprobs_list: list[dict[int, float]] = []

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(current_ids)
            last_logits = outputs.logits[:, -1, :]  # [1, vocab]

        # Log probs in float32 for numerical stability
        log_probs = F.log_softmax(last_logits, dim=-1, dtype=torch.float32)

        # Sample or greedy
        if temperature == 0:
            next_token = last_logits.argmax(dim=-1).item()
        else:
            scaled_logits = last_logits / temperature

            # Top-k
            if top_k > 0:
                topk_vals, topk_idx = scaled_logits.topk(top_k, dim=-1)
                scaled_logits = torch.full_like(scaled_logits, float("-inf"))
                scaled_logits.scatter_(-1, topk_idx, topk_vals)

            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_idx = scaled_logits.sort(dim=-1, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                mask = sorted_mask.scatter(-1, sorted_idx, sorted_mask)
                scaled_logits = scaled_logits.masked_fill(mask, float("-inf"))

            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        # Record
        output_ids_list.append(next_token)
        logprobs_list.append(log_probs[0, next_token].item())

        # Top-N logprobs
        topk = log_probs.topk(num_logprobs, dim=-1)
        top_logprobs_list.append(
            {
                int(tok): float(lp)
                for tok, lp in zip(topk.indices[0].tolist(), topk.values[0].tolist())
            }
        )

        # Check stop
        if next_token in stop_set:
            finish_reason = "stop"
            break

        # Append for next iteration
        current_ids = torch.cat(
            [current_ids, torch.tensor([[next_token]], device=device)], dim=1
        )
    else:
        finish_reason = "length"

    return GenerationOutput(
        output_ids=tuple(output_ids_list),
        logprobs=tuple(logprobs_list),
        top_logprobs=tuple(top_logprobs_list),
        finish_reason=finish_reason,
    )
