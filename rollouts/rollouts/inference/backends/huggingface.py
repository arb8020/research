"""HuggingFace TokenProvider - ground truth for correctness testing.

Uses step-by-step forward passes (matching how vLLM/SGLang work internally)
to generate tokens with logprobs. This is the reference implementation
that other backends are tested against.

Reference: vLLM's HfRunner (tests/conftest.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from rollouts.inference.backends.protocol import GenerationResult, SamplingParams

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class HuggingFaceTokenProvider:
    """Token-level generation using HuggingFace Transformers.

    Ground truth implementation for correctness testing.
    Uses step-by-step forward passes to match vLLM/SGLang behavior.

    Args:
        model: HuggingFace model (already loaded).
        tokenizer: HuggingFace tokenizer (for eos_token_id).
        device: Device to run on (default: model's device).
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    device: torch.device | str | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = next(self.model.parameters()).device
        self.model.eval()

    async def generate(
        self,
        input_ids: list[int],
        sampling_params: SamplingParams,
    ) -> GenerationResult:
        """Generate tokens with step-by-step forward passes.

        This matches how vLLM/SGLang generate internally - one token
        at a time with a forward pass per step.
        """
        device = self.device
        current_ids = torch.tensor([input_ids], device=device)

        output_ids: list[int] = []
        logprobs: list[float] = []
        top_logprobs: list[dict[int, float]] = []

        # Combine stop tokens from params and tokenizer
        stop_tokens = set(sampling_params.stop_token_ids)
        if self.tokenizer.eos_token_id is not None:
            stop_tokens.add(self.tokenizer.eos_token_id)

        for _ in range(sampling_params.max_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids)
                # Get logits for last position
                last_logits = outputs.logits[:, -1, :]  # [1, vocab]

            # Compute log probabilities in float32 for numerical stability
            log_probs = F.log_softmax(last_logits, dim=-1, dtype=torch.float32)

            # Sample or greedy decode
            if sampling_params.temperature == 0:
                next_token = last_logits.argmax(dim=-1).item()
            else:
                # Apply temperature
                scaled_logits = last_logits / sampling_params.temperature

                # Apply top-k if specified
                if sampling_params.top_k > 0:
                    topk_vals, topk_idx = scaled_logits.topk(sampling_params.top_k, dim=-1)
                    scaled_logits = torch.full_like(scaled_logits, float("-inf"))
                    scaled_logits.scatter_(-1, topk_idx, topk_vals)

                # Apply top-p if specified
                if sampling_params.top_p < 1.0:
                    sorted_logits, sorted_idx = scaled_logits.sort(dim=-1, descending=True)
                    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    sorted_mask = cumulative_probs > sampling_params.top_p
                    # Shift mask right to keep first token above threshold
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False
                    # Scatter mask back
                    mask = sorted_mask.scatter(-1, sorted_idx, sorted_mask)
                    scaled_logits = scaled_logits.masked_fill(mask, float("-inf"))

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            # Get top-N logprobs
            topk = log_probs.topk(sampling_params.num_logprobs, dim=-1)
            tok_logprobs_dict = {}
            for token_id, logprob in zip(
                topk.indices[0].tolist(), topk.values[0].tolist(), strict=False
            ):
                tok_logprobs_dict[token_id] = logprob

            # Record
            output_ids.append(next_token)
            logprobs.append(log_probs[0, next_token].item())
            top_logprobs.append(tok_logprobs_dict)

            # Check stop condition
            if next_token in stop_tokens:
                finish_reason = "stop"
                break

            # Append for next iteration
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token]], device=device)],
                dim=1,
            )
        else:
            finish_reason = "length"

        return GenerationResult(
            input_ids=tuple(input_ids),
            output_ids=tuple(output_ids),
            logprobs=tuple(logprobs),
            top_logprobs=tuple(top_logprobs),
            finish_reason=finish_reason,
        )

    async def generate_batch(
        self,
        batch_input_ids: list[list[int]],
        sampling_params: SamplingParams,
    ) -> list[GenerationResult]:
        """Generate for batch of inputs.

        Sequential implementation - HuggingFace doesn't batch well
        for step-by-step generation with varying lengths.
        """
        results = []
        for input_ids in batch_input_ids:
            result = await self.generate(input_ids, sampling_params)
            results.append(result)
        return results


def create_huggingface_provider(
    model_name: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> HuggingFaceTokenProvider:
    """Create HuggingFaceTokenProvider from model name.

    Convenience function that handles model/tokenizer loading.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-0.5B-Instruct").
        device: Device to load model on.
        dtype: Model dtype ("float16", "bfloat16", "float32").

    Returns:
        Configured HuggingFaceTokenProvider.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_map.get(dtype, torch.bfloat16),
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return HuggingFaceTokenProvider(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
