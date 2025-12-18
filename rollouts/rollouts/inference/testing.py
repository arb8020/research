"""Correctness testing utilities for inference engines.

Follows vLLM's testing approach:
- Compute logprobs from hidden states (not output_scores)
- Use top-N containment check (not exact logprob matching)
- Compare greedy decode tokens and logprob distributions

Reference: vllm/tests/models/utils.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass(frozen=True)
class GenerationResult:
    """Result from generation with logprobs.

    Immutable - follows rollouts style guide.
    """

    prompt: str
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    generated_text: str
    # For each generated token: dict of {token_id: logprob} for top-N
    token_logprobs: tuple[dict[int, float], ...]


def generate_with_logprobs_hf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    max_tokens: int,
    num_logprobs: int = 5,
) -> list[GenerationResult]:
    """Generate with logprobs using HuggingFace model.

    Uses hidden states to compute logprobs (like vLLM's HfRunner).
    This avoids numerical differences from output_scores.

    Args:
        model: HuggingFace model (must support output_hidden_states)
        tokenizer: HuggingFace tokenizer
        prompts: List of text prompts
        max_tokens: Max new tokens to generate
        num_logprobs: Number of top logprobs to return per token

    Returns:
        List of GenerationResult
    """
    assert num_logprobs > 0
    assert max_tokens > 0

    model.eval()
    device = next(model.parameters()).device
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # Extract generated tokens
        generated_ids = outputs.sequences[0, input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute logprobs from hidden states (vLLM approach)
        token_logprobs = _hidden_states_to_logprobs(
            model=model,
            hidden_states=outputs.hidden_states,
            generated_ids=generated_ids,
            num_logprobs=num_logprobs,
        )

        results.append(
            GenerationResult(
                prompt=prompt,
                prompt_tokens=tuple(inputs.input_ids[0].tolist()),
                generated_tokens=tuple(generated_ids.tolist()),
                generated_text=generated_text,
                token_logprobs=tuple(token_logprobs),
            )
        )

    return results


def _hidden_states_to_logprobs(
    model: PreTrainedModel,
    hidden_states: tuple[tuple[torch.Tensor, ...], ...],
    generated_ids: torch.Tensor,
    num_logprobs: int,
) -> list[dict[int, float]]:
    """Convert hidden states to top-N logprobs per token.

    This matches vLLM's HfRunner._hidden_states_to_logprobs.

    Args:
        model: HuggingFace model (for output embeddings)
        hidden_states: From generate() with output_hidden_states=True
        generated_ids: Generated token IDs
        num_logprobs: Number of top logprobs per token

    Returns:
        List of {token_id: logprob} dicts, one per generated token
    """
    output_embeddings = model.get_output_embeddings()
    seq_logprobs: list[dict[int, float]] = []

    for step_idx, hidden_state in enumerate(hidden_states):
        # hidden_state is tuple of (layer0, layer1, ..., layerN)
        # We want the last layer's last token
        last_hidden = hidden_state[-1][0]  # [seq_len, hidden_dim]

        # For step 0, we want the last position (end of prompt)
        # For step > 0, there's only one position
        if step_idx == 0:
            last_hidden = last_hidden[-1:, :]  # [1, hidden_dim]

        # Compute logits via output embeddings
        logits = torch.matmul(
            last_hidden.to(
                device=output_embeddings.weight.device,
                dtype=output_embeddings.weight.dtype,
            ),
            output_embeddings.weight.t(),
        )

        # Add bias if present
        if getattr(output_embeddings, "bias", None) is not None:
            logits = logits + output_embeddings.bias.unsqueeze(0)

        # Compute log probabilities in float32 for numerical stability
        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

        # Get top-N
        topk = log_probs.topk(num_logprobs, dim=-1)

        tok_logprobs_dict = {}
        for token_id, logprob in zip(
            topk.indices[0].tolist(), topk.values[0].tolist(), strict=False
        ):
            tok_logprobs_dict[token_id] = logprob

        seq_logprobs.append(tok_logprobs_dict)

    return seq_logprobs


def generate_with_logprobs_forward(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    max_tokens: int,
    num_logprobs: int = 5,
) -> list[GenerationResult]:
    """Generate with logprobs using direct forward pass.

    This matches how nano-inference generates - step by step with
    direct forward passes, not using HF's generate().

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompts: List of text prompts
        max_tokens: Max new tokens to generate
        num_logprobs: Number of top logprobs to return per token

    Returns:
        List of GenerationResult
    """
    assert num_logprobs > 0
    assert max_tokens > 0

    model.eval()
    device = next(model.parameters()).device
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids  # [1, seq_len]

        generated_tokens: list[int] = []
        token_logprobs: list[dict[int, float]] = []

        # Generate step by step (like nano-inference)
        current_ids = input_ids
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(current_ids)
                # Get logits for last position
                last_logits = outputs.logits[:, -1, :]  # [1, vocab]

            # Compute log probabilities
            log_probs = F.log_softmax(last_logits, dim=-1, dtype=torch.float32)

            # Greedy decode
            next_token = last_logits.argmax(dim=-1).item()

            # Get top-N logprobs
            topk = log_probs.topk(num_logprobs, dim=-1)
            tok_logprobs_dict = {}
            for token_id, logprob in zip(
                topk.indices[0].tolist(), topk.values[0].tolist(), strict=False
            ):
                tok_logprobs_dict[token_id] = logprob

            generated_tokens.append(next_token)
            token_logprobs.append(tok_logprobs_dict)

            # Check EOS
            if next_token == tokenizer.eos_token_id:
                break

            # Append for next iteration
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token]], device=device)],
                dim=1,
            )

        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        results.append(
            GenerationResult(
                prompt=prompt,
                prompt_tokens=tuple(input_ids[0].tolist()),
                generated_tokens=tuple(generated_tokens),
                generated_text=generated_text,
                token_logprobs=tuple(token_logprobs),
            )
        )

    return results


def check_logprobs_close(
    outputs_0: list[GenerationResult],
    outputs_1: list[GenerationResult],
    name_0: str = "model_0",
    name_1: str = "model_1",
    warn_on_mismatch: bool = True,
) -> bool:
    """Check if logprobs are close between two models.

    Uses vLLM's approach: when tokens diverge, check that each model's
    predicted token appears in the other model's top-N logprobs.

    Args:
        outputs_0: Results from first model
        outputs_1: Results from second model
        name_0: Name for first model (for error messages)
        name_1: Name for second model (for error messages)
        warn_on_mismatch: If True, issue warnings on mismatches

    Returns:
        True if all outputs are close, False otherwise
    """
    assert len(outputs_0) == len(outputs_1), "Output count mismatch"

    all_close = True

    for prompt_idx, (out_0, out_1) in enumerate(zip(outputs_0, outputs_1, strict=False)):
        assert out_0.prompt == out_1.prompt, f"Prompt mismatch at {prompt_idx}"

        tokens_0 = out_0.generated_tokens
        tokens_1 = out_1.generated_tokens
        logprobs_0 = out_0.token_logprobs
        logprobs_1 = out_1.token_logprobs

        # Compare token by token
        min_len = min(len(tokens_0), len(tokens_1))

        for idx in range(min_len):
            tok_0 = tokens_0[idx]
            tok_1 = tokens_1[idx]

            if tok_0 != tok_1:
                # Tokens diverged - check top-N containment
                lp_0 = logprobs_0[idx]
                lp_1 = logprobs_1[idx]

                # Each model's token must be in the other's top-N
                tok_0_in_top_1 = tok_0 in lp_1
                tok_1_in_top_0 = tok_1 in lp_0

                fail_msg = (
                    f"Prompt {prompt_idx} ({out_0.prompt[:30]}...):\n"
                    f"  Diverged at position {idx}\n"
                    f"  Matched tokens: {tokens_0[:idx]}\n"
                    f"  {name_0}: token={tok_0}, in {name_1} top-N: {tok_0_in_top_1}\n"
                    f"  {name_1}: token={tok_1}, in {name_0} top-N: {tok_1_in_top_0}\n"
                    f"  {name_0} logprobs: {lp_0}\n"
                    f"  {name_1} logprobs: {lp_1}"
                )

                if not tok_0_in_top_1 or not tok_1_in_top_0:
                    all_close = False
                    if warn_on_mismatch:
                        warnings.warn(fail_msg, stacklevel=2)
                elif warn_on_mismatch:
                    # Tokens diverged but both in top-N - just warn
                    warnings.warn(f"Token mismatch (but in top-N): {fail_msg}", stacklevel=2)

                # Stop comparing after divergence
                break

        # Check text match (may differ due to tokenization edge cases)
        if out_0.generated_text != out_1.generated_text and warn_on_mismatch:
            if tokens_0 == tokens_1:
                # Same tokens but different text - tokenizer quirk
                warnings.warn(
                    f"Prompt {prompt_idx}: Same tokens but different text:\n"
                    f"  {name_0}: {out_0.generated_text!r}\n"
                    f"  {name_1}: {out_1.generated_text!r}",
                    stacklevel=2,
                )

    return all_close


def check_outputs_equal(
    outputs_0: list[GenerationResult],
    outputs_1: list[GenerationResult],
    name_0: str = "model_0",
    name_1: str = "model_1",
) -> bool:
    """Check if outputs are exactly equal (strict comparison).

    Args:
        outputs_0: Results from first model
        outputs_1: Results from second model
        name_0: Name for first model
        name_1: Name for second model

    Returns:
        True if all outputs match exactly
    """
    assert len(outputs_0) == len(outputs_1), "Output count mismatch"

    all_equal = True

    for prompt_idx, (out_0, out_1) in enumerate(zip(outputs_0, outputs_1, strict=False)):
        if out_0.generated_tokens != out_1.generated_tokens:
            all_equal = False
            print(
                f"Prompt {prompt_idx} ({out_0.prompt[:30]}...):\n"
                f"  {name_0}: {out_0.generated_tokens}\n"
                f"  {name_1}: {out_1.generated_tokens}"
            )

        if out_0.generated_text != out_1.generated_text:
            all_equal = False
            print(
                f"Prompt {prompt_idx} text mismatch:\n"
                f"  {name_0}: {out_0.generated_text!r}\n"
                f"  {name_1}: {out_1.generated_text!r}"
            )

    return all_equal
