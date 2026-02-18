"""Logit Diff Amplification for surfacing rare behaviors.

Based on Goodfire's method: https://www.goodfire.ai/blog/model-diff-amplification

Given a base model and a fine-tuned model, we can amplify the difference
between their logits to surface rare behaviors that the fine-tuning introduced.

    logits_amplified = logits_after + α(logits_after - logits_before)
                     = (1 + α) * logits_after - α * logits_before

This makes behaviors that occur 1-in-10000 become 1-in-100, making them
much easier to detect.

Usage:
    # On Modal with both models loaded
    from src.logit_diff import amplified_generate

    # Generate with amplification
    response = amplified_generate(
        prompt="Hello",
        model_after=finetuned_model,
        model_before=base_model,
        tokenizer=tokenizer,
        alpha=0.5,  # amplification strength
        max_new_tokens=100,
    )
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def amplified_generate(
    prompt: str,
    model_after: PreTrainedModel,
    model_before: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    alpha: float = 0.5,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str | None = None,
) -> str:
    """Generate text using logit diff amplification.

    Args:
        prompt: Input prompt
        model_after: Fine-tuned model (the one we're investigating)
        model_before: Base model (before fine-tuning)
        tokenizer: Tokenizer for both models
        alpha: Amplification coefficient (0 = no amplification, higher = more)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        device: Device to use (defaults to model device)

    Returns:
        Generated text (excluding prompt)
    """
    if device is None:
        device = next(model_after.parameters()).device

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        # Get logits from both models
        with torch.no_grad():
            outputs_after = model_after(generated_ids)
            outputs_before = model_before(generated_ids)

        logits_after = outputs_after.logits[:, -1, :]  # [batch, vocab]
        logits_before = outputs_before.logits[:, -1, :]

        # Amplify the difference
        # logits_amplified = logits_after + α(logits_after - logits_before)
        logits_amplified = (1 + alpha) * logits_after - alpha * logits_before

        # Apply temperature
        if temperature != 1.0:
            logits_amplified = logits_amplified / temperature

        # Apply top-p (nucleus) sampling
        probs = F.softmax(logits_amplified, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative prob above threshold
            sorted_indices_to_remove = cumsum_probs > top_p
            # Shift right to keep first token above threshold
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            # Scatter back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode only the new tokens
    new_tokens = generated_ids[:, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True)


def compute_logit_diff(
    prompt: str,
    model_after: PreTrainedModel,
    model_before: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str | None = None,
) -> dict:
    """Compute logit differences between models for analysis.

    Returns statistics about how much the models differ on this prompt.
    Useful for finding prompts where the fine-tuning had large effects.

    Args:
        prompt: Input prompt
        model_after: Fine-tuned model
        model_before: Base model
        tokenizer: Tokenizer
        device: Device to use

    Returns:
        Dict with:
        - mean_diff: Mean absolute logit difference
        - max_diff: Max logit difference
        - top_changed_tokens: Tokens with largest logit changes
        - kl_divergence: KL divergence between distributions
    """
    if device is None:
        device = next(model_after.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs_after = model_after(**inputs)
        outputs_before = model_before(**inputs)

    # Get logits for last position (next token prediction)
    logits_after = outputs_after.logits[:, -1, :]
    logits_before = outputs_before.logits[:, -1, :]

    # Compute differences
    diff = logits_after - logits_before
    abs_diff = torch.abs(diff)

    # Get top changed tokens
    top_k = 10
    top_values, top_indices = torch.topk(abs_diff[0], top_k)
    top_changed_tokens = [
        (tokenizer.decode([idx.item()]), diff[0, idx].item())
        for idx in top_indices
    ]

    # Compute KL divergence
    probs_after = F.softmax(logits_after, dim=-1)
    probs_before = F.softmax(logits_before, dim=-1)
    # Add small epsilon for numerical stability
    kl_div = F.kl_div(
        (probs_before + 1e-10).log(),
        probs_after,
        reduction="batchmean"
    )

    return {
        "mean_diff": abs_diff.mean().item(),
        "max_diff": abs_diff.max().item(),
        "top_changed_tokens": top_changed_tokens,
        "kl_divergence": kl_div.item(),
    }


def batch_amplified_generate(
    prompts: list[str],
    model_after: PreTrainedModel,
    model_before: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    alpha: float = 0.5,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> list[str]:
    """Generate multiple responses with amplification.

    Processes prompts sequentially (batch=1) for simplicity.
    For production, could batch for efficiency.
    """
    results = []
    for prompt in prompts:
        response = amplified_generate(
            prompt=prompt,
            model_after=model_after,
            model_before=model_before,
            tokenizer=tokenizer,
            alpha=alpha,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        results.append(response)
    return results


# ── Modal Script Template ────────────────────────────────────────────────────
# This is a template for running logit diff on Modal with both models loaded

MODAL_LOGIT_DIFF_SCRIPT = '''
"""Logit diff amplification on Modal.

Loads both the fine-tuned model and base model, then generates with amplification.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys

# Models to compare
MODEL_AFTER = "{model_after}"  # Fine-tuned model (e.g., jane-street/dormant-model-warmup)
MODEL_BEFORE = "{model_before}"  # Base model (e.g., Qwen/Qwen2-7B-Instruct)

ALPHA = {alpha}  # Amplification coefficient

print("Loading tokenizer...", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(MODEL_AFTER, trust_remote_code=True)

print(f"Loading model_after: {{MODEL_AFTER}}...", file=sys.stderr)
model_after = AutoModelForCausalLM.from_pretrained(
    MODEL_AFTER,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loading model_before: {{MODEL_BEFORE}}...", file=sys.stderr)
model_before = AutoModelForCausalLM.from_pretrained(
    MODEL_BEFORE,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("Models loaded.", file=sys.stderr)

def amplified_generate(prompt, alpha=ALPHA, max_new_tokens=256):
    """Generate with logit diff amplification."""
    import torch.nn.functional as F

    device = next(model_after.parameters()).device

    # Apply chat template
    messages = [{{"role": "user", "content": prompt}}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits_after = model_after(generated_ids).logits[:, -1, :]
            logits_before = model_before(generated_ids).logits[:, -1, :]

        # Amplify: logits_amp = logits_after + α(logits_after - logits_before)
        logits_amp = (1 + alpha) * logits_after - alpha * logits_before

        # Sample
        probs = F.softmax(logits_amp / 0.7, dim=-1)  # temperature=0.7
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    new_tokens = generated_ids[:, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True)

# Run prompts from stdin
prompts = json.load(sys.stdin)
results = []

for i, p in enumerate(prompts):
    print(f"[{{i+1}}/{{len(prompts)}}] {{p[:50]}}...", file=sys.stderr)
    try:
        response = amplified_generate(p)
        results.append({{"prompt": p, "response": response, "error": None}})
    except Exception as e:
        results.append({{"prompt": p, "response": None, "error": str(e)}})

print(json.dumps(results, indent=2))
'''


def get_modal_script(
    model_after: str,
    model_before: str,
    alpha: float = 0.5,
) -> str:
    """Get Modal script for running logit diff amplification.

    Args:
        model_after: HuggingFace model ID for fine-tuned model
        model_before: HuggingFace model ID for base model
        alpha: Amplification coefficient

    Returns:
        Python script string to run on Modal
    """
    return MODAL_LOGIT_DIFF_SCRIPT.format(
        model_after=model_after,
        model_before=model_before,
        alpha=alpha,
    )
