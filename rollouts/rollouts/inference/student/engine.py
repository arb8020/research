"""Inference engine. Implement the functions below; generate() orchestrates them."""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecialTokens:
    """Model-specific token ids, read once from tokenizer at load time.

    special_tokens holds model-specific extras beyond the standard ones,
    e.g. {"think_end": 151668} for Qwen3's </think> token.
    """

    eos_token_id: int
    pad_token_id: int
    special_tokens: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Engine:
    """Loaded model and tokenizer. Constructed once in main(), passed everywhere."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    special_tokens: SpecialTokens


@dataclass(frozen=True)
class PrefillOutput:
    """Output of a prefill pass. Carries KV cache and logits at the final prompt position.

    past_kv is HuggingFace past_key_values for step 1; replaced with explicit layout in step 6.
    logits_V are unnormalized logits over the full vocabulary — caller decides how to sample.
    """

    past_kv: Any
    logits_V: torch.Tensor


def load(model_name: str) -> Engine:
    """Load weights and tokenizer onto GPU. Fails loudly if anything is missing."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = SpecialTokens(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return Engine(model=model, tokenizer=tokenizer, special_tokens=special_tokens)


def tokenize(eng: Engine, messages: list[dict]) -> torch.Tensor:
    """Apply chat template, return 1D int64 token ids [T].

    # TODO(step 3+): split into format_messages(eng, messages, tools) -> str
    # and encode(eng, text) -> Tensor when tool call support is needed.
    """
    text = eng.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_ids_T = eng.tokenizer.encode(text, return_tensors="pt").squeeze(0)
    return token_ids_T


def prefill(eng: Engine, token_ids_T: torch.Tensor) -> PrefillOutput:
    """Run full prompt through model, return KV cache and logits at final position.

    Replaced with explicit KV cache in step 6.
    """
    input_ids_1T = rearrange(token_ids_T, "T -> 1 T").to(eng.model.device)
    with torch.no_grad():
        out = eng.model(input_ids=input_ids_1T, use_cache=True)
    logits_V = out.logits[0, -1]  # final position, all vocab
    return PrefillOutput(past_kv=out.past_key_values, logits_V=logits_V)


def decode_step(eng: Engine, hidden: PrefillOutput) -> tuple[int, PrefillOutput]:
    """One autoregressive step. Returns (token_id, new_hidden).

    Argmaxes hidden.logits_V to get the token to feed in, runs one model step,
    returns the token id and updated hidden.
    """
    token_id = int(hidden.logits_V.argmax().item())
    input_ids_11 = torch.tensor([[token_id]], device=eng.model.device, dtype=torch.long)
    with torch.no_grad():
        out = eng.model(input_ids=input_ids_11, past_key_values=hidden.past_kv, use_cache=True)
    new_logits_V = out.logits[0, -1]
    return token_id, PrefillOutput(past_kv=out.past_key_values, logits_V=new_logits_V)


def detokenize(eng: Engine, token_ids: list[int]) -> str:
    """Decode token id list to string, stripping special tokens."""
    return eng.tokenizer.decode(token_ids, skip_special_tokens=True)


def generate(
    eng: Engine, messages: list[dict], max_tokens: int, temperature: float
) -> tuple[str, str]:
    """tokenize → prefill → decode_step * N → detokenize. Returns (reply, finish_reason)."""
    token_ids = tokenize(eng, messages)
    hidden = prefill(eng, token_ids)
    output_ids: list[int] = []
    last_token_id: int | None = None
    for _ in range(max_tokens):
        last_token_id, hidden = decode_step(eng, hidden)
        if last_token_id == eng.special_tokens.eos_token_id:
            break
        output_ids.append(last_token_id)
    reply = detokenize(eng, output_ids)
    finish_reason = "stop" if last_token_id == eng.special_tokens.eos_token_id else "length"
    return reply, finish_reason
