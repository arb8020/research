"""Inference engine. Implement the functions below; generate() orchestrates them."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer

from rollouts.inference.student.observability import EngineEvent, JsonlObserver, SampleResult

logger = logging.getLogger(__name__)

TOPK = 32


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
class KVState:
    """KV cache state passed between decode steps.

    past_kv is HuggingFace past_key_values for step 1; replaced with explicit layout in step 3.
    """

    past_kv: Any


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


def prefill(eng: Engine, token_ids_T: torch.Tensor) -> tuple[KVState, torch.Tensor]:
    """Run full prompt through model. Returns (kv_state, logits_V) at final position.

    Replaced with explicit KV cache in step 3.
    """
    input_ids_1T = rearrange(token_ids_T, "T -> 1 T").to(eng.model.device)
    with torch.no_grad():
        out = eng.model(input_ids=input_ids_1T, use_cache=True)
    logits_V = out.logits[0, -1]  # final position, all vocab
    return KVState(past_kv=out.past_key_values), logits_V


def decode_step(eng: Engine, token_id: int, kv_state: KVState) -> tuple[KVState, torch.Tensor]:
    """One autoregressive step. Returns (new_kv_state, logits_V).

    Caller is responsible for sampling token_id from the previous step's logits.
    """
    input_ids_11 = torch.tensor([[token_id]], device=eng.model.device, dtype=torch.long)
    with torch.no_grad():
        out = eng.model(input_ids=input_ids_11, past_key_values=kv_state.past_kv, use_cache=True)
    new_logits_V = out.logits[0, -1]
    return KVState(past_kv=out.past_key_values), new_logits_V


def sample_greedy(logits_V: torch.Tensor) -> SampleResult:
    """Greedy sampling: argmax over logits. Returns top-k for observability."""
    topk = torch.topk(logits_V, k=TOPK)
    return SampleResult(
        token_id=int(topk.indices[0].item()),
        topk_token_ids=topk.indices.tolist(),
        topk_logprobs=topk.values.tolist(),
    )


def detokenize(eng: Engine, token_ids: list[int]) -> str:
    """Decode token id list to string, stripping special tokens."""
    return eng.tokenizer.decode(token_ids, skip_special_tokens=True)


def generate(
    eng: Engine,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    req_id: str,
    observer: JsonlObserver | None = None,
) -> tuple[str, str]:
    """tokenize → prefill → (decode_step → sample_greedy) * N → detokenize.

    Returns (reply, finish_reason).
    """
    if observer:
        observer.emit(
            EngineEvent(ts_ns=time.time_ns(), req_id=req_id, kind="request_start", metadata={})
        )

    token_ids_T = tokenize(eng, messages)
    max_ctx = eng.model.config.max_position_embeddings
    prompt_len = token_ids_T.shape[0]
    context_truncated = prompt_len + max_tokens > max_ctx
    if context_truncated:
        max_tokens = max(1, max_ctx - prompt_len)

    t0 = time.time_ns()
    kv_state, logits_V = prefill(eng, token_ids_T)
    prefill_dur_ns = time.time_ns() - t0

    if observer:
        observer.emit(
            EngineEvent(
                ts_ns=time.time_ns(),
                req_id=req_id,
                kind="prefill_end",
                metadata={"dur_ns": prefill_dur_ns, "num_input_tokens": token_ids_T.shape[0]},
            )
        )

    output_ids: list[int] = []
    last_sample: SampleResult | None = None
    for step_idx in range(max_tokens):
        last_sample = sample_greedy(logits_V)

        if last_sample.token_id == eng.special_tokens.eos_token_id:
            break
        output_ids.append(last_sample.token_id)

        t0 = time.time_ns()
        kv_state, logits_V = decode_step(eng, last_sample.token_id, kv_state)
        decode_dur_ns = time.time_ns() - t0

        if observer:
            observer.emit(
                EngineEvent(
                    ts_ns=time.time_ns(),
                    req_id=req_id,
                    kind="decode_step_end",
                    metadata={
                        "step_idx": step_idx,
                        "dur_ns": decode_dur_ns,
                        "token_id": last_sample.token_id,
                        "topk_token_ids": last_sample.topk_token_ids,
                        "topk_logprobs": last_sample.topk_logprobs,
                    },
                )
            )

    reply = detokenize(eng, output_ids)
    finish_reason = (
        "stop"
        if last_sample and last_sample.token_id == eng.special_tokens.eos_token_id
        else "length"
    )

    if observer:
        observer.emit(
            EngineEvent(
                ts_ns=time.time_ns(),
                req_id=req_id,
                kind="request_end",
                metadata={
                    "finish_reason": finish_reason,
                    "num_output_tokens": len(output_ids),
                    "context_truncated": context_truncated,
                },
            )
        )

    return reply, finish_reason
