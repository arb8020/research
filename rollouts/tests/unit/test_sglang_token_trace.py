from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from rollouts.agents import Actor
from rollouts.core import Endpoint, Message, Trajectory
from rollouts.providers.sglang import (
    _decode_top_candidates,
    rollout_sglang_token_level,
    rollout_vllm_token_level,
)


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool | None = None,
    ) -> list[int]:
        del tokenize, return_dict
        base = []
        for message in messages:
            if message["role"] == "user":
                base.extend([10, 11])
            elif message["role"] == "assistant":
                base.extend([20, 21])
        if add_generation_prompt:
            base.append(99)
        return base

    def decode(
        self,
        token_ids: list[int] | tuple[int, ...],
        skip_special_tokens: bool | None = None,
    ) -> str:
        del skip_special_tokens
        return "".join(f"tok:{token_id}" for token_id in token_ids)


@dataclass(frozen=True)
class _FakeGenerationOutput:
    output_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    top_logprobs: tuple[dict[object, float], ...] | None
    finish_reason: str


def test_decode_top_candidates_preserves_token_ids_for_sglang_shape() -> None:
    tokenizer = _FakeTokenizer()

    top_logprobs, top_candidates = _decode_top_candidates(
        tokenizer,
        {101: -0.1, 202: -0.5},
    )

    assert top_logprobs == [-0.1, -0.5]
    assert top_candidates == [
        {"token": "tok:101", "token_id": 101, "logprob": -0.1, "bytes": list(b"tok:101")},
        {"token": "tok:202", "token_id": 202, "logprob": -0.5, "bytes": list(b"tok:202")},
    ]


def test_decode_top_candidates_handles_vllm_token_string_keys() -> None:
    tokenizer = _FakeTokenizer()

    top_logprobs, top_candidates = _decode_top_candidates(
        tokenizer,
        {"hello": -0.2, "world": -0.6},
    )

    assert top_logprobs == [-0.2, -0.6]
    assert top_candidates == [
        {"token": "hello", "token_id": None, "logprob": -0.2, "bytes": list(b"hello")},
        {"token": "world", "token_id": None, "logprob": -0.6, "bytes": list(b"world")},
    ]


async def _noop_on_chunk(event: object) -> None:
    del event


def _install_fake_inference_backends(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generate_sglang: object | None = None,
    generate_vllm: object | None = None,
) -> None:
    fake_module = types.ModuleType("rollouts.inference.backends")
    fake_module.compute_suffix_ids = lambda tokenizer: [77]
    fake_module.log_token_mismatch = lambda *args, **kwargs: None
    fake_module.tokenize_chat = (
        lambda tokenizer, messages, add_generation_prompt=False: tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    )
    fake_module.append_suffix_with_overlap = lambda token_ids, suffix_ids: token_ids + suffix_ids
    fake_module.tokenize_message_with_delimiter = (
        lambda tokenizer, message: tokenizer.apply_chat_template(
            [message],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if generate_sglang is not None:
        fake_module.generate_sglang = generate_sglang
    if generate_vllm is not None:
        fake_module.generate_vllm = generate_vllm
    monkeypatch.setitem(sys.modules, "rollouts.inference.backends", fake_module)


@pytest.mark.trio
async def test_rollout_sglang_token_level_preserves_prompt_and_output_token_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_generate_sglang(**_: object) -> _FakeGenerationOutput:
        return _FakeGenerationOutput(
            output_ids=(101, 102),
            logprobs=(-0.1, -0.2),
            top_logprobs=({101: -0.1, 201: -0.7}, {102: -0.2, 202: -0.8}),
            finish_reason="stop",
        )

    _install_fake_inference_backends(monkeypatch, generate_sglang=_fake_generate_sglang)

    actor = Actor(
        trajectory=Trajectory(messages=[Message(role="user", content="hello")]),
        endpoint=Endpoint(
            model="sglang/Qwen/Qwen3-0.6B",
            base_url="http://localhost:30000/v1",
            api_format="openai-completions",
            max_tokens=4,
            temperature=0.0,
        ),
        tools=[],
    )

    updated = await rollout_sglang_token_level(
        actor,
        _noop_on_chunk,
        tokenizer=_FakeTokenizer(),
        suffix_ids=[77],
    )

    completion = updated.trajectory.completions[-1]
    choice = completion.choices[0]
    assert completion.prompt_token_ids == (10, 11, 99)
    assert choice.token_ids == (101, 102)
    assert choice.logprobs is not None
    assert choice.logprobs.content[0].token_id == 101
    assert choice.logprobs.content[0].top_candidates[1]["token_id"] == 201
    assert choice.logprobs.content[1].top_candidates[0]["token"] == "tok:102"


@pytest.mark.trio
async def test_rollout_vllm_token_level_preserves_prompt_and_output_token_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_generate_vllm(**_: object) -> _FakeGenerationOutput:
        return _FakeGenerationOutput(
            output_ids=(301,),
            logprobs=(-0.3,),
            top_logprobs=({"candidate-a": -0.3, "candidate-b": -0.9},),
            finish_reason="length",
        )

    _install_fake_inference_backends(monkeypatch, generate_vllm=_fake_generate_vllm)

    actor = Actor(
        trajectory=Trajectory(messages=[Message(role="user", content="hello")]),
        endpoint=Endpoint(
            model="vllm/Qwen/Qwen3-0.6B",
            base_url="http://localhost:30000/v1",
            api_format="openai-completions",
            max_tokens=4,
            temperature=0.0,
        ),
        tools=[],
    )

    updated = await rollout_vllm_token_level(
        actor,
        _noop_on_chunk,
        tokenizer=_FakeTokenizer(),
        suffix_ids=[77],
    )

    completion = updated.trajectory.completions[-1]
    choice = completion.choices[0]
    assert completion.prompt_token_ids == (10, 11, 99)
    assert choice.token_ids == (301,)
    assert choice.logprobs is not None
    assert choice.logprobs.content[0].token_id == 301
    assert choice.logprobs.content[0].top_candidates[0]["token"] == "candidate-a"
    assert choice.logprobs.content[0].top_candidates[1]["token"] == "candidate-b"
