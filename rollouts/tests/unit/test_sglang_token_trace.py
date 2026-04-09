from __future__ import annotations

from rollouts.providers.sglang import _decode_top_candidates


class _FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        assert len(token_ids) == 1
        return f"tok:{token_ids[0]}"


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
