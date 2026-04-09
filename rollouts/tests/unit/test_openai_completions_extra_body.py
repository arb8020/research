from __future__ import annotations

from types import SimpleNamespace

import pytest

from rollouts.agents.types import Actor
from rollouts.core import Endpoint, Message, Trajectory
from rollouts.dtypes import ChatCompletion, Choice, StreamDone, StreamStart, TextDelta, Usage
from rollouts.providers import openai_completions


class _CaptureCreate:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    async def create(self, **kwargs: object) -> object:
        self.kwargs = dict(kwargs)
        return object()


@pytest.mark.trio
async def test_rollout_openai_passes_extra_params_via_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _CaptureCreate()

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.chat = SimpleNamespace(completions=capture)

    async def _fake_aggregate_stream(
        stream: object,
        on_chunk: object,
        request_start: float,
    ) -> tuple[ChatCompletion, float | None]:
        del stream, on_chunk, request_start
        completion = ChatCompletion(
            id="test",
            object="chat.completion",
            created=0,
            model="",
            usage=Usage(input_tokens=1, output_tokens=1),
            choices=[Choice(0, Message(role="assistant", content="olleh"), "stop", logprobs=None)],
        )
        return completion, None

    monkeypatch.setattr(openai_completions, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(openai_completions, "aggregate_stream", _fake_aggregate_stream)

    actor = Actor(
        trajectory=Trajectory(messages=[Message(role="user", content="hello")]),
        endpoint=Endpoint(
            model="sglang/Qwen/Qwen3-0.6B",
            base_url="http://localhost:30000/v1",
            api_format="openai-completions",
            extra_params={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        tools=[],
    )

    async def _on_chunk(event: object) -> None:
        del event

    updated = await openai_completions.rollout_openai(actor, _on_chunk)

    assert capture.kwargs is not None
    assert "chat_template_kwargs" not in capture.kwargs
    assert capture.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert updated.trajectory.messages[-1].role == "assistant"


@pytest.mark.trio
async def test_aggregate_openai_compatible_sse_accepts_minisglang_style_chunks() -> None:
    async def _stream() -> object:
        chunks = [
            ' {"id":"cmpl-1","object":"text_completion.chunk","choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]} ',
            '{"id":"cmpl-1","object":"text_completion.chunk","choices":[{"delta":{"content":"olle"},"index":0,"finish_reason":null}]}',
            '{"id":"cmpl-1","object":"text_completion.chunk","choices":[{"delta":{"content":"h"},"index":0,"finish_reason":null}]}',
            '{"id":"cmpl-1","object":"text_completion.chunk","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}',
            "[DONE]",
        ]
        for chunk in chunks:
            yield chunk

    events: list[object] = []

    async def _on_chunk(event: object) -> None:
        events.append(event)

    completion, _ttft_ms = await openai_completions.aggregate_openai_compatible_sse(
        _stream(),
        _on_chunk,
    )

    assert completion.choices[0].message.content == "olleh"
    assert isinstance(events[0], StreamStart)
    assert any(isinstance(event, TextDelta) and event.delta == "olle" for event in events)
    assert isinstance(events[-1], StreamDone)


@pytest.mark.trio
async def test_aggregate_stream_preserves_top_logprob_candidates() -> None:
    chunk = SimpleNamespace(
        id="cmpl-1",
        created=123,
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="stub", tool_calls=None),
                finish_reason="stop",
                logprobs=SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            token="stub",
                            logprob=-0.02,
                            bytes=[115, 116, 117, 98],
                            token_id=42,
                            top_logprobs=[
                                SimpleNamespace(
                                    token="stub", logprob=-0.02, bytes=[115], token_id=42
                                ),
                                SimpleNamespace(
                                    token="hello", logprob=-4.4, bytes=[104], token_id=99
                                ),
                            ],
                        )
                    ]
                ),
            )
        ],
    )

    async def _stream() -> object:
        yield chunk

    events: list[object] = []

    async def _on_chunk(event: object) -> None:
        events.append(event)

    completion, _ttft_ms = await openai_completions.aggregate_stream(_stream(), _on_chunk)

    logprob = completion.choices[0].logprobs.content[0]
    assert logprob.token == "stub"
    assert logprob.token_id == 42
    assert logprob.top_candidates[0]["token"] == "stub"
    assert logprob.top_candidates[1]["token"] == "hello"
    assert completion.choices[0].token_ids == (42,)
