from __future__ import annotations

from types import SimpleNamespace

import pytest
import trio
import trio_asyncio

from broker.providers import modal_image


def test_eager_build_modal_image_includes_logs_in_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class _FakeBuild:
        async def aio(self, app: object) -> object:
            del app
            raise RuntimeError("wheel build exploded")

    image = SimpleNamespace(
        object_id="im-test",
        build=_FakeBuild(),
    )

    async def fake_emit_private_modal_image_logs(
        image_obj: object,
        emit: object,
        *,
        image_id_override: str | None = None,
    ) -> list[str]:
        del image_obj, emit, image_id_override
        return [
            "Collecting sglang[all]",
            "Building wheel for flashinfer-python",
            "error: command 'g++' failed with exit status 1",
        ]

    monkeypatch.setattr(
        modal_image,
        "_emit_private_modal_image_logs",
        fake_emit_private_modal_image_logs,
    )

    async def fake_sleep(_: float) -> None:
        await modal_image.trio.lowlevel.checkpoint()

    monkeypatch.setattr(modal_image.trio, "sleep", fake_sleep)
    monkeypatch.setattr(trio_asyncio, "aio_as_trio", lambda coro: coro)

    def emit(event: str, **data: object) -> None:
        events.append((event, data))

    async def _exercise() -> None:
        with pytest.raises(RuntimeError, match="Recent Modal image build logs"):
            await modal_image.eager_build_modal_image(image, app=object(), emit=emit)

    trio.run(_exercise)

    assert any(event == "modal_image_build_failed" for event, _ in events)


def test_raise_modal_image_build_failure_preserves_original_when_logs_absent() -> None:
    exc = RuntimeError("boom")
    image = SimpleNamespace(object_id="im-test")

    with pytest.raises(RuntimeError, match="boom"):
        modal_image._raise_modal_image_build_failure(exc, image, [])


def test_extract_modal_image_id_from_remote_error() -> None:
    exc = RuntimeError("Image build for im-cAVetPJulvu9ek7UOebROV failed. See build logs.")

    assert modal_image._extract_modal_image_id(exc) == "im-cAVetPJulvu9ek7UOebROV"


def test_emit_private_modal_image_logs_uses_fresh_client_when_image_has_no_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class _FakeTaskProgress:
        pos = 0
        len = 0
        progress_type = 0

    class _FakeTaskLog:
        def __init__(self, data: str) -> None:
            self.data = data
            self.task_progress = _FakeTaskProgress()

    class _FakeResult:
        def __init__(self, status: int) -> None:
            self.status = status

    class _FakeResponse:
        def __init__(self, *, entry_id: str, status: int, lines: list[str]) -> None:
            self.entry_id = entry_id
            self.result = _FakeResult(status)
            self.task_logs = [_FakeTaskLog(line) for line in lines]

    class _FakeJoinStream:
        async def unary_stream(self, request: object):
            del request
            yield _FakeResponse(entry_id="1", status=0, lines=["line one"])
            yield _FakeResponse(entry_id="2", status=1, lines=["line two"])

    class _FakeStub:
        ImageJoinStreaming = _FakeJoinStream()

    class _FakeClient:
        stub = _FakeStub()

    async def fake_from_env() -> object:
        return _FakeClient()

    class _FakeLoop:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(trio_asyncio, "aio_as_trio", lambda coro: coro)
    monkeypatch.setattr(trio_asyncio, "open_loop", lambda: _FakeLoop())

    import modal.client

    monkeypatch.setattr(modal.client._Client, "from_env", fake_from_env)

    def emit(event: str, **data: object) -> None:
        events.append((event, data))

    async def _exercise() -> None:
        lines = await modal_image._emit_private_modal_image_logs(
            SimpleNamespace(object_id="im-test", client=None),
            emit,
        )
        assert lines == ["line one", "line two"]

    trio.run(_exercise)
    assert not any(event == "modal_image_build_logs_unavailable" for event, _ in events)
