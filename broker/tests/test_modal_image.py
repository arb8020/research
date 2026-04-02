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
