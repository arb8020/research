from __future__ import annotations

import pytest

from rollouts.run import _read_remote_manifest


class _FakeBifrost:
    def __init__(self, response: str) -> None:
        self.response = response

    def exec(self, _command: str) -> str:
        return self.response


def test_read_remote_manifest_returns_none_when_missing() -> None:
    assert _read_remote_manifest(_FakeBifrost("")) is None


def test_read_remote_manifest_fails_loud_on_invalid_json() -> None:
    with pytest.raises(RuntimeError, match="unreadable"):
        _read_remote_manifest(_FakeBifrost("{not-json"))
