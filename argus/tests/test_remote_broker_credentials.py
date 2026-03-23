from __future__ import annotations

from pytest import MonkeyPatch

from argus import run as argus_run


def test_remote_broker_credentials_env_uses_canonical_env_names(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "broker.credentials.get_credentials",
        lambda: {
            "runpod": "runpod-key",
            "primeintellect": "prime-key",
            "digitalocean": "do-key",
        },
    )

    env = argus_run._remote_broker_credentials_env()

    assert env == {
        "RUNPOD_API_KEY": "runpod-key",
        "PRIME_API_KEY": "prime-key",
    }
