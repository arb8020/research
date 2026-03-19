from __future__ import annotations

import pytest

import rollouts.fingerprint as fingerprint_module
from rollouts.core import Endpoint, EvalConfig, Message, Metric, Score
from rollouts.fingerprint import fingerprint_eval
from rollouts.training.types import AttemptResult


class _FingerprintScorer:
    async def score(self, result: AttemptResult, context: object) -> Score:
        del result, context
        return Score(metrics=(Metric("reward", 0.0, weight=1.0),))


def _prepare_messages(sample: dict[str, str]) -> list[Message]:
    return [Message(role="user", content=sample["prompt"])]


def test_fingerprint_eval_hash_changes_when_extra_config_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fingerprint_module,
        "require_clean_git",
        lambda allow_dirty=False: ("deadbeef", False),
    )
    config = EvalConfig(
        endpoint=Endpoint(
            model="anthropic/test-model",
            base_url="https://api.anthropic.com/v1",
            api_format="anthropic-messages",
        ),
        scorer=_FingerprintScorer(),
        prepare_messages=_prepare_messages,
        max_samples=4,
    )

    fingerprint_a = fingerprint_eval(
        config,
        extra_config={"backend": "CUDA", "levels": [1], "max_turns": 8},
    )
    fingerprint_b = fingerprint_eval(
        config,
        extra_config={"backend": "CUDA", "levels": [2], "max_turns": 8},
    )

    assert fingerprint_a["config_hash"] != fingerprint_b["config_hash"]
    assert fingerprint_a["git_sha"] == "deadbeef"
    assert fingerprint_b["git_sha"] == "deadbeef"
