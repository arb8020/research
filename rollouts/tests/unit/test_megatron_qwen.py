from __future__ import annotations

from argparse import Namespace

import pytest

from rollouts.training.backends.megatron import qwen


def test_disable_te_only_qwen_features_when_transformer_engine_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = Namespace(apply_rope_fusion=True)
    monkeypatch.setattr(qwen, "_transformer_engine_available", lambda: False)

    qwen._disable_te_only_qwen_features_when_unavailable(args)

    assert args.apply_rope_fusion is False


def test_disable_te_only_qwen_features_keeps_flag_when_transformer_engine_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = Namespace(apply_rope_fusion=True)
    monkeypatch.setattr(qwen, "_transformer_engine_available", lambda: True)

    qwen._disable_te_only_qwen_features_when_unavailable(args)

    assert args.apply_rope_fusion is True
