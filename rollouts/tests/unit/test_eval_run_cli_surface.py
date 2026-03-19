from __future__ import annotations

import sys

import pytest

from rollouts.eval import run as eval_run


@pytest.mark.parametrize(
    "argv",
    [
        ["rollouts.eval.run", "--config", "dummy.py", "--model", "claude-opus-4-20250514"],
        ["rollouts.eval.run", "--config", "dummy.py", "--provider", "openai"],
        ["rollouts.eval.run", "--config", "dummy.py", "--base-url", "http://localhost:30000/v1"],
        ["rollouts.eval.run", "--config", "dummy.py", "--limit", "5"],
        ["rollouts.eval.run", "--config", "dummy.py", "--max-concurrent", "4"],
        ["rollouts.eval.run", "--config", "dummy.py", "--max-turns", "20"],
        ["rollouts.eval.run", "--config", "dummy.py", "--provision"],
        ["rollouts.eval.run", "--config", "dummy.py", "--gpu-type", "H100"],
        ["rollouts.eval.run", "--config", "dummy.py", "--hardware-provider", "runpod"],
        ["rollouts.eval.run", "--config", "dummy.py", "--output-dir", "/tmp/results"],
        [
            "rollouts.eval.run",
            "launch",
            "--config",
            "dummy.py",
            "--sample",
            "0",
            "--runtime",
            "codex",
            "--model",
            "gpt-5.1-codex-mini",
        ],
    ],
)
def test_eval_run_main_rejects_removed_config_owned_flags(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        eval_run.main()

    assert excinfo.value.code == 2
