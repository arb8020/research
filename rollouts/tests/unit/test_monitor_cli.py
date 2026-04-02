from __future__ import annotations

import io
from pathlib import Path

from rollouts.tui import monitor_cli


def test_resolve_job_connection_modal_skips_broker_lookup(monkeypatch: object) -> None:
    class _Node:
        provider = "modal"
        node_id = "sb-123"

    class _Job:
        job_id = "run_123"
        nodes = [_Node()]
        log_path = "results/rl/run_123"

    monkeypatch.setattr(
        monitor_cli,
        "_get_instance",
        lambda provider, node_id: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    monkeypatch.setattr("rollouts.jobs.get_job", lambda job_id: _Job())

    resolved = monitor_cli._resolve_job_connection("run_123")

    assert resolved == {
        "run_id": "run_123",
        "provider": "modal",
        "node_id": "modal:sb-123",
        "log_path": "results/rl/run_123",
        "logs_host": None,
        "logs_port": None,
    }


def test_resolve_modal_output_dir_prefers_log_path_candidate() -> None:
    class _Sandbox:
        def __init__(self) -> None:
            self.paths = {
                "/workspace/research/rollouts/results/rl/run_123",
            }

        def ls(self, path: str) -> list[str]:
            if path not in self.paths:
                raise FileNotFoundError(path)
            return ["training.jsonl", "metrics.jsonl"]

    sandbox = _Sandbox()

    resolved = monitor_cli._resolve_modal_output_dir(
        sandbox,
        run_id="run_123",
        log_path="results/rl/run_123",
    )

    assert resolved == "/workspace/research/rollouts/results/rl/run_123"


def test_sync_modal_files_once_appends_new_lines(tmp_path: Path) -> None:
    class _Sandbox:
        def __init__(self) -> None:
            self.contents = {
                "/remote/run/training.jsonl": "one\ntwo\n",
                "/remote/run/metrics.jsonl": "m1\n",
            }

        def open(self, path: str, mode: str) -> io.StringIO:
            assert mode == "r"
            if path not in self.contents:
                raise FileNotFoundError(path)
            return io.StringIO(self.contents[path])

    sandbox = _Sandbox()
    offsets = {"training.jsonl": 1, "metrics.jsonl": 0}

    files, new_lines = monitor_cli._sync_modal_files_once(
        sandbox=sandbox,
        remote_output_dir="/remote/run",
        local_sync_dir=tmp_path,
        line_offsets=offsets,
    )

    assert files == ("training.jsonl", "metrics.jsonl")
    assert new_lines == 2
    assert (tmp_path / "training.jsonl").read_text() == "two\n"
    assert (tmp_path / "metrics.jsonl").read_text() == "m1\n"
    assert offsets == {"training.jsonl": 2, "metrics.jsonl": 1}
