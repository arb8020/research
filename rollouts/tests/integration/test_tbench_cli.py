import json
from pathlib import Path

import pytest

from rollouts.cli import (
    CLIConfig,
    create_session_store,
    finalize_tbench_run,
    get_tbench_agent_timeout_sec,
)
from rollouts.environments.resources import TerminalTaskResult


class _FakeTBenchEnvironment:
    def __init__(self, logging_dir: Path) -> None:
        self.logging_dir = logging_dir


def test_create_session_store_uses_tbench_logging_layout(tmp_path: Path) -> None:
    config = CLIConfig(
        env="tbench",
        tbench_surface="terminal",
        environment=_FakeTBenchEnvironment(tmp_path / "tb_run"),
    )

    store = create_session_store(config)

    assert store is not None
    assert store.base_dir == tmp_path / "tb_run" / "agent" / "sessions"
    assert store.atif_filename == "trajectory.json"
    assert store.atif_output_path == tmp_path / "tb_run" / "agent" / "trajectory.json"


def test_cli_config_defaults_tbench_surface_to_terminal() -> None:
    assert CLIConfig().tbench_surface == "terminal"


class _FakeRunnableTBenchEnvironment(_FakeTBenchEnvironment):
    def __init__(self, logging_dir: Path, timeout_sec: float = 180.0) -> None:
        super().__init__(logging_dir)
        self.max_agent_timeout_sec = timeout_sec

    async def run_tests(self) -> TerminalTaskResult:
        return TerminalTaskResult(
            score=0.5,
            success=True,
            failure_reason="",
            output="pytest output",
        )


def test_get_tbench_agent_timeout_sec_reads_environment_timeout(tmp_path: Path) -> None:
    env = _FakeRunnableTBenchEnvironment(tmp_path / "tb_run", timeout_sec=42.0)

    assert get_tbench_agent_timeout_sec(env) == 42.0


@pytest.mark.trio
async def test_finalize_tbench_run_writes_results_and_output(tmp_path: Path) -> None:
    env = _FakeRunnableTBenchEnvironment(tmp_path / "tb_run")

    exit_code = await finalize_tbench_run(env)

    assert exit_code == 0
    assert json.loads((env.logging_dir / "results.json").read_text()) == {
        "score": 0.5,
        "success": True,
        "failure_reason": "",
        "agent_timed_out": False,
    }
    assert (env.logging_dir / "test_output.txt").read_text() == "pytest output"


@pytest.mark.trio
async def test_finalize_tbench_run_marks_agent_timeout(tmp_path: Path) -> None:
    env = _FakeRunnableTBenchEnvironment(tmp_path / "tb_run")

    exit_code = await finalize_tbench_run(env, agent_timed_out=True)

    assert exit_code == 1
    assert json.loads((env.logging_dir / "results.json").read_text())["failure_reason"] == (
        "AGENT_TIMEOUT"
    )
