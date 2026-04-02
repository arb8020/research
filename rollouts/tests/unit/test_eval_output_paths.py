from __future__ import annotations

import os
from pathlib import Path

import pytest

from rollouts.eval.configs import EndpointConfig, EvalOutputConfig
from rollouts.eval.run import (
    _apply_endpoint_env_overrides,
    _find_config_project_root,
    _resolve_output_dir,
)


def test_find_config_project_root_prefers_owning_repo(tmp_path: Path) -> None:
    repo_root = tmp_path / "charisma"
    config_path = repo_root / "charisma" / "configs" / "kernelbench_v3" / "eval.py"
    (repo_root / ".git").mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    config_path.write_text("# test config")

    assert _find_config_project_root(config_path) == repo_root


def test_resolve_output_dir_defaults_under_config_project_results(tmp_path: Path) -> None:
    repo_root = tmp_path / "charisma"
    config_path = repo_root / "charisma" / "configs" / "kernelbench_v3" / "eval.py"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='charisma'\n")
    config_path.parent.mkdir(parents=True)
    config_path.write_text("# test config")

    output_dir = _resolve_output_dir(
        config_path=config_path,
        output_config=EvalOutputConfig(experiment_name="kernelbench_v3_smoke_cuda"),
    )

    assert output_dir.parent == repo_root / "results"
    assert output_dir.name.startswith("kernelbench_v3_smoke_cuda_")


def test_resolve_output_dir_uses_project_root_for_relative_config_override(tmp_path: Path) -> None:
    repo_root = tmp_path / "charisma"
    config_path = repo_root / "charisma" / "configs" / "kernelbench_v3" / "eval.py"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='charisma'\n")
    config_path.parent.mkdir(parents=True)
    config_path.write_text("# test config")

    output_dir = _resolve_output_dir(
        config_path=config_path,
        output_config=EvalOutputConfig(
            experiment_name="kernelbench_v3_smoke_cuda",
            output_dir=Path("custom-results") / "smoke",
        ),
    )

    assert output_dir == repo_root / "custom-results" / "smoke"


def test_resolve_output_dir_prefers_env_over_cli_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "charisma"
    config_path = repo_root / "charisma" / "configs" / "kernelbench_v3" / "eval.py"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='charisma'\n")
    config_path.parent.mkdir(parents=True)
    config_path.write_text("# test config")

    env_output_dir = tmp_path / "env-results" / "run_123"
    monkeypatch.setenv("ROLLOUTS_OUTPUT_DIR", os.fspath(env_output_dir))

    output_dir = _resolve_output_dir(
        config_path=config_path,
        output_config=EvalOutputConfig(
            experiment_name="kernelbench_v3_smoke_cuda",
            output_dir=Path("config-results"),
        ),
        cli_output_dir=Path("cli-results"),
    )

    assert output_dir == env_output_dir


def test_apply_endpoint_env_overrides_prefers_realized_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROLLOUTS_ENDPOINT_BASE_URL", "https://example.test/v1")

    resolved = _apply_endpoint_env_overrides(
        EndpointConfig(provider="sglang", model="Qwen/Qwen2.5-7B-Instruct")
    )

    assert resolved.base_url == "https://example.test/v1"
