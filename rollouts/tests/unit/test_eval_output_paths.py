from __future__ import annotations

from pathlib import Path

from rollouts.eval.configs import EvalOutputConfig
from rollouts.eval.run import _find_config_project_root, _resolve_output_dir


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
