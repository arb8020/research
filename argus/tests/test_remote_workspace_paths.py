from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from argus import run as argus_run
from bifrost import PythonProjectMaterialization


def test_normalize_remote_workspace_root_expands_tilde() -> None:
    fake_bifrost = SimpleNamespace(expand_path=lambda path: path.replace("~", "/root", 1))

    normalized = argus_run._normalize_remote_workspace_root(
        fake_bifrost, "~/.bifrost/workspaces/rollouts-rl"
    )

    assert normalized == "/root/.bifrost/workspaces/rollouts-rl"


def test_remote_materialized_path_uses_absolute_extra_project_root() -> None:
    workspace = "/root/.bifrost/workspaces/rollouts-rl"
    project = PythonProjectMaterialization(local_root="/Users/chiraagbalu/silares_stuff/charisma")

    remote_path = argus_run._remote_materialized_path(
        local_path=Path(
            "/Users/chiraagbalu/silares_stuff/charisma/charisma/configs/kernelbench_v3/demo.py"
        ),
        workspace_root=workspace,
        extra_python_projects=(project,),
    )

    assert remote_path == (
        "/root/.bifrost/workspaces/rollouts-rl/.bifrost-extra/src/charisma/"
        "charisma/configs/kernelbench_v3/demo.py"
    )
