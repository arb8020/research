from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evals.parameter_golf.common import (
    TASK_FILES_DIRNAME,
    TASK_METADATA_FILENAME,
    TASK_PROMPT_FILENAME,
    build_prompt_text,
    default_sample,
    make_environment,
)


def test_build_prompt_mentions_core_files() -> None:
    prompt = build_prompt_text(default_sample())
    assert "README.md" in prompt
    assert "train_gpt.py" in prompt
    assert "final_int8_zlib_roundtrip" in prompt


@pytest.mark.trio
async def test_make_environment_materializes_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "parameter-golf"
    source_dir.mkdir()
    (source_dir / "README.md").write_text("hello\n")
    (source_dir / "train_gpt.py").write_text("print('train')\n")
    (source_dir / "data").mkdir()
    (source_dir / "data" / "README.md").write_text("data\n")

    workspace_base = tmp_path / "workspaces"
    monkeypatch.setenv("PARAMETER_GOLF_SOURCE_DIR", str(source_dir))
    monkeypatch.setenv("PARAMETER_GOLF_WORKSPACE_BASE", str(workspace_base))

    env = make_environment(default_sample())
    state = await env.serialize()
    workspace_dir = Path(state["working_dir"])

    try:
        assert workspace_dir.exists()
        assert (workspace_dir / "README.md").read_text() == "hello\n"
        task_dir = workspace_dir / TASK_FILES_DIRNAME
        assert (task_dir / TASK_PROMPT_FILENAME).exists()
        metadata = json.loads((task_dir / TASK_METADATA_FILENAME).read_text())
        assert metadata["sample"]["id"] == "baseline"
        assert metadata["workspace_dir"] == str(workspace_dir)
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)
