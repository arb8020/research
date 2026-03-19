from __future__ import annotations

import json
import os
import shlex
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rollouts.core import Message
from rollouts.environments.coding import LocalFilesystemEnvironment

DEFAULT_SOURCE_DIR = Path("/tmp/parameter-golf")
DEFAULT_WORKSPACE_BASE = Path("/tmp/rollouts-parameter-golf")
TASK_FILES_DIRNAME = ".rollouts-parameter-golf"
TASK_PROMPT_FILENAME = "TASK_PROMPT.md"
TASK_METADATA_FILENAME = "task_metadata.json"


@dataclass(frozen=True)
class ParameterGolfWorkspace:
    sample_id: str
    source_dir: Path
    workspace_dir: Path
    prompt_path: Path
    metadata_path: Path


def default_sample() -> dict[str, Any]:
    return {
        "id": "baseline",
        "name": "parameter_golf_baseline",
        "variant": "sp1024",
        "train_shards": 1,
        "local_smoke_iterations": 200,
        "train_batch_tokens": 8192,
        "val_batch_size": 8192,
        "notes": (
            "Start by understanding the baseline training loop, getting a local smoke run "
            "to complete, and only then make a small, defensible improvement."
        ),
    }


def resolve_source_dir() -> Path:
    configured = os.environ.get("PARAMETER_GOLF_SOURCE_DIR")
    source_dir = Path(configured).expanduser() if configured else DEFAULT_SOURCE_DIR
    source_dir = source_dir.resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(
            "Parameter Golf source repo not found. "
            f"Expected {source_dir}. Clone https://github.com/openai/parameter-golf "
            "there, or set PARAMETER_GOLF_SOURCE_DIR."
        )
    return source_dir


def resolve_workspace_base() -> Path:
    configured = os.environ.get("PARAMETER_GOLF_WORKSPACE_BASE")
    base = Path(configured).expanduser() if configured else DEFAULT_WORKSPACE_BASE
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def sample_id_for(sample_data: dict[str, Any]) -> str:
    for key in ("id", "problem_id", "name"):
        value = sample_data.get(key)
        if isinstance(value, str) and value:
            return value
    return "parameter-golf"


def build_prompt_text(sample_data: dict[str, Any], workspace_dir: Path | None = None) -> str:
    variant = sample_data.get("variant", "sp1024")
    train_shards = sample_data.get("train_shards", 1)
    iterations = sample_data.get("local_smoke_iterations", 200)
    train_batch_tokens = sample_data.get("train_batch_tokens", 8192)
    val_batch_size = sample_data.get("val_batch_size", 8192)
    workspace_hint = f"The workspace is {workspace_dir}." if workspace_dir is not None else ""

    return (
        "You are working on OpenAI's Parameter Golf challenge repository.\n\n"
        "Goal:\n"
        "- understand the baseline training/evaluation loop\n"
        "- get a small local smoke run working\n"
        "- make one small, defensible improvement or prepare a clean experiment plan\n\n"
        f"{workspace_hint}\n\n"
        "Start by reading:\n"
        "- README.md\n"
        "- data/README.md\n"
        "- train_gpt.py\n\n"
        "Then get a smoke run working with the smallest honest local setup you can. "
        "A good baseline is:\n"
        f"1. download cached FineWeb with variant `{variant}` and `--train-shards {train_shards}`\n"
        "2. run a short training job\n"
        "3. inspect the final `final_int8_zlib_roundtrip` output and submission size\n\n"
        "Suggested smoke command:\n"
        "```bash\n"
        f"python3 data/cached_challenge_fineweb.py --variant {variant} --train-shards {train_shards}\n"
        f"RUN_ID=smoke_{sample_id_for(sample_data)} \\\n"
        f"ITERATIONS={iterations} \\\n"
        f"TRAIN_BATCH_TOKENS={train_batch_tokens} \\\n"
        "VAL_LOSS_EVERY=0 \\\n"
        f"VAL_BATCH_SIZE={val_batch_size} \\\n"
        "python3 train_gpt_mlx.py\n"
        "```\n\n"
        "If you are not on Apple Silicon, adapt toward the CUDA `train_gpt.py` path instead of "
        "pretending the MLX path is available.\n\n"
        "Do not assume success without running real commands. Prefer small, inspectable edits."
    )


def prepare_messages(sample_data: dict[str, Any]) -> list[Message]:
    system_prompt = (
        "You are a coding agent working inside an isolated copy of the Parameter Golf repo. "
        "Be rigorous about reading the repository before changing it. "
        "Prefer honest local smoke runs over speculative large rewrites."
    )
    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=build_prompt_text(sample_data)),
    ]


def _write_task_files(workspace_dir: Path, sample_data: dict[str, Any]) -> ParameterGolfWorkspace:
    task_dir = workspace_dir / TASK_FILES_DIRNAME
    task_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = task_dir / TASK_PROMPT_FILENAME
    prompt_path.write_text(build_prompt_text(sample_data, workspace_dir) + "\n")

    metadata = {
        "sample": sample_data,
        "source_dir": str(resolve_source_dir()),
        "workspace_dir": str(workspace_dir),
        "prompt_path": str(prompt_path),
    }
    metadata_path = task_dir / TASK_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    return ParameterGolfWorkspace(
        sample_id=sample_id_for(sample_data),
        source_dir=resolve_source_dir(),
        workspace_dir=workspace_dir,
        prompt_path=prompt_path,
        metadata_path=metadata_path,
    )


def materialize_workspace(sample_data: dict[str, Any]) -> ParameterGolfWorkspace:
    source_dir = resolve_source_dir()
    workspace_base = resolve_workspace_base()
    sample_id = sample_id_for(sample_data)
    workspace_dir = workspace_base / f"{sample_id}-{uuid.uuid4().hex[:8]}"
    shutil.copytree(source_dir, workspace_dir, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    return _write_task_files(workspace_dir, sample_data)


def make_environment(sample_data: dict[str, Any]) -> LocalFilesystemEnvironment:
    workspace = materialize_workspace(sample_data)
    return LocalFilesystemEnvironment(working_dir=workspace.workspace_dir, tools="full")


def build_rollouts_sdk_command(workspace: ParameterGolfWorkspace) -> str:
    return (
        f"cat {shlex.quote(str(workspace.prompt_path))} | "
        f"uv run rollouts --env coding --cwd {shlex.quote(str(workspace.workspace_dir))}"
    )
