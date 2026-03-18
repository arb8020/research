"""Terminal-bench task adapter for rollouts.

Bridges the terminal-bench task directory format to the rollouts
(Environment, Endpoint, DatasetRow) pattern.

A terminal-bench task is a directory with this structure:

    my-task/
    ├── task.yaml           # instruction, difficulty, timeouts, parser
    ├── docker-compose.yaml # container definition
    ├── run-tests.sh        # verification script
    ├── solution.sh         # reference solution
    └── tests/              # test files

To run a terminal-bench task directory through rollouts:

    from rollouts.environments.terminal_bench_adapter import (
        load_terminal_bench_tasks,
        terminal_bench_row_to_state,
    )
    from rollouts.environments.terminal_bench import TerminalBenchEnvironment

    # Load tasks from a directory of terminal-bench task dirs
    tasks = load_terminal_bench_tasks("/path/to/tasks/")

    # In your eval config:
    environment_factory = lambda row: TerminalBenchEnvironment.create(
        **terminal_bench_row_to_state(row)
    )

Or, using the deserialize path once TerminalBenchEnvironment.deserialize is implemented:

    environment_factory = lambda row: TerminalBenchEnvironment.deserialize(
        terminal_bench_row_to_state(row)
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def terminal_bench_row_to_state(row: dict[str, Any]) -> dict[str, Any]:
    """Pure function: terminal-bench dataset row → state dict for environment construction.

    The row is expected to have task_dir in row["metadata"]["task_dir"],
    as produced by load_terminal_bench_tasks().

    Returns a dict suitable for passing to TerminalBenchEnvironment.create(**state)
    or TerminalBenchEnvironment.deserialize(state) once deserialize is implemented.
    """
    task_dir = row.get("metadata", {}).get("task_dir")
    assert task_dir is not None, (
        "row missing metadata.task_dir — use load_terminal_bench_tasks() to build rows"
    )
    task_path = Path(task_dir)
    assert task_path.is_dir(), f"task_dir does not exist: {task_dir}"

    state: dict[str, Any] = {
        "task_id": task_path.name,
        "dataset_name": row.get("metadata", {}).get("dataset_name", "local"),
        "dataset_version": row.get("metadata", {}).get("dataset_version", "local"),
    }

    logging_dir = row.get("metadata", {}).get("logging_dir")
    if logging_dir is not None:
        state["logging_dir"] = logging_dir

    return state


def load_terminal_bench_tasks(
    tasks_dir: str | Path,
    *,
    dataset_name: str = "local",
    dataset_version: str = "local",
) -> list[dict[str, Any]]:
    """Load terminal-bench task directories as rollouts dataset rows.

    Scans tasks_dir for subdirectories containing a task.yaml file.
    Each becomes one row in the tasks list.

    Args:
        tasks_dir: Directory containing terminal-bench task subdirectories.
        dataset_name: Logical dataset name to embed in row metadata.
        dataset_version: Logical dataset version to embed in row metadata.

    Returns:
        List of row dicts, each with:
            name: task directory name
            problem_id: task directory name
            instruction: task instruction text (from task.yaml)
            metadata: task_dir, dataset_name, dataset_version
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to load terminal-bench tasks: pip install pyyaml") from None

    tasks_path = Path(tasks_dir)
    assert tasks_path.is_dir(), f"tasks_dir does not exist: {tasks_dir}"

    rows: list[dict[str, Any]] = []
    for task_dir in sorted(tasks_path.iterdir()):
        task_yaml = task_dir / "task.yaml"
        if not task_dir.is_dir() or not task_yaml.exists():
            continue

        task_data = yaml.safe_load(task_yaml.read_text())
        instruction = task_data.get("instruction", "")

        rows.append({
            "name": task_dir.name,
            "problem_id": task_dir.name,
            "instruction": instruction,
            "metadata": {
                "task_dir": str(task_dir),
                "dataset_name": dataset_name,
                "dataset_version": dataset_version,
                "difficulty": task_data.get("difficulty", "unknown"),
                "tags": task_data.get("tags", []),
            },
        })

    return rows
