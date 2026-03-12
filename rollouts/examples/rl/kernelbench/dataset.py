"""KernelBench dataset loading.

Supports two dataset surfaces:
- HuggingFace `ScalingIntelligence/KernelBench`
- filesystem-backed `KernelBench-v3`
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .prompts import format_system_prompt, format_user_prompt


def load_kernelbench_dataset(
    levels: list[int] | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load KernelBench problems from HuggingFace.

    Args:
        levels: List of levels to load (1-4). Default: all levels.
        max_samples: Maximum total samples to load.

    Returns:
        List of dicts with:
        - "code": str (full Python code with Model, get_inputs, get_init_inputs)
        - "level": int
        - "name": str
        - "problem_id": int
    """
    from datasets import load_dataset

    # KernelBench has per-level splits: level_1, level_2, level_3, level_4
    if levels is None:
        levels = [1, 2, 3, 4]

    all_problems = []
    for level in levels:
        split_name = f"level_{level}"
        ds = load_dataset("ScalingIntelligence/KernelBench", split=split_name)
        for row in ds:
            all_problems.append({
                "code": row["code"],
                "level": row["level"],
                "name": row["name"],
                "problem_id": row["problem_id"],
            })

    # Limit samples
    if max_samples and len(all_problems) > max_samples:
        all_problems = all_problems[:max_samples]

    return all_problems


def _resolve_kernelbench_v3_root(root_path: str | os.PathLike[str] | None) -> Path:
    """Resolve the KernelBench-v3 repo root or `problems/` directory.

    The caller should pass either:
    - the repo root containing `problems/level*/`
    - or the `problems/` directory directly
    """
    candidate = root_path or os.environ.get("KERNELBENCH_V3_ROOT")
    if candidate is None:
        raise ValueError(
            "KernelBench-v3 root is required. Pass root_path=... or set KERNELBENCH_V3_ROOT."
        )

    root = Path(candidate).expanduser().resolve()
    if (root / "problems").is_dir():
        return root
    if root.name == "problems" and root.is_dir():
        return root.parent

    raise ValueError(
        f"KernelBench-v3 root {root} is invalid; expected a repo root or problems/ directory."
    )


def _v3_problem_row(problem_path: Path, level: int) -> dict[str, Any]:
    stem = problem_path.stem
    prefix, _, suffix = stem.partition("_")
    try:
        problem_id = int(prefix)
        problem_name = suffix or stem
    except ValueError:
        problem_id = -1
        problem_name = stem

    return {
        "code": problem_path.read_text(),
        "level": level,
        "name": stem,
        "problem_name": problem_name,
        "problem_id": problem_id,
        "dataset": "kernelbench_v3",
        "problem_path": str(problem_path),
    }


def load_kernelbench_v3_dataset(
    *,
    root_path: str | os.PathLike[str] | None = None,
    levels: list[int] | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load KernelBench-v3 problems from a local checkout."""
    repo_root = _resolve_kernelbench_v3_root(root_path)
    if levels is None:
        levels = [1, 2, 3, 4]

    all_problems: list[dict[str, Any]] = []
    for level in levels:
        level_dir = repo_root / "problems" / f"level{level}"
        if not level_dir.is_dir():
            raise ValueError(f"KernelBench-v3 level directory not found: {level_dir}")
        for problem_path in sorted(level_dir.glob("*.py")):
            all_problems.append(_v3_problem_row(problem_path, level))

    if max_samples is not None and len(all_problems) > max_samples:
        return all_problems[:max_samples]
    return all_problems


def _problems_to_prompts(
    problems: list[dict[str, Any]],
    *,
    backend: str,
) -> list[dict[str, Any]]:
    prompts = []
    for problem in problems:
        system_msg = {"role": "system", "content": format_system_prompt(backend)}
        user_msg = {
            "role": "user",
            "content": format_user_prompt(
                name=problem["name"],
                level=f"level{problem['level']}",
                ref_arch_src=problem["code"],
                backend=backend,
            ),
        }

        prompts.append({
            "messages": [system_msg, user_msg],
            "problem_id": problem["problem_id"],
            "level": problem["level"],
            "name": problem["name"],
            "problem_name": problem.get("problem_name", problem["name"]),
            "ref_code": problem["code"],
            "metadata": {
                "dataset": problem.get("dataset", "kernelbench"),
                "problem_path": problem.get("problem_path"),
            },
        })
    return prompts


def load_kernelbench_prompts(
    levels: list[int] | None = None,
    max_samples: int | None = None,
    backend: str = "CUDA",
) -> list[dict[str, Any]]:
    """Load KernelBench problems as prompt dicts for GRPO training.

    Args:
        levels: List of levels to load (default: [1] for easiest)
        max_samples: Maximum total samples to load
        backend: Backend for kernels ("CUDA" or "HIP")

    Returns:
        List of prompt dicts with:
        - "messages": [system_msg, user_msg]
        - "problem_id": int
        - "level": int
        - "name": str
        - "ref_code": str (reference code for evaluation)
    """
    if levels is None:
        levels = [1]

    problems = load_kernelbench_dataset(levels=levels, max_samples=max_samples)

    return _problems_to_prompts(problems, backend=backend)


def load_kernelbench_v3_prompts(
    *,
    root_path: str | os.PathLike[str] | None = None,
    levels: list[int] | None = None,
    max_samples: int | None = None,
    backend: str = "CUDA",
) -> list[dict[str, Any]]:
    """Load KernelBench-v3 problems as prompt dicts for eval/training."""
    problems = load_kernelbench_v3_dataset(
        root_path=root_path,
        levels=levels,
        max_samples=max_samples,
    )
    return _problems_to_prompts(problems, backend=backend)
