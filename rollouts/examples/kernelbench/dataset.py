"""KernelBench dataset loading.

Loads KernelBench problems from HuggingFace and formats them as prompts.

Dataset: https://huggingface.co/datasets/ScalingIntelligence/KernelBench
"""

from __future__ import annotations

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
    # Import HuggingFace datasets, not the local rollouts.datasets
    import sys
    # Temporarily remove the rollouts package paths that shadow HF datasets
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if "rollouts" not in p]
    try:
        from datasets import load_dataset as hf_load_dataset
    finally:
        sys.path = original_path
    load_dataset = hf_load_dataset

    # KernelBench has per-level splits: level_1, level_2, level_3, level_4
    if levels is None:
        levels = [1, 2, 3, 4]

    all_problems = []
    for level in levels:
        split_name = f"level_{level}"
        ds = load_dataset("ScalingIntelligence/KernelBench", split=split_name)
        for row in ds:  # type: ignore[union-attr]
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


def load_kernelbench_prompts(
    levels: list[int] | None = None,
    max_samples: int | None = None,
    backend: str = "CUDA",
) -> list[dict[str, Any]]:
    """Load KernelBench problems as prompt dicts for evaluation/training.

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

    prompts = []
    for problem in problems:
        # Format as chat messages
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
            "ref_code": problem["code"],  # Keep reference for evaluation
        })

    return prompts
