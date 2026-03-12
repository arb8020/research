"""Canonical KernelBench problem subsets."""

from __future__ import annotations

from typing import Any

from .dataset import load_kernelbench_prompts

# Mirrors the easy multi-turn smoke subset used in Wafer's KernelBench evals.
SMOKE_PROBLEM_SUFFIXES = ("ReLU", "Softmax")


def select_problem_suffixes(
    prompts: list[dict[str, Any]],
    suffixes: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Keep prompts whose names end with one of the requested suffixes."""
    selected = [
        prompt
        for prompt in prompts
        if any(str(prompt.get("name", "")).endswith(suffix) for suffix in suffixes)
    ]
    if not selected:
        raise ValueError(f"No KernelBench prompts matched suffixes {suffixes!r}")
    return selected


def load_kernelbench_smoke_prompts(*, backend: str = "cuda") -> list[dict[str, Any]]:
    """Load the canonical easy smoke subset for KernelBench eval bring-up."""
    prompts = load_kernelbench_prompts(levels=[1], backend=backend)
    return select_problem_suffixes(prompts, SMOKE_PROBLEM_SUFFIXES)
