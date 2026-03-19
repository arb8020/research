"""Megakernel Eval.

An eval where the agent writes a fused CUDA/Triton kernel that outperforms
a baseline composed implementation on a given computation.

Task:
- Agent is given a reference implementation (e.g. multi-op PyTorch or naive kernels)
- Agent writes a fused megakernel (CUDA C++ or Triton)
- Success = correctness (outputs match reference) + speedup vs baseline

Scoring:
- correctness: outputs match reference within tolerance
- speedup: kernel runtime vs baseline (primary reward signal)
- compile: kernel compiles without error

TODO: flesh out task dataset and sandbox environment.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from rollouts.training.scoring import FunctionScorer

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
TASKS_PATH = EVAL_DIR / "tasks.json"


def prepare_messages(sample_data: dict[str, Any]) -> list:
    """Prepare initial messages for the agent."""
    from rollouts.dtypes import Message

    task_name = sample_data.get("name", "unknown")
    description = sample_data.get("description", "")
    reference_code = sample_data.get("reference_code", "# TODO: reference implementation")
    verify_cmd = sample_data.get("verify_cmd", "python verify.py")

    system_prompt = """You are an expert GPU kernel engineer.

Your task is to write a fused CUDA or Triton megakernel that outperforms the given \
reference implementation.

## Tools Available
- read: Read file contents
- write: Write content to a file
- bash: Execute shell commands (CUDA compiler, nvcc, triton, pytest available)

## How Success Is Measured
1. Correctness: your kernel output must match the reference within tolerance
2. Speedup: your kernel must be faster than the baseline
3. Run the verify script to check both

DO NOT claim success without running verification and seeing passing output."""

    user_message = f"""Task: {task_name}

{description}

Reference implementation:
```python
{reference_code}
```

Verify with:
```bash
{verify_cmd}
```

Write a fused megakernel that matches the reference output and achieves the best speedup \
you can. Start by profiling the reference to understand where time is spent."""

    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_message),
    ]


def score_sample(sample: Any, _context: object) -> Any:
    """Score a completed sample.

    Looks for correctness and speedup signals in the trajectory.
    """
    from rollouts.dtypes import Metric, Score

    trajectory = getattr(sample, "trajectory", None)
    if trajectory is None:
        return Score(
            metrics=(Metric("passed", 0.0, weight=1.0, metadata={"error": "no trajectory"}),)
        )

    messages = getattr(trajectory, "messages", None)
    if messages is None:
        return Score(
            metrics=(Metric("passed", 0.0, weight=1.0, metadata={"error": "no messages"}),)
        )

    correct = False
    speedup = 0.0
    compiled = False

    # TODO: replace with structured output from verify.py once task format is settled
    # Patterns below are placeholders — adjust to match actual verify script output
    correct_pattern = re.compile(r"CORRECT|outputs match", re.IGNORECASE)
    speedup_pattern = re.compile(r"speedup[:\s]+([0-9.]+)x", re.IGNORECASE)
    compile_pattern = re.compile(r"compiled successfully|compilation ok", re.IGNORECASE)

    for msg in messages:
        content = getattr(msg, "content", None)
        if content is None:
            continue
        if isinstance(content, list):
            content = "\n".join(
                getattr(b, "text", "") or (b.get("text", "") if isinstance(b, dict) else "")
                for b in content
            )
        if not isinstance(content, str):
            continue

        if correct_pattern.search(content):
            correct = True
        if compile_pattern.search(content):
            compiled = True
        m = speedup_pattern.search(content)
        if m:
            try:
                speedup = max(speedup, float(m.group(1)))
            except ValueError:
                pass

    passed = correct and speedup > 1.0

    return Score(
        metrics=(
            Metric("passed", 1.0 if passed else 0.0, weight=1.0),
            Metric("correct", 1.0 if correct else 0.0, weight=0.0),
            Metric("compiled", 1.0 if compiled else 0.0, weight=0.0),
            Metric("speedup", speedup, weight=0.0),
        )
    )


# ── EvalSpec Definition ───────────────────────────────────────────────────────

from rollouts.eval_runner import EvalSpec

spec = EvalSpec(
    name="megakernel",
    prepare_messages=prepare_messages,
    scorer=FunctionScorer(score_sample),
    make_environment=None,  # TODO: add GPU sandbox (Modal or sandboxed worktree)
    default_tasks_path=TASKS_PATH,
    per_sample_environment=False,
)


def get_spec() -> EvalSpec:
    return spec
