"""Calculator smoke test dataset transformation.

Simple math problems for testing SGLang provider with calculator environment.
"""

import json
from pathlib import Path
from typing import Any

from ..dtypes import Message, Trajectory


def load_calc_smoke(data_path: Path) -> list[dict[str, Any]]:
    """Load calc_smoke dataset from JSONL file.

    Why JSONL: One problem per line, easy to append/inspect.
    Tiger Style: Explicit file format, no hidden parsing.
    """
    assert data_path is not None
    assert isinstance(data_path, Path)
    assert data_path.exists(), f"Dataset not found: {data_path}"

    problems = []
    with open(data_path / "problems.jsonl") as f:
        for line in f:
            assert line is not None
            assert len(line.strip()) > 0

            problem = json.loads(line)
            assert "question" in problem
            assert "answer" in problem
            problems.append(problem)

    assert len(problems) > 0, "Dataset is empty"
    return problems


def calc_smoke_to_trajectory(row: dict[str, Any]) -> Trajectory:
    """Transform calc_smoke row to initial trajectory.

    Why Message format: Agent expects chat-style messages.
    System message: Instructs agent to use calculator tools.
    User message: The actual math problem to solve.
    """
    assert row is not None
    assert isinstance(row, dict)
    assert "question" in row, "Row must have 'question' field"

    question = row["question"]
    assert isinstance(question, str)
    assert len(question) > 0

    # System message instructs agent to use calculator tools
    system_msg = Message(
        role="system",
        content="You are a helpful math assistant. Use the calculator tools (add, subtract, multiply, divide) to solve problems accurately.",
    )

    # User message contains the question
    user_msg = Message(role="user", content=question)

    trajectory = Trajectory(messages=[system_msg, user_msg])
    assert trajectory is not None
    assert len(trajectory.messages) == 2

    return trajectory
