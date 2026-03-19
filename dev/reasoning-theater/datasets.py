"""Dataset helpers for reasoning-theater multiple-choice evaluations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from records import MultipleChoiceTask


ANSWER_RE = re.compile(r"\b([A-Z])\b")


def normalize_choices(raw_choices: Any) -> dict[str, str]:
    """Normalize common benchmark choice formats into a label -> text mapping."""
    if isinstance(raw_choices, dict):
        return {str(k).strip(): str(v).strip() for k, v in raw_choices.items()}
    if isinstance(raw_choices, list):
        labels = [chr(ord("A") + idx) for idx in range(len(raw_choices))]
        return {label: str(choice).strip() for label, choice in zip(labels, raw_choices, strict=True)}
    raise TypeError(f"Unsupported choice format: {type(raw_choices).__name__}")


def parse_multiple_choice_answer(text: str, valid_labels: Iterable[str]) -> str | None:
    """Best-effort parser for a final multiple-choice answer."""
    valid = {label.upper() for label in valid_labels}
    matches = ANSWER_RE.findall(text.upper())
    for match in reversed(matches):
        if match in valid:
            return match

    explicit_patterns = [
        re.compile(r"answer\s*[:=]\s*([A-Z])", re.IGNORECASE),
        re.compile(r"final\s+answer\s*[:=]?\s*([A-Z])", re.IGNORECASE),
        re.compile(r"\(([A-Z])\)", re.IGNORECASE),
    ]
    for pattern in explicit_patterns:
        found = pattern.search(text)
        if found:
            candidate = found.group(1).upper()
            if candidate in valid:
                return candidate
    return None


def format_multiple_choice_prompt(task: MultipleChoiceTask) -> str:
    """Canonical prompt for one multiple-choice reasoning sample."""
    choice_lines = "\n".join(f"{label}. {choice}" for label, choice in task.choices.items())
    return (
        "Answer the multiple-choice question below.\n"
        "Think step by step, then give the final answer as a single letter.\n\n"
        f"Question:\n{task.question}\n\n"
        f"Choices:\n{choice_lines}\n\n"
        "Final answer:"
    )


def load_tasks_from_jsonl(path: Path, *, limit: int | None = None) -> list[MultipleChoiceTask]:
    """Load normalized tasks from JSONL."""
    tasks: list[MultipleChoiceTask] = []
    with path.open() as handle:
        for line_idx, line in enumerate(handle):
            if limit is not None and len(tasks) >= limit:
                break
            raw = json.loads(line)
            task_id = raw.get("task_id") or raw.get("id") or f"row_{line_idx:05d}"
            task = MultipleChoiceTask(
                task_id=str(task_id),
                question=str(raw["question"]),
                choices=normalize_choices(raw["choices"]),
                correct_answer=raw.get("answer") or raw.get("correct_answer"),
                metadata=dict(raw.get("metadata", {})),
            )
            tasks.append(task)
    return tasks


def write_tasks_jsonl(tasks: Iterable[MultipleChoiceTask], path: Path) -> None:
    """Write normalized tasks to JSONL."""
    with path.open("w") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_payload()))
            handle.write("\n")
