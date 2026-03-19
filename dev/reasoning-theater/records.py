"""Task and analysis records for reasoning-theater experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MultipleChoiceTask:
    """Normalized multiple-choice task."""

    task_id: str
    question: str
    choices: dict[str, str]
    correct_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def choice_labels(self) -> tuple[str, ...]:
        return tuple(self.choices.keys())

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "choices": self.choices,
            "answer": self.correct_answer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MultipleChoiceTask":
        return cls(
            task_id=str(payload["task_id"]),
            question=str(payload["question"]),
            choices={str(k): str(v) for k, v in dict(payload["choices"]).items()},
            correct_answer=payload.get("answer"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class PrefixSlice:
    """A reasoning prefix cut point."""

    label: str
    fraction: float
    char_stop: int | None = None
    token_stop: int | None = None
    text_prefix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrefixPrediction:
    """Prediction attached to a prefix."""

    prefix_label: str
    predicted_answer: str | None
    raw_response: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
