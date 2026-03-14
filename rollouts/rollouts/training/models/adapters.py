"""Backend-neutral model construction adapter protocol."""

from __future__ import annotations

from typing import Protocol

from .denotation import ModelDenotation


class ModelConstructionAdapter(Protocol):
    """Protocol for backend-specific model construction adapters.

    Adapters own model-family normalization and compatibility validation at the
    backend boundary. They do not own parallel/layout lowering semantics.
    """

    adapter_name: str

    def normalize_denotation(self, denotation: ModelDenotation) -> ModelDenotation:
        """Normalize external model metadata into trusted denotation."""
        ...

    def validate_support(self, denotation: ModelDenotation, *, backend_name: str) -> None:
        """Reject unsupported model/backend combinations early."""
        ...
