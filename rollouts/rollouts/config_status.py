from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConfigConfidence(str, Enum):
    """How strongly we believe a config works on the current codebase."""

    DRAFT = "draft"
    IMPORT_TESTED = "import_tested"
    KNOWN_GOOD = "known_good"


@dataclass(frozen=True)
class ConfigStatus:
    """Explicit provenance for a runnable config entrypoint."""

    confidence: ConfigConfidence
    verified_commit: str | None = None
    notes: str = ""


def draft(notes: str) -> ConfigStatus:
    return ConfigStatus(confidence=ConfigConfidence.DRAFT, notes=notes)


def import_tested(verified_commit: str, notes: str) -> ConfigStatus:
    return ConfigStatus(
        confidence=ConfigConfidence.IMPORT_TESTED,
        verified_commit=verified_commit,
        notes=notes,
    )


def known_good(verified_commit: str, notes: str) -> ConfigStatus:
    return ConfigStatus(
        confidence=ConfigConfidence.KNOWN_GOOD,
        verified_commit=verified_commit,
        notes=notes,
    )
