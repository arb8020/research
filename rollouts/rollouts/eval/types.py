"""Shared result types for external agent runs.

Kept in a leaf module so both eval/ and drivers/ can import from here
without creating a circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core import Trajectory
from ..training.types import Status


@dataclass(frozen=True)
class ExternalAttemptArtifact:
    trajectory: Trajectory
    metadata: dict[str, Any] = field(default_factory=dict)
    environment_state: dict[str, Any] | None = None
    status: Status = Status.COMPLETED
