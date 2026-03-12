"""Supervisor command types.

Commands are requests to mutate runtime state indirectly. They are not facts;
events record the results of command execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from .model import utc_now


class CommandKind(StrEnum):
    """Supported first-pass supervisor commands."""

    LAUNCH_RUN = "launch_run"
    CANCEL_RUN = "cancel_run"
    RETRY_RUN = "retry_run"
    PUBLISH_ARTIFACT = "publish_artifact"


@dataclass(frozen=True)
class Command:
    """Durable request issued to the supervisor."""

    kind: CommandKind
    run_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    command_id: str = field(default_factory=lambda: f"cmd_{uuid4().hex[:10]}")
    created_at: datetime = field(default_factory=utc_now)
