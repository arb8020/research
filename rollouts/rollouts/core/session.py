from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..dtypes import Endpoint, JsonSerializable, StopReason, Trajectory


def normalize_stop_reason(
    raw_stop_reason: StopReason | str | None,
    *,
    legacy_status: str | None = None,
) -> StopReason | None:
    if isinstance(raw_stop_reason, StopReason):
        return raw_stop_reason
    if raw_stop_reason not in (None, ""):
        return StopReason(str(raw_stop_reason))

    legacy = legacy_status
    if legacy in (None, "", "pending", "waiting"):
        return None
    if legacy == "completed":
        return StopReason.TASK_COMPLETED
    if legacy == "truncated":
        return StopReason.MAX_TURNS
    if legacy == "aborted":
        return StopReason.ABORTED
    if legacy == "interrupted":
        return StopReason.INTERRUPTED
    if legacy == "failed":
        return StopReason.ERROR
    raise ValueError(f"Unknown legacy session status: {legacy!r}")


def derive_session_status(stop_reason: StopReason | None, *, is_live: bool = False) -> str:
    if is_live:
        return "live"
    if stop_reason is None:
        return "pending"
    if stop_reason in (StopReason.NO_TOOL_CALLED,):
        return "pending"
    if stop_reason in (StopReason.TASK_COMPLETED, StopReason.END_TURN):
        return "completed"
    if stop_reason in (StopReason.MAX_TURNS, StopReason.BUDGET_EXCEEDED):
        return "truncated"
    if stop_reason in (StopReason.ABORTED, StopReason.USER_ABORT):
        return "aborted"
    if stop_reason == StopReason.INTERRUPTED:
        return "interrupted"
    if stop_reason in (StopReason.ERROR, StopReason.PROVIDER_ERROR, StopReason.TOOL_ERROR):
        return "failed"
    return "pending"


@dataclass(frozen=True)
class EnvironmentConfig:
    type: str
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "config": self.config}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentConfig:
        return cls(type=data["type"], config=data.get("config", {}))


@dataclass(frozen=True)
class SessionSummary(JsonSerializable):
    session_id: str
    parent_id: str | None = None
    branch_point: int | None = None
    endpoint: Endpoint = field(
        default_factory=lambda: Endpoint(model="", base_url="", api_format="")
    )
    environment: EnvironmentConfig = field(default_factory=lambda: EnvironmentConfig(type=""))
    status: str = "pending"
    tags: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    vcs: dict[str, str] | None = None
    message_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "parent_id": self.parent_id,
            "branch_point": self.branch_point,
            "endpoint": self.endpoint.to_dict(exclude_secrets=True),
            "environment": self.environment.to_dict(),
            "status": self.status,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "vcs": self.vcs,
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionSummary:
        return cls(
            session_id=data["session_id"],
            parent_id=data.get("parent_id"),
            branch_point=data.get("branch_point"),
            endpoint=Endpoint.from_dict(data["endpoint"]),
            environment=EnvironmentConfig.from_dict(data["environment"]),
            status=str(data.get("status", "pending")),
            tags=data.get("tags", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            vcs=data.get("vcs"),
            message_count=data.get("message_count"),
        )

    @classmethod
    def from_trajectory(
        cls, trajectory: Trajectory, *, message_count: int | None = None
    ) -> SessionSummary:
        return trajectory.to_session_summary(message_count=message_count)
