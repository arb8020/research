from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any

from ..dtypes import Endpoint, JsonSerializable, Message, Trajectory, TrajectoryEnvironment


class SessionStatus(Enum):
    PENDING = "pending"
    WAITING = "waiting"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


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
    status: SessionStatus = SessionStatus.PENDING
    reward: float | dict[str, float] | None = None
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
            "status": self.status.value,
            "reward": self.reward,
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
            status=SessionStatus(data.get("status", "pending")),
            reward=data.get("reward"),
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


@dataclass(frozen=True)
class PendingInput(JsonSerializable):
    kind: str
    prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {"type": self.kind}
        if self.prompt is not None:
            result["prompt"] = self.prompt
        result.update(self.metadata)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingInput:
        prompt = data.get("prompt")
        metadata = {key: value for key, value in data.items() if key not in ("type", "prompt")}
        return cls(
            kind=str(data.get("type", "unknown")),
            prompt=prompt if isinstance(prompt, str) else None,
            metadata=metadata,
        )


@dataclass(frozen=True)
class SessionHandle(JsonSerializable):
    trajectory: Trajectory
    message_count: int | None = None
    pending_input: PendingInput | None = None
    queued_messages: tuple[Message, ...] = ()

    @property
    def session_id(self) -> str:
        return self.trajectory.session.session_id or ""

    @property
    def parent_id(self) -> str | None:
        return self.trajectory.session.parent_id

    @property
    def branch_point(self) -> int | None:
        return self.trajectory.session.branch_point

    @property
    def endpoint(self) -> Endpoint:
        return self.trajectory.endpoint_or_default()

    @property
    def environment(self) -> EnvironmentConfig:
        return self.trajectory.environment_config()

    @property
    def messages(self) -> list[Message]:
        return self.trajectory.messages

    @property
    def environment_state(self) -> dict[str, Any] | None:
        return self.trajectory.environment_state()

    @property
    def status(self) -> SessionStatus:
        return self.trajectory.session_status()

    @property
    def reward(self) -> float | dict[str, float] | None:
        return self.trajectory.session_reward()

    @property
    def tags(self) -> dict[str, str]:
        return dict(self.trajectory.session.tags)

    @property
    def created_at(self) -> str:
        return self.trajectory.session.created_at or datetime.now().isoformat()

    @property
    def updated_at(self) -> str:
        return self.trajectory.session.updated_at or datetime.now().isoformat()

    @property
    def vcs(self) -> dict[str, str] | None:
        return self.trajectory.session.vcs

    def to_trajectory(self) -> Trajectory:
        return self.trajectory

    def to_summary(self) -> SessionSummary:
        return self.trajectory.to_session_summary(message_count=self.message_count)

    def to_dict(self) -> dict[str, Any]:
        return self.trajectory.to_session_record()

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        messages: list[Message] | None = None,
        pending_input: PendingInput | None = None,
        queued_messages: tuple[Message, ...] = (),
    ) -> SessionHandle:
        return cls(
            trajectory=Trajectory.from_session_record(data, messages),
            pending_input=pending_input,
            queued_messages=queued_messages,
        )

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        *,
        endpoint: Endpoint | None = None,
        environment: EnvironmentConfig | None = None,
        message_count: int | None = None,
        pending_input: PendingInput | None = None,
        queued_messages: tuple[Message, ...] = (),
    ) -> SessionHandle:
        session = trajectory.session
        env_bundle = trajectory.environment
        if endpoint is not None:
            session = replace(session, endpoint=endpoint)
        if environment is not None:
            env_bundle = TrajectoryEnvironment.from_session_parts(
                environment,
                env_bundle.state if env_bundle is not None else None,
            )
        return cls(
            trajectory=replace(trajectory, session=session, environment=env_bundle),
            message_count=message_count,
            pending_input=pending_input,
            queued_messages=queued_messages,
        )
