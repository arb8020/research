from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from rollouts.environments.resources import (
    CommandExecutionResult,
    SessionBackedWorkspaceHandle,
    SessionExecSpec,
)


@dataclass
class FakeInspectableSession:
    exec_calls: list[SessionExecSpec] = field(default_factory=list)
    uploaded: dict[str, bytes] = field(default_factory=dict)
    downloaded: dict[str, bytes] = field(default_factory=dict)
    closed: bool = False

    async def exec(self, spec: SessionExecSpec) -> CommandExecutionResult:
        self.exec_calls.append(spec)
        return CommandExecutionResult(returncode=0, stdout="ok", stderr="", cwd=spec.cwd)

    async def upload_bytes(self, remote_path: str, content: bytes) -> None:
        self.uploaded[remote_path] = content

    async def download_bytes(self, remote_path: str) -> bytes:
        return self.downloaded[remote_path]

    async def close(self) -> None:
        self.closed = True

    async def describe_runtime(self) -> dict[str, Any]:
        return {"runtime_ok": True, "kind": "fake-session"}

    def stats(self) -> dict[str, Any]:
        return {"kind": "fake-session", "exec_count": len(self.exec_calls)}

    def serialize_state(self) -> dict[str, Any]:
        return {
            "kind": "fake-session",
            "downloaded": self.downloaded,
        }

    @classmethod
    def deserialize_state(cls, data: dict[str, Any]) -> FakeInspectableSession:
        return cls(downloaded=dict(data.get("downloaded", {})))


@pytest.mark.trio
async def test_session_backed_workspace_handle_routes_ops_through_session() -> None:
    session = FakeInspectableSession(downloaded={"/workspace/result.txt": b"hello"})
    workspace = SessionBackedWorkspaceHandle(session=session, working_dir="/workspace")

    await workspace.write_file("submission.py", b"print('hi')")
    contents = await workspace.read_file("result.txt")
    result = await workspace.run("python submission.py", cwd=".", timeout=30.0)
    runtime = await workspace.describe_runtime()
    stats = workspace.stats()
    await workspace.close()

    assert session.uploaded == {"/workspace/submission.py": b"print('hi')"}
    assert contents == b"hello"
    assert result.cwd == "/workspace"
    assert session.exec_calls == [
        SessionExecSpec(
            command="python submission.py",
            cwd="/workspace",
            timeout=30.0,
            session_id=None,
            cancel_scope=None,
        )
    ]
    assert runtime == {"runtime_ok": True, "kind": "fake-session"}
    assert stats == {"kind": "fake-session", "exec_count": 1}
    assert session.closed is True


def test_session_backed_workspace_handle_round_trips_serializable_session_state() -> None:
    session = FakeInspectableSession(downloaded={"/workspace/result.txt": b"hello"})
    workspace = SessionBackedWorkspaceHandle(session=session, working_dir="/workspace")

    state = workspace.serialize_state()

    assert state == {
        "kind": "session_backed_workspace_handle",
        "working_dir": "/workspace",
        "session": {
            "kind": "fake-session",
            "downloaded": {"/workspace/result.txt": b"hello"},
        },
    }
