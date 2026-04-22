from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import trio
from _pytest.monkeypatch import MonkeyPatch

from rollouts.environments.harbor_environment import (
    HarborEnvironment,
    HarborEnvironmentSpec,
    LocalHarborHost,
    ModalHarborHost,
    SSHHarborHost,
    _modal_sandbox_download_file,
    _modal_sandbox_upload_file,
    _patch_harbor_modal_filesystem_compat,
    attach_harbor_host_to_tasks,
)
from rollouts.training.types import ScoringContext

_EVALS_ROOT = Path(__file__).resolve().parents[2] / "evals"
if str(_EVALS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVALS_ROOT))

from harbor_v0.eval import _resolve_scoring_harbor_environment, _run_harbor_verifier


class _AsyncMethod:
    def __init__(self, fn: Callable[..., Awaitable[Any]]) -> None:
        self.aio = fn


class _FakeRemoteFile:
    def __init__(self, data: bytes = b"") -> None:
        self._buffer = bytearray(data)
        self._offset = 0
        self.closed = False
        self.read = _AsyncMethod(self._read)
        self.write = _AsyncMethod(self._write)
        self.close = _AsyncMethod(self._close)

    @property
    def data(self) -> bytes:
        return bytes(self._buffer)

    async def _read(self, n: int | None = None) -> bytes:
        if n is None:
            n = len(self._buffer) - self._offset
        start = self._offset
        end = min(start + n, len(self._buffer))
        self._offset = end
        return bytes(self._buffer[start:end])

    async def _write(self, data: bytes) -> None:
        self._buffer.extend(data)

    async def _close(self) -> None:
        self.closed = True


class _FakeSandbox:
    def __init__(self, initial_files: dict[str, bytes] | None = None) -> None:
        self.mkdir_calls: list[tuple[str, bool]] = []
        self.open_calls: list[tuple[str, str]] = []
        self._files = {path: _FakeRemoteFile(data) for path, data in (initial_files or {}).items()}
        self.mkdir = _AsyncMethod(self._mkdir)
        self.open = _AsyncMethod(self._open)

    async def _mkdir(self, path: str, parents: bool = False) -> None:
        self.mkdir_calls.append((path, parents))

    async def _open(self, path: str, mode: str) -> _FakeRemoteFile:
        self.open_calls.append((path, mode))
        if "r" in mode:
            try:
                return self._files[path]
            except KeyError as exc:
                raise FileNotFoundError(path) from exc
        if "w" in mode:
            remote_file = _FakeRemoteFile()
            self._files[path] = remote_file
            return remote_file
        raise AssertionError(f"unexpected open mode: {mode}")


def test_harbor_environment_spec_roundtrips_modal_host() -> None:
    spec = HarborEnvironmentSpec(
        task_dir="/tmp/task",
        environment_name="demo-task",
        session_id="demo-task__1234",
        host=ModalHarborHost(
            app_name="rollouts-harbor",
            secrets=("modal-secret",),
            registry_secret="registry-secret",
            volumes={"/cache": "harbor-cache"},
            sandbox_timeout_secs=123,
            sandbox_idle_timeout_secs=45,
        ),
    )

    restored = HarborEnvironmentSpec.from_dict(spec.to_dict())

    assert restored == spec


def test_harbor_environment_spec_roundtrips_ssh_stub_host() -> None:
    spec = HarborEnvironmentSpec(
        task_dir="/tmp/task",
        environment_name="demo-task",
        session_id="demo-task__1234",
        host=SSHHarborHost(
            ssh="ubuntu@example:22",
            ssh_key_path="~/.ssh/id_ed25519",
        ),
    )

    restored = HarborEnvironmentSpec.from_dict(spec.to_dict())

    assert restored == spec


def test_attach_harbor_host_to_tasks_threads_serialized_host() -> None:
    tasks = [{"task_id": "cancel-async-tasks"}]

    enriched = attach_harbor_host_to_tasks(tasks, LocalHarborHost())

    assert enriched == [
        {
            "task_id": "cancel-async-tasks",
            "harbor_host": {"kind": "local"},
        }
    ]


def test_harbor_environment_warm_roundtrip_reattaches_live_backend() -> None:
    async def _run() -> None:
        spec = HarborEnvironmentSpec(
            task_dir="/tmp/task",
            environment_name="demo-task",
            session_id="demo-task__1234",
            host=ModalHarborHost(app_name="rollouts-harbor"),
        )
        live_backend = object()
        env = HarborEnvironment._from_live_backend(
            spec=spec,
            harbor_env=live_backend,
            current_working_dir="/app/subdir",
            tools=["read", "bash"],
        )

        serialized = await env.serialize()
        json.dumps(serialized)
        restored = await HarborEnvironment.deserialize(serialized)

        assert restored.harbor_env is live_backend
        assert restored.current_working_dir == "/app/subdir"
        assert restored.spec.working_dir == "/app/subdir"
        assert restored.tools == ["read", "bash"]

    trio.run(_run)


def test_scoring_uses_live_harbor_environment_from_context() -> None:
    spec = HarborEnvironmentSpec(
        task_dir="/tmp/task",
        environment_name="demo-task",
        session_id="demo-task__1234",
        host=LocalHarborHost(),
    )
    env = HarborEnvironment._from_live_backend(
        spec=spec,
        harbor_env=object(),
        current_working_dir="/app",
    )

    resolved = _resolve_scoring_harbor_environment(
        sample=object(),
        context=ScoringContext(environment=env),
    )

    assert resolved is env


def test_run_harbor_verifier_uses_asyncio_bridge(monkeypatch: MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class FakeVerifier:
        async def verify(self) -> str:
            return "ok"

    async def fake_bridge(coro_factory: Callable[[], Awaitable[Any]]) -> Any:
        observed["result"] = await coro_factory()
        return observed["result"]

    async def _run() -> None:
        monkeypatch.setattr("harbor_v0.eval._aio_in_thread", fake_bridge)
        result = await _run_harbor_verifier(FakeVerifier())
        assert result == "ok"

    trio.run(_run)
    assert observed["result"] == "ok"


def test_modal_sandbox_upload_file_uses_open_api(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_bytes(b"hello from local")
    sandbox = _FakeSandbox()

    asyncio.run(
        _modal_sandbox_upload_file(
            sandbox=sandbox,
            source_path=source,
            target_path="/app/subdir/input.txt",
        )
    )

    assert sandbox.mkdir_calls == [("/app/subdir", True)]
    assert sandbox.open_calls == [("/app/subdir/input.txt", "wb")]
    assert sandbox._files["/app/subdir/input.txt"].data == b"hello from local"
    assert sandbox._files["/app/subdir/input.txt"].closed is True


def test_modal_sandbox_download_file_uses_open_api(tmp_path: Path) -> None:
    sandbox = _FakeSandbox(initial_files={"/logs/verifier/reward.txt": b"1.0\n"})
    target = tmp_path / "nested" / "reward.txt"

    asyncio.run(
        _modal_sandbox_download_file(
            sandbox=sandbox,
            source_path="/logs/verifier/reward.txt",
            target_path=target,
        )
    )

    assert sandbox.open_calls == [("/logs/verifier/reward.txt", "rb")]
    assert target.read_bytes() == b"1.0\n"
    assert sandbox._files["/logs/verifier/reward.txt"].closed is True


def test_patch_harbor_modal_filesystem_compat_installs_sdk_shim(tmp_path: Path) -> None:
    class _FakeSandboxClass:
        pass

    class _FakeHarborModalEnvironment:
        def __init__(self, sandbox: _FakeSandbox) -> None:
            self._sandbox = sandbox

    _patch_harbor_modal_filesystem_compat(
        harbor_modal_environment_cls=_FakeHarborModalEnvironment,
        sandbox_cls=_FakeSandboxClass,
    )

    source = tmp_path / "payload.txt"
    source.write_bytes(b"shimmed upload")
    target = tmp_path / "downloaded" / "reward.txt"
    sandbox = _FakeSandbox(initial_files={"/logs/verifier/reward.txt": b"0.5\n"})
    env = _FakeHarborModalEnvironment(sandbox)

    async def _run() -> None:
        await env._sdk_upload_file(source, "/app/payload.txt")
        await env._sdk_download_file("/logs/verifier/reward.txt", target)

    asyncio.run(_run())

    assert _FakeHarborModalEnvironment._rollouts_modal_fs_compat is True
    assert sandbox._files["/app/payload.txt"].data == b"shimmed upload"
    assert target.read_bytes() == b"0.5\n"
