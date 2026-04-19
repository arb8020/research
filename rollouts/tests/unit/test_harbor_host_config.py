from __future__ import annotations

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
    attach_harbor_host_to_tasks,
)
from rollouts.training.types import ScoringContext

_EVALS_ROOT = Path(__file__).resolve().parents[2] / "evals"
if str(_EVALS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVALS_ROOT))

from harbor_v0.eval import _resolve_scoring_harbor_environment, _run_harbor_verifier


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
