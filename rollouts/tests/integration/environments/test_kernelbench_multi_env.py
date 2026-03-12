from __future__ import annotations

import pytest

from rollouts.agents import AgentState
from rollouts.agents.types import Actor
from rollouts.core import Endpoint, Message, Trajectory
from rollouts.dtypes import StopReason, ToolCall
from rollouts.environments.kernelbench_multi import KernelBenchMultiTurnEnvironment
from rollouts.environments.modal_sandbox_resource import (
    ModalSandboxManager,
    ModalSandboxResourceConfig,
)
from rollouts.environments.resources import CommandExecutionResult


class FakeKernelEvaluator:
    def __init__(self) -> None:
        self.start_calls = 0
        self.score_calls: list[tuple[str, str, float]] = []
        self.runtime_provenance = {
            "hostname": "gpu-box-1",
            "machine": "x86_64",
            "torch": {
                "version": "2.8.0",
                "cuda_version": "12.4",
                "device_name": "NVIDIA A100",
                "device_capability": [8, 0],
            },
        }

    async def start(self) -> None:
        self.start_calls += 1

    async def score_one(
        self,
        kernel_code: str,
        ref_code: str,
        timeout: float,
    ) -> dict[str, float | str | None]:
        self.score_calls.append((kernel_code, ref_code, timeout))
        return {
            "compiled": 1.0,
            "correct": 1.0,
            "speedup": 1.75,
            "pass_rate": 1.0,
            "error": None,
            "runtime_provenance": self.runtime_provenance,
        }


class FakeKernelWorkspace:
    def __init__(self) -> None:
        self.working_dir = "/workspace"
        self.started = 0
        self.closed = 0
        self.writes: list[tuple[str, bytes]] = []

    async def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1

    async def describe_runtime(self) -> dict[str, object]:
        return {"runtime_ok": True, "kind": "fake-modal"}

    def stats(self) -> dict[str, object]:
        return {
            "kind": "fake-modal",
            "started": self.started > 0,
            "start_attempts": self.started,
            "start_failures": 0,
        }

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        if path.startswith("/"):
            return path
        return f"{current_working_dir}/{path}"

    async def read_file(self, path: str) -> bytes:
        for candidate_path, content in reversed(self.writes):
            if candidate_path == path:
                return content
        raise FileNotFoundError(path)

    async def write_file(self, path: str, content: bytes) -> None:
        self.writes.append((path, content))

    async def run(
        self,
        command: str,
        *,
        cwd: str,
        timeout: float,
        session_id: str | None = None,
        cancel_scope: object | None = None,
    ) -> CommandExecutionResult:
        del timeout, session_id, cancel_scope
        return CommandExecutionResult(returncode=0, stdout=command, stderr="", cwd=cwd)


class FakeManagedModalResource(FakeKernelWorkspace):
    def __init__(
        self,
        config: ModalSandboxResourceConfig,
        workspace_setup: object | None,
    ) -> None:
        del workspace_setup
        super().__init__()
        self.config = config

    async def prepare(self, sample_data: dict[str, object] | None = None) -> None:
        await self.start()

    def stats(self) -> dict[str, object]:
        return {
            "kind": "fake-managed-modal-resource",
            "started": self.started > 0,
        }

    def serialize_state(self) -> dict[str, object]:
        return {
            "kind": "modal_sandbox_resource",
            "config": {
                "app_name": self.config.app_name,
                "gpu": self.config.gpu,
                "timeout_seconds": self.config.timeout_seconds,
                "python_version": self.config.python_version,
                "workspace_dir": self.config.workspace_dir,
                "image_registry": self.config.image_registry,
                "apt_packages": list(self.config.apt_packages),
                "pip_packages": list(self.config.pip_packages),
                "env": self.config.env,
            },
            "sample_data": {},
            "working_dir": self.working_dir,
            "sandbox_id": None,
            "started": True,
            "start_attempts": self.started,
            "start_failures": 0,
            "provision_duration_ms": None,
            "last_error": None,
            "runtime": {"runtime_ok": True, "kind": "fake-modal"},
        }


@pytest.mark.trio
async def test_kernelbench_uses_injected_evaluator() -> None:
    evaluator = FakeKernelEvaluator()
    env = KernelBenchMultiTurnEnvironment(
        ref_code="def ref(): pass",
        evaluator=evaluator,
    )

    await env.on_session_start("session-1")
    result = await env._evaluate_kernel("class ModelNew: pass")

    assert evaluator.start_calls == 1
    assert evaluator.score_calls == [("class ModelNew: pass", "def ref(): pass", 120.0)]
    assert result["compiled"] is True
    assert result["correct"] is True
    assert result["speedup"] == 1.75
    assert result["pass_rate"] == 1.0
    assert result["error"] is None
    assert result["runtime_provenance"] == evaluator.runtime_provenance
    assert result["debug_stdout_tail"] is None
    assert result["debug_stderr_tail"] is None
    assert result["returncode"] is None


@pytest.mark.trio
async def test_kernelbench_persists_evaluator_provenance_in_trajectory_metadata() -> None:
    evaluator = FakeKernelEvaluator()
    env = KernelBenchMultiTurnEnvironment(
        ref_code="def ref(): pass",
        evaluator=evaluator,
    )
    state = AgentState(
        actor=Actor(
            trajectory=Trajectory(messages=[]),
            endpoint=Endpoint(
                model="anthropic/test-model",
                base_url="https://api.anthropic.com/v1",
                api_format="anthropic-messages",
            ),
            tools=[],
        ),
        environment=env,
    )

    next_state = await env.on_assistant_message(
        Message(
            role="assistant",
            content="```python\nclass ModelNew: pass\n```",
        ),
        state,
    )

    assert env.evaluator_provenance == evaluator.runtime_provenance
    assert (
        next_state.actor.trajectory.metadata["evaluator_provenance"] == evaluator.runtime_provenance
    )
    turn_history = next_state.actor.trajectory.metadata["turn_history"]
    assert len(turn_history) == 1
    assert turn_history[0]["turn"] == 1
    assert turn_history[0]["has_code"] is True
    assert turn_history[0]["compiled"] is True
    assert turn_history[0]["correct"] is True
    assert turn_history[0]["speedup"] == 1.75
    assert turn_history[0]["error"] is None
    assert turn_history[0]["runtime_provenance"] == evaluator.runtime_provenance
    assert turn_history[0]["debug_stdout_tail"] is None
    assert turn_history[0]["debug_stderr_tail"] is None
    assert turn_history[0]["returncode"] is None


@pytest.mark.trio
async def test_kernelbench_write_kernel_tool_uses_injected_workspace() -> None:
    evaluator = FakeKernelEvaluator()
    workspace = FakeKernelWorkspace()
    env = KernelBenchMultiTurnEnvironment(
        ref_code="def ref(): pass",
        evaluator=evaluator,
        kernel_workspace=workspace,
    )
    state = AgentState(
        actor=Actor(
            trajectory=Trajectory(messages=[]),
            endpoint=Endpoint(
                model="anthropic/test-model",
                base_url="https://api.anthropic.com/v1",
                api_format="anthropic-messages",
            ),
            tools=env.get_tools(),
        ),
        environment=env,
    )

    result = await env.exec_tool(
        ToolCall(
            id="write-1",
            name="write_kernel",
            args={"kernel_code": "class ModelNew: pass"},
        ),
        state,
        run_config=None,
    )

    assert workspace.started == 1
    assert workspace.writes == [("/workspace/kernel_submission.py", b"class ModelNew: pass")]
    assert result.is_error is False
    assert result.details is not None
    assert result.details["compiled"] is True
    assert result.details["correct"] is True
    assert result.details["submission_path"] == "/workspace/kernel_submission.py"
    assert env.current_turn == 1
    assert env.best_kernel == "class ModelNew: pass"
    assert env.turn_history[0]["submission_path"] == "/workspace/kernel_submission.py"
    assert env.sandbox_runtime_provenance == {"runtime_ok": True, "kind": "fake-modal"}
    assert env.sandbox_resource_stats == {
        "kind": "fake-modal",
        "started": True,
        "start_attempts": 1,
        "start_failures": 0,
    }


@pytest.mark.trio
async def test_kernelbench_write_kernel_finalizes_on_done_message() -> None:
    evaluator = FakeKernelEvaluator()
    workspace = FakeKernelWorkspace()
    env = KernelBenchMultiTurnEnvironment(
        ref_code="def ref(): pass",
        evaluator=evaluator,
        kernel_workspace=workspace,
    )
    state = AgentState(
        actor=Actor(
            trajectory=Trajectory(messages=[]),
            endpoint=Endpoint(
                model="anthropic/test-model",
                base_url="https://api.anthropic.com/v1",
                api_format="anthropic-messages",
            ),
            tools=env.get_tools(),
        ),
        environment=env,
    )

    await env.on_session_start("session-1")
    await env.exec_tool(
        ToolCall(
            id="write-1",
            name="write_kernel",
            args={"kernel_code": "class ModelNew: pass"},
        ),
        state,
        run_config=None,
    )
    final_state = await env.on_assistant_message(
        Message(role="assistant", content="I am done."),
        state,
    )

    assert final_state.stop == StopReason.TASK_COMPLETED
    assert final_state.actor.trajectory.metadata["best_kernel"] == "class ModelNew: pass"
    assert final_state.actor.trajectory.metadata["submission_path"] == "/workspace/kernel_submission.py"
    assert final_state.actor.trajectory.metadata["sandbox_resource_stats"] == {
        "kind": "fake-modal",
        "started": True,
        "start_attempts": 1,
        "start_failures": 0,
    }


@pytest.mark.trio
async def test_kernelbench_deserialize_restores_injected_resources() -> None:
    manager = ModalSandboxManager(
        config=ModalSandboxResourceConfig(gpu="A100"),
        resource_factory=FakeManagedModalResource,
    )
    workspace = manager.make_resource({})
    await workspace.start()
    env = KernelBenchMultiTurnEnvironment(
        ref_code="def ref(): pass",
        kernel_workspace=workspace,
    )

    data = await env.serialize()
    restored = await KernelBenchMultiTurnEnvironment.deserialize(data)
    await restored.initialize("session-1")

    assert restored.kernel_workspace is not None
    assert restored.evaluator is not None


@pytest.mark.trio
async def test_kernelbench_initialize_eagerly_starts_workspace() -> None:
    workspace = FakeKernelWorkspace()
    env = KernelBenchMultiTurnEnvironment(
        ref_code="def ref(): pass",
        evaluator=FakeKernelEvaluator(),
        kernel_workspace=workspace,
    )

    await env.initialize("session-1")

    assert workspace.started == 1
    assert env.sandbox_runtime_provenance == {"runtime_ok": True, "kind": "fake-modal"}


@pytest.mark.trio
async def test_kernelbench_cleanup_releases_workspace() -> None:
    env = KernelBenchMultiTurnEnvironment(
        ref_code="def ref(): pass",
        evaluator=FakeKernelEvaluator(),
        kernel_workspace=FakeKernelWorkspace(),
    )

    await env.cleanup()

    assert env.kernel_workspace is not None
    assert env.kernel_workspace.closed == 1
