from __future__ import annotations

import pytest

from examples.rl.kernelbench.resources import (
    KernelBenchRolloutResources,
    KernelBenchRuntimeRequirements,
)
from rollouts.environments.kernelbench_multi import SandboxPoolKernelEvaluator
from rollouts.environments.modal_sandbox_resource import (
    ModalSandboxManager,
    ModalSandboxResourceConfig,
)
from rollouts.environments.resources import CommandExecutionResult


class FakePool:
    def __init__(self, *, is_local: bool, runtime_descriptions: list[dict[str, object]]) -> None:
        self.is_local = is_local
        self._runtime_descriptions = runtime_descriptions
        self.started = False

    async def ensure_capacity(self) -> None:
        self.started = True

    async def describe_runtime(self) -> list[dict[str, object]]:
        return self._runtime_descriptions

    async def stop(self) -> None:
        self.started = False

    def stats(self) -> dict[str, object]:
        return {"started": self.started}


def _healthy_runtime() -> dict[str, object]:
    return {
        "runtime_ok": True,
        "thunderkittens_root_exists": True,
        "torch": {
            "available": True,
            "version": "2.8.0",
            "cuda_available": True,
            "cuda_version": "12.4",
            "device_count": 1,
            "device_name": "NVIDIA A100",
            "ninja_available": True,
            "triton_available": True,
            "cupy_available": True,
            "tilelang_available": True,
            "cutlass_available": True,
            "thunderkittens_root_exists": True,
        },
        "errors": [],
    }


def _runtime_with_missing_requirement(backend: str) -> dict[str, object]:
    runtime = _healthy_runtime()
    backend_lower = backend.lower()
    if backend_lower == "triton":
        runtime["torch"]["triton_available"] = False
    elif backend_lower in {"cute", "cutlass"}:
        runtime["torch"]["cutlass_available"] = False
    elif backend_lower == "thunderkittens":
        runtime["thunderkittens_root_exists"] = False
        runtime["torch"]["thunderkittens_root_exists"] = False
    elif backend_lower in {"cuda", "hip"}:
        runtime["torch"]["ninja_available"] = False
    return runtime


@pytest.mark.trio
async def test_kernelbench_rollout_resources_fail_fast_on_local_fallback() -> None:
    pool = FakePool(
        is_local=True,
        runtime_descriptions=[
            {
                "runtime_ok": False,
                "error": "import torch failed: ModuleNotFoundError('torch')",
                "torch": {
                    "available": False,
                    "cuda_available": False,
                    "cuda_version": None,
                },
                "errors": ["import torch failed: ModuleNotFoundError('torch')"],
            }
        ],
    )
    resources = KernelBenchRolloutResources(
        evaluator=SandboxPoolKernelEvaluator(pool),  # type: ignore[arg-type]
        backend="cuda",
        runtime_requirements=KernelBenchRuntimeRequirements.for_backend("cuda"),
    )

    with pytest.raises(ValueError, match="KernelBench sandbox requirements not met"):
        await resources.start()


@pytest.mark.trio
async def test_kernelbench_rollout_resources_accept_valid_remote_runtime() -> None:
    pool = FakePool(is_local=False, runtime_descriptions=[_healthy_runtime()])
    resources = KernelBenchRolloutResources(
        evaluator=SandboxPoolKernelEvaluator(pool),  # type: ignore[arg-type]
        backend="cuda",
        runtime_requirements=KernelBenchRuntimeRequirements.for_backend("cuda"),
    )

    await resources.start()

    stats = resources.stats()
    assert "runtime" in stats
    assert len(stats["runtime"]) == 1
    assert stats["runtime"][0]["runtime_ok"] is True


@pytest.mark.parametrize(
    ("backend", "expected_substring"),
    [
        ("triton", "required package: triton"),
        ("cute", "required package: cutlass"),
        ("thunderkittens", "required environment support: thunderkittens_root"),
    ],
)
def test_kernelbench_runtime_requirements_reject_missing_backend_support(
    backend: str,
    expected_substring: str,
) -> None:
    runtime = _runtime_with_missing_requirement(backend)

    requirements = KernelBenchRuntimeRequirements.for_backend(backend)
    errors = requirements.validate_worker(runtime, is_local=False)

    assert any(expected_substring in error for error in errors)


def test_cuda_runtime_requirements_ignore_non_cuda_backend_dependencies() -> None:
    runtime = _healthy_runtime()
    runtime["torch"]["cutlass_available"] = False
    runtime["torch"]["triton_available"] = False
    runtime["thunderkittens_root_exists"] = False

    requirements = KernelBenchRuntimeRequirements.for_backend("cuda")
    errors = requirements.validate_worker(runtime, is_local=False)

    assert all("cutlass" not in error for error in errors)
    assert all("triton" not in error for error in errors)
    assert all("thunderkittens_root" not in error for error in errors)


class FakeManagedResource:
    def __init__(self, config: ModalSandboxResourceConfig, workspace_setup: object | None) -> None:
        del workspace_setup
        self.config = config
        self.working_dir = config.workspace_dir
        self.prepared: list[dict[str, object]] = []
        self.closed = 0

    async def prepare(self, sample_data: dict[str, object] | None = None) -> None:
        self.prepared.append(sample_data or {})

    async def close(self) -> None:
        self.closed += 1

    async def describe_runtime(self) -> dict[str, object]:
        return {"runtime_ok": True}

    def stats(self) -> dict[str, object]:
        return {
            "kind": "fake-managed-resource",
            "prepared_count": len(self.prepared),
            "closed": self.closed,
        }

    def resolve_path(self, current_working_dir: str, path: str) -> str:
        return f"{current_working_dir}/{path}"

    async def read_file(self, path: str) -> bytes:
        return path.encode()

    async def write_file(self, path: str, content: bytes) -> None:
        return None

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


@pytest.mark.trio
async def test_modal_sandbox_manager_reuses_resources_when_keep_warm() -> None:
    manager = ModalSandboxManager(
        config=ModalSandboxResourceConfig(gpu="A100"),
        max_sandboxes=1,
        keep_warm=True,
        resource_factory=FakeManagedResource,
    )

    lease1, resource1 = await manager.acquire({"problem_id": "one"})
    await manager.release(lease1)
    lease2, resource2 = await manager.acquire({"problem_id": "two"})
    await manager.release(lease2)

    assert resource1 is resource2
    assert resource1.prepared == [{"problem_id": "one"}, {"problem_id": "two"}]
    stats = manager.stats()
    assert stats["create_count"] == 1
    assert stats["reuse_count"] == 1
    assert stats["release_count"] == 2
