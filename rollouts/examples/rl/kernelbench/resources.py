"""Explicit KernelBench resource ownership for rollout and scoring stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from rollouts.environments.kernelbench_multi import (
    KernelBenchMultiTurnEnvironment,
    SandboxPoolKernelEvaluator,
)
from rollouts.environments.modal_sandbox_resource import (
    ModalSandboxManager,
    ModalSandboxResourceConfig,
)
from rollouts.environments.resources import KernelEvaluator, SandboxWorkspaceResource
from rollouts.gpu_sandbox import SandboxPool

from .scoring import (
    DEFAULT_REWARD_WEIGHTS,
    KernelBenchRewardWeights,
    KernelBenchSampleScorer,
    KernelJudge,
)


async def _maybe_stop_owner(owner: Any) -> None:
    stop = getattr(owner, "stop", None)
    if callable(stop):
        await stop()


def _parse_version(version: str | None) -> tuple[int, ...] | None:
    if version is None:
        return None
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return None


@dataclass(frozen=True)
class KernelBenchBackendContract:
    backend: str
    required_package_flags: tuple[str, ...] = ()
    required_env_flags: tuple[str, ...] = ()
    notes: str = ""


BACKEND_CONTRACTS: dict[str, KernelBenchBackendContract] = {
    "cuda": KernelBenchBackendContract(
        backend="cuda",
        required_package_flags=("ninja_available",),
        notes="PyTorch CUDA extension flow via torch.utils.cpp_extension.load_inline.",
    ),
    "hip": KernelBenchBackendContract(
        backend="hip",
        required_package_flags=("ninja_available",),
        notes="HIP path also relies on PyTorch extension compilation.",
    ),
    "triton": KernelBenchBackendContract(
        backend="triton",
        required_package_flags=("triton_available",),
        notes="Triton kernels require triton importability and tempfile/module loading.",
    ),
    "tilelang": KernelBenchBackendContract(
        backend="tilelang",
        required_package_flags=("tilelang_available",),
    ),
    "cute": KernelBenchBackendContract(
        backend="cute",
        required_package_flags=("cutlass_available",),
        notes="CuTe/CUTLASS backend relies on Python CUTLASS bindings.",
    ),
    "cutlass": KernelBenchBackendContract(
        backend="cutlass",
        required_package_flags=("cutlass_available",),
        notes="Alias for CuTe/CUTLASS.",
    ),
    "thunderkittens": KernelBenchBackendContract(
        backend="thunderkittens",
        required_env_flags=("thunderkittens_root_exists",),
        notes="ThunderKittens requires a checked-out repo exposed via THUNDERKITTENS_ROOT.",
    ),
}


@dataclass(frozen=True)
class KernelBenchRuntimeRequirements:
    """Explicit runtime contract for KernelBench execution resources."""

    backend: str = "cuda"
    require_torch: bool = True
    require_gpu: bool = True
    require_remote: bool = True
    min_cuda_version: str | None = None

    @classmethod
    def for_backend(
        cls,
        backend: str,
        *,
        require_remote: bool = True,
        min_cuda_version: str | None = None,
    ) -> KernelBenchRuntimeRequirements:
        return cls(
            backend=backend,
            require_remote=require_remote,
            min_cuda_version=min_cuda_version,
        )

    def required_package_flags(self) -> tuple[str, ...]:
        contract = BACKEND_CONTRACTS.get(self.backend.lower())
        return contract.required_package_flags if contract is not None else ()

    def required_env_flags(self) -> tuple[str, ...]:
        contract = BACKEND_CONTRACTS.get(self.backend.lower())
        return contract.required_env_flags if contract is not None else ()

    def validate_worker(self, runtime: dict[str, Any], *, is_local: bool) -> list[str]:
        errors: list[str] = []

        if self.require_remote and is_local:
            errors.append(
                "KernelBench requires explicit remote sandbox configs; local subprocess fallback is disabled."
            )

        if runtime.get("runtime_ok") is False and runtime.get("error"):
            errors.append(str(runtime["error"]))

        torch_info = runtime.get("torch", {})
        if not isinstance(torch_info, dict):
            torch_info = {}

        if self.require_torch and not torch_info.get("available", False):
            errors.append("sandbox runtime does not have importable torch")

        if self.require_gpu and not torch_info.get("cuda_available", False):
            errors.append(f"sandbox runtime does not expose a {self.backend} GPU to torch")

        for package_flag in self.required_package_flags():
            if package_flag in torch_info and not torch_info.get(package_flag, False):
                package_name = package_flag.removesuffix("_available")
                errors.append(f"sandbox runtime is missing required package: {package_name}")

        for env_flag in self.required_env_flags():
            if not runtime.get(env_flag, False):
                env_name = env_flag.removesuffix("_exists")
                errors.append(f"sandbox runtime is missing required environment support: {env_name}")

        if self.min_cuda_version is not None:
            actual_cuda = _parse_version(torch_info.get("cuda_version"))
            minimum_cuda = _parse_version(self.min_cuda_version)
            if actual_cuda is None or minimum_cuda is None or actual_cuda < minimum_cuda:
                errors.append(
                    f"sandbox CUDA version {torch_info.get('cuda_version')} does not satisfy >= {self.min_cuda_version}"
                )

        runtime_errors = runtime.get("errors", [])
        if isinstance(runtime_errors, list):
            errors.extend(str(error) for error in runtime_errors)

        return errors


@dataclass
class KernelBenchRolloutResources:
    """Own rollout-side environment resources for multi-turn KernelBench."""

    evaluator: KernelEvaluator | None = None
    backend: str = "cuda"
    max_turns: int = 8
    runtime_requirements: KernelBenchRuntimeRequirements = field(
        default_factory=KernelBenchRuntimeRequirements
    )
    sandbox_resource_factory: Callable[[dict[str, Any]], SandboxWorkspaceResource] | None = field(
        default=None,
        repr=False,
    )
    sandbox_manager: ModalSandboxManager | None = field(default=None, repr=False)
    _runtime_descriptions: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @classmethod
    def from_sandbox_configs(
        cls,
        sandbox_configs: list[Any] | None = None,
        *,
        backend: str = "cuda",
        max_turns: int = 8,
        runtime_requirements: KernelBenchRuntimeRequirements | None = None,
        allow_local_fallback: bool = False,
        sandbox_resource_factory: Callable[[dict[str, Any]], SandboxWorkspaceResource] | None = None,
    ) -> KernelBenchRolloutResources:
        pool = SandboxPool(sandbox_configs or [])
        evaluator = SandboxPoolKernelEvaluator(pool)
        return cls(
            evaluator=evaluator,
            backend=backend,
            max_turns=max_turns,
            runtime_requirements=runtime_requirements
            or KernelBenchRuntimeRequirements.for_backend(
                backend,
                require_remote=not allow_local_fallback,
            ),
            sandbox_resource_factory=sandbox_resource_factory,
        )

    @classmethod
    def with_modal_sandbox(
        cls,
        sandbox_config: ModalSandboxResourceConfig,
        *,
        backend: str = "cuda",
        max_turns: int = 8,
        workspace_setup: Any | None = None,
        runtime_requirements: KernelBenchRuntimeRequirements | None = None,
        max_sandboxes: int = 1,
        keep_warm: bool = False,
    ) -> KernelBenchRolloutResources:
        sandbox_manager = ModalSandboxManager(
            config=sandbox_config,
            workspace_setup=workspace_setup,
            max_sandboxes=max_sandboxes,
            keep_warm=keep_warm,
        )
        return cls(
            evaluator=None,
            backend=backend,
            max_turns=max_turns,
            runtime_requirements=runtime_requirements
            or KernelBenchRuntimeRequirements.for_backend(backend, require_remote=True),
            sandbox_resource_factory=KernelBenchModalSandboxFactory(manager=sandbox_manager),
            sandbox_manager=sandbox_manager,
        )

    async def start(self) -> None:
        if self.sandbox_manager is not None:
            await self.sandbox_manager.start()
        if self.evaluator is None:
            return
        await self.evaluator.start()
        pool = getattr(self.evaluator, "pool", None)
        if pool is None:
            return
        runtime_descriptions = await pool.describe_runtime()
        self._runtime_descriptions = runtime_descriptions
        validation_errors: list[str] = []
        for idx, runtime in enumerate(runtime_descriptions):
            for error in self.runtime_requirements.validate_worker(
                runtime,
                is_local=pool.is_local,
            ):
                validation_errors.append(f"worker[{idx}]: {error}")
        if validation_errors:
            raise ValueError(
                "KernelBench sandbox requirements not met:\n- " + "\n- ".join(validation_errors)
            )

    async def stop(self) -> None:
        if self.evaluator is not None:
            pool = getattr(self.evaluator, "pool", None)
            if pool is not None:
                await _maybe_stop_owner(pool)
        await _maybe_stop_owner(self.sandbox_manager)

    def stats(self) -> dict[str, Any]:
        stats = self.evaluator.stats() if self.evaluator is not None else {}
        stats.setdefault("kind", "kernelbench_rollout_resources")
        if self._runtime_descriptions:
            stats["runtime"] = self._runtime_descriptions
        if self.sandbox_manager is not None:
            stats["sandbox_manager"] = self.sandbox_manager.stats()
        return stats

    def __call__(self, sample_data: dict[str, Any]) -> KernelBenchMultiTurnEnvironment:
        return KernelBenchMultiTurnEnvironment(
            ref_code=sample_data.get("ref_code", ""),
            backend=self.backend,
            max_turns=self.max_turns,
            evaluator=self.evaluator,
            evaluator_spec=None,
            kernel_workspace=(
                self.sandbox_resource_factory(sample_data)
                if self.sandbox_resource_factory is not None
                else None
            ),
            runtime_requirements=self.runtime_requirements,
        )


@dataclass(frozen=True)
class KernelBenchModalSandboxFactory:
    manager: ModalSandboxManager

    def __call__(self, sample_data: dict[str, Any]) -> SandboxWorkspaceResource:
        return self.manager.make_resource(sample_data)


@dataclass
class KernelBenchScoringResources:
    """Own scorer-side resources for single-turn or judge-based scoring."""

    scorer: KernelBenchSampleScorer
    evaluator_pool: SandboxPool | None = None

    @classmethod
    def from_sandbox_configs(
        cls,
        sandbox_configs: list[Any] | None = None,
        *,
        reward_weights: KernelBenchRewardWeights = DEFAULT_REWARD_WEIGHTS,
        judge: KernelJudge | None = None,
        gate_reward_on_judge: bool = False,
        timeout: float = 120.0,
    ) -> KernelBenchScoringResources:
        pool = SandboxPool(sandbox_configs or [])
        evaluator = SandboxPoolKernelEvaluator(pool)
        scorer = KernelBenchSampleScorer(
            evaluator=evaluator,
            judge=judge,
            reward_weights=reward_weights,
            timeout=timeout,
            gate_reward_on_judge=gate_reward_on_judge,
        )
        return cls(
            scorer=scorer,
            evaluator_pool=pool,
        )

    @classmethod
    def metadata_only(
        cls,
        *,
        reward_weights: KernelBenchRewardWeights = DEFAULT_REWARD_WEIGHTS,
        judge: KernelJudge | None = None,
        gate_reward_on_judge: bool = False,
        timeout: float = 120.0,
    ) -> KernelBenchScoringResources:
        scorer = KernelBenchSampleScorer(
            judge=judge,
            reward_weights=reward_weights,
            timeout=timeout,
            gate_reward_on_judge=gate_reward_on_judge,
        )
        return cls(scorer=scorer)

    async def start(self) -> None:
        # TODO(async-design-decisions.md): If judge inference gets its own pool or
        # endpoint manager, start it here alongside evaluator resources so scorer
        # ownership stays explicit and symmetric.
        evaluator = self.scorer.evaluator
        if evaluator is not None:
            await evaluator.start()

    async def stop(self) -> None:
        if self.evaluator_pool is not None:
            await self.evaluator_pool.stop()

    def stats(self) -> dict[str, Any]:
        return self.scorer.stats()
