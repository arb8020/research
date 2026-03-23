"""Shared training sub-configs.

Composable building blocks for GRPO, SFT, and future training loops.
Each is a frozen dataclass with sensible defaults. Compose via dataclass fields.
Override via replace().

These live here (not in grpo.py or sft/) so any training loop can import
without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..image_spec import ImageSpec, RuntimeOverlay, default_cuda_image

# =============================================================================
# Hardware & Distributed Configs (provisioning + parallelism)
# =============================================================================


@dataclass(frozen=True)
class DepsConfig:
    """Environment dependencies for remote execution.

    Explicit specification of what goes into the container/environment.
    Required for Modal, optional for SSH providers.

    `image` describes the desired base image. `runtime_overlay` describes
    run-time additions that should be applied on top. The legacy package fields
    are still supported and are folded into the resolved image so existing
    configs keep working while the runners migrate to the new contract.

    Example:
        DepsConfig(
            pip_packages=(
                "torch>=2.4",
                "sglang[all]",
                "transformers>=5.0",
            ),
            pip_index_url="https://download.pytorch.org/whl/cu124",
        )
    """

    # TODO(boundary): `DepsConfig` + `HardwareConfig` currently want to be one
    # explicit runtime-contract product type. They still blur base runtime
    # contract, project-local overlays, cache/volume knobs, and provider-facing
    # execution settings.

    python_version: str = "3.12"
    base_image: str = "debian:bookworm-slim"
    system_packages: tuple[str, ...] = (
        "bash",
        "curl",
        "git",
        "build-essential",
        "libnuma1",
        "tmux",
    )
    pip_packages: tuple[str, ...] = ()
    pip_index_url: str | None = None
    pip_extra_index_url: str | None = None
    pip_prerelease: bool = False
    bootstrap_commands: tuple[str, ...] = ()
    image: ImageSpec | None = None
    runtime_overlay: RuntimeOverlay = field(default_factory=RuntimeOverlay)

    def __post_init__(self) -> None:
        assert self.python_version, "python_version cannot be empty"
        if self.image is None:
            assert self.base_image, "base_image cannot be empty"

    def resolved_image(self, gpu_type: str) -> ImageSpec:
        """Resolve the explicit image contract used by Modal and SSH backends."""
        if self.image is None:
            base = ImageSpec.from_registry(
                default_cuda_image(gpu_type, self.pip_index_url),
                python_version=self.python_version,
            )
        else:
            base = self.image

        return base.extended(
            system_packages=self.system_packages,
            pip_packages=self.pip_packages,
            pip_index_url=self.pip_index_url,
            pip_extra_index_url=self.pip_extra_index_url,
            pip_prerelease=self.pip_prerelease,
            build_commands=self.bootstrap_commands,
        )

    def resolved_runtime_overlay(self) -> RuntimeOverlay:
        return self.runtime_overlay

    def merged_with(self, other: DepsConfig) -> DepsConfig:
        """Merge two service-scoped dependency contracts into one shared env contract.

        This is only valid when the non-additive parts of the contract agree.
        If the two services want different base images, Python versions, or
        package index policy, shared_env is not an honest realization and we
        fail loudly.
        """
        if self.python_version != other.python_version:
            raise ValueError(
                f"shared_env deps mismatch: python_version {self.python_version!r} != {other.python_version!r}"
            )
        if self.base_image != other.base_image:
            raise ValueError(
                f"shared_env deps mismatch: base_image {self.base_image!r} != {other.base_image!r}"
            )
        if self.pip_index_url != other.pip_index_url:
            raise ValueError(
                f"shared_env deps mismatch: pip_index_url {self.pip_index_url!r} != {other.pip_index_url!r}"
            )
        if self.pip_extra_index_url != other.pip_extra_index_url:
            raise ValueError(
                "shared_env deps mismatch: pip_extra_index_url "
                f"{self.pip_extra_index_url!r} != {other.pip_extra_index_url!r}"
            )
        if self.pip_prerelease != other.pip_prerelease:
            raise ValueError(
                f"shared_env deps mismatch: pip_prerelease {self.pip_prerelease!r} != {other.pip_prerelease!r}"
            )
        if self.image != other.image:
            raise ValueError("shared_env deps mismatch: image specs differ")

        return DepsConfig(
            python_version=self.python_version,
            base_image=self.base_image,
            system_packages=tuple(dict.fromkeys(self.system_packages + other.system_packages)),
            pip_packages=tuple(dict.fromkeys(self.pip_packages + other.pip_packages)),
            pip_index_url=self.pip_index_url,
            pip_extra_index_url=self.pip_extra_index_url,
            pip_prerelease=self.pip_prerelease,
            bootstrap_commands=tuple(
                dict.fromkeys(self.bootstrap_commands + other.bootstrap_commands)
            ),
            image=self.image,
            runtime_overlay=self.runtime_overlay.extended(
                system_packages=other.runtime_overlay.system_packages,
                pip_packages=other.runtime_overlay.pip_packages,
                pip_index_url=other.runtime_overlay.pip_index_url,
                pip_extra_index_url=other.runtime_overlay.pip_extra_index_url,
                pip_prerelease=other.runtime_overlay.pip_prerelease,
                commands=other.runtime_overlay.commands,
                env=other.runtime_overlay.env,
                features=other.runtime_overlay.features,
                installed_groups=other.runtime_overlay.installed_groups,
            ),
        )


def _image_spec_from_data(data: ImageSpec | dict[str, Any] | None) -> ImageSpec | None:
    if data is None or isinstance(data, ImageSpec):
        return data
    assert isinstance(data, dict), f"image must be ImageSpec|dict|None, got {type(data)}"
    return ImageSpec(**data)


def _runtime_overlay_from_data(
    data: RuntimeOverlay | dict[str, Any] | None,
) -> RuntimeOverlay:
    if data is None:
        return RuntimeOverlay()
    if isinstance(data, RuntimeOverlay):
        return data
    assert isinstance(data, dict), (
        f"runtime_overlay must be RuntimeOverlay|dict|None, got {type(data)}"
    )
    return RuntimeOverlay(**data)


def deps_config_from_data(data: DepsConfig | dict[str, Any] | None) -> DepsConfig | None:
    if data is None or isinstance(data, DepsConfig):
        return data
    assert isinstance(data, dict), f"deps must be DepsConfig|dict|None, got {type(data)}"
    payload = dict(data)
    payload["image"] = _image_spec_from_data(payload.get("image"))
    payload["runtime_overlay"] = _runtime_overlay_from_data(payload.get("runtime_overlay"))
    return DepsConfig(**payload)


# Known GPU specs: (memory_gb, compute_capability)
# Used for validation and auto-derivation
GPU_SPECS: dict[str, tuple[int, str]] = {
    "H100": (80, "9.0"),
    "H200": (141, "9.0"),
    "A100": (80, "8.0"),
    "A100-40GB": (40, "8.0"),
    "A10G": (24, "8.6"),
    "L40S": (48, "8.9"),
    "L40": (48, "8.9"),
    "L4": (24, "8.9"),
    "T4": (16, "7.5"),
    "B200": (192, "10.0"),
}


@dataclass(frozen=True)
class HardwareConfig:
    """What hardware to provision.

    This describes the physical resources to acquire, not how to use them.
    The runner uses this to provision via Modal, RunPod, etc.

    Example:
        # Single A100 on Modal (deps required)
        HardwareConfig(
            gpu_type="A100",
            gpu_count=1,
            provider="modal",
            deps=DepsConfig(
                pip_packages=("torch>=2.4", "sglang[all]"),
                pip_index_url="https://download.pytorch.org/whl/cu124",
            ),
        )

        # 2x H100 on RunPod (deps optional, uses hardcoded bootstrap)
        HardwareConfig(gpu_type="H100", gpu_count=2, provider="runpod")

        # Local execution (no provisioning)
        HardwareConfig(provider="local")
    """

    # TODO(boundary): shrink this toward provision-time runtime contract only.
    # Mutable workspace-scoped realization should move to the execution layer
    # instead of continuing to accumulate here.

    gpu_type: str = "A100"
    gpu_count: int = 1
    provider: Literal["modal", "runpod", "lambdalabs", "vast", "local"] = "runpod"

    # Shared runtime deps for the current single-env runners.
    # Current launchers still realize one environment for the whole workload,
    # so remote execution owns deps here rather than on trainer/inference.
    deps: DepsConfig | None = None

    # Remote provisioning/runtime settings
    container_disk_gb: int = 100
    hf_cache_dir: str = "/workspace/.cache/huggingface"
    persistent_volume_id: str | None = None
    persistent_volume_mount_path: str = "/workspace"
    persistent_volume_location: str | None = None

    # Auto-derived from gpu_type if None (for known GPUs)
    gpu_memory_gb: int | None = None
    compute_capability: str | None = None

    # Whether to use torchrun for multi-GPU (set False for torchtitan which handles FSDP internally)
    use_torchrun: bool = True

    def __post_init__(self) -> None:
        # Validate: Modal requires deps
        if self.provider == "modal" and self.deps is None:
            raise ValueError(
                "HardwareConfig with provider='modal' requires deps. "
                "Example: deps=DepsConfig(pip_packages=('torch>=2.4', 'sglang[all]'))"
            )
        if self.provider in {"runpod", "lambdalabs", "vast"} and self.deps is None:
            raise ValueError(
                "HardwareConfig for SSH providers requires explicit deps. "
                "Declare a DepsConfig in hardware.deps."
            )

        assert self.container_disk_gb > 0, "container_disk_gb must be positive"
        assert self.hf_cache_dir, "hf_cache_dir cannot be empty"
        if self.persistent_volume_id is not None:
            assert self.persistent_volume_mount_path.startswith("/"), (
                "persistent_volume_mount_path must be an absolute path"
            )

        # Auto-derive GPU specs for known types
        if self.gpu_type in GPU_SPECS and (
            self.gpu_memory_gb is None or self.compute_capability is None
        ):
            mem, cc = GPU_SPECS[self.gpu_type]
            # Use object.__setattr__ for frozen dataclass
            if self.gpu_memory_gb is None:
                object.__setattr__(self, "gpu_memory_gb", mem)
            if self.compute_capability is None:
                object.__setattr__(self, "compute_capability", cc)

    @property
    def total_memory_gb(self) -> int | None:
        """Total VRAM across all GPUs."""
        if self.gpu_memory_gb is None:
            return None
        return self.gpu_memory_gb * self.gpu_count


@dataclass(frozen=True)
class DistributedConfig:
    """How to use the provisioned GPUs (parallelism strategy).

    Separates inference parallelism from training parallelism.
    GPU assignment determines which GPUs run inference vs training.

    Example:
        # Single GPU, shared between inference and training
        DistributedConfig()  # defaults: inference_gpus=(0,), trainer_gpus=(0,)

        # 2 GPUs: one for inference, one for training
        DistributedConfig(
            inference_gpus=(0,),
            trainer_gpus=(1,),
        )

        # 4 GPUs: TP=2 for inference, DP=1 for training, 1 workspace GPU
        DistributedConfig(
            inference_gpus=(0, 1),
            inference_tp=2,
            trainer_gpus=(2,),
            workspace_gpus=(3,),
            trainer_fsdp=True,
        )
    """

    # Inference parallelism (for SGLang/vLLM)
    inference_gpus: tuple[int, ...] = (0,)
    inference_tp: int = 1  # Tensor parallel size (must divide len(inference_gpus))

    # Training parallelism
    trainer_gpus: tuple[int, ...] = (0,)
    workspace_gpus: tuple[int, ...] = ()
    trainer_fsdp: bool = True  # Use FSDP (ZeRO-3 style sharding)
    trainer_dp: int | None = None  # Data parallel size (derived from trainer_gpus if None)

    # Multi-node settings (for future use)
    # If master_addr is None, assumes single-node (torchrun handles setup)
    master_addr: str | None = None
    master_port: int = 29500

    def __post_init__(self) -> None:
        # Validate inference TP
        if self.inference_tp > 1:
            assert len(self.inference_gpus) % self.inference_tp == 0, (
                f"inference_tp={self.inference_tp} must divide "
                f"len(inference_gpus)={len(self.inference_gpus)}"
            )

        # Derive trainer_dp if not set
        if self.trainer_dp is None:
            object.__setattr__(self, "trainer_dp", len(self.trainer_gpus))

    @property
    def is_shared_gpu(self) -> bool:
        """True if inference and training share any GPUs."""
        return bool(set(self.inference_gpus) & set(self.trainer_gpus))

    @property
    def all_gpus(self) -> tuple[int, ...]:
        """All unique GPUs used (for CUDA_VISIBLE_DEVICES)."""
        return tuple(
            sorted(set(self.inference_gpus) | set(self.trainer_gpus) | set(self.workspace_gpus))
        )


@dataclass(frozen=True)
class ModelConfig:
    """Model identity and format."""

    name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    # LoRA (for efficient test-time training)
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    # Checkpoint loading (for SFT → RL pipeline)
    checkpoint_path: str | None = None
    # Expert pruning recipe (path to JSON with experts_to_keep mapping)
    # If set, model is downloaded, pruned, and cached before training
    # Recipe format: {"base_model": "...", "experts_to_keep": {"layer_idx": [expert_indices]}}
    pruning_recipe: str | None = None


# =============================================================================
# Training Sub-Configs (algorithm-specific settings)
# =============================================================================


@dataclass(frozen=True)
class MegatronOverrides:
    """Temporary Megatron-native escape hatch.

    TODO(denotation): Promote stable fields here into honest training-side
    batch/output realization semantics, and leave only irreducibly
    Megatron-native residue behind.
    """

    sequence_parallel: bool | None = None
    allocator_expandable_segments: bool = False
    output_materialization: Literal["default", "chunked_logits"] = "default"
    lm_head_token_chunk_size: int | None = None
    max_tokens_per_microbatch: int | None = None

    def __post_init__(self) -> None:
        if self.lm_head_token_chunk_size is not None:
            assert self.lm_head_token_chunk_size > 0, "lm_head_token_chunk_size must be positive"
        if self.max_tokens_per_microbatch is not None:
            assert self.max_tokens_per_microbatch > 0, "max_tokens_per_microbatch must be positive"


def megatron_overrides_from_data(
    data: MegatronOverrides | dict[str, Any] | None,
) -> MegatronOverrides | None:
    if data is None or isinstance(data, MegatronOverrides):
        return data
    assert isinstance(data, dict), (
        f"megatron_overrides must be MegatronOverrides|dict|None, got {type(data)}"
    )
    return MegatronOverrides(**data)


@dataclass(frozen=True)
class TrainerConfig:
    """Optimizer and gradient settings.

    Note: cuda_device_ids is deprecated in favor of DistributedConfig.trainer_gpus.
    When DistributedConfig is provided, it takes precedence.
    """

    # Training backend implementation.
    # `nmoe` is a reserved backend name that currently fails loudly until a
    # native runtime adapter lands.
    backend: Literal["pytorch", "fsdp", "fsdp2", "nmoe", "megatron", "torchtitan"] = "pytorch"
    # Service-scoped runtime deps for the trainer process.
    # Current launchers do not realize per-service environments yet, so configs
    # that set this should be rejected at the runner boundary rather than
    # silently collapsing back to one shared env.
    deps: DepsConfig | None = None

    # DEPRECATED: Use DistributedConfig.trainer_gpus instead
    cuda_device_ids: tuple[int, ...] = (0,)
    lr: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_minibatches: int = 8
    # Loss function: "vanilla" (simple PG), "clipped" (PPO-style), "masked" (Prime-RL ratio masking)
    loss_type: str = "vanilla"
    # Importance ratio masking bounds (for loss_type="masked")
    mask_ratio_low: float = 0.125
    mask_ratio_high: float = 8.0
    # VRAM preflight check settings
    skip_vram_check: bool = False
    vram_safety_margin: float = 0.05  # fraction of GPU VRAM reserved for allocator fragmentation

    # On-Policy Distillation (OPD) settings
    # Advantage estimator: "grpo" (standard group-relative), "opd" (on-policy distillation)
    advantage_estimator: str = "grpo"
    # Teacher model for OPD (if None, uses same model as student - not recommended)
    teacher_model: str | None = None
    # Teacher inference server port (separate from student inference)
    teacher_port: int = 30100

    # Megatron-specific settings (only used when backend="megatron")
    # Parallelism dimensions
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
    # Sequence length for Megatron
    seq_length: int = 4096
    # Micro batch size per GPU (if None, computed from num_minibatches)
    micro_batch_size: int | None = None
    # Temporary backend-native escape hatch until batch/output realization is
    # denoted more honestly above the Megatron lowering boundary.
    megatron_overrides: MegatronOverrides | None = None
    # Memory optimizations (from SLIME)
    optimizer_cpu_offload: bool = False  # Offload Adam states to CPU
    activation_checkpointing: bool = True  # Gradient checkpointing
    recompute_granularity: str = "selective"  # "full", "selective", or "none"
    recompute_method: str = "uniform"  # "uniform" or "block"
    recompute_num_layers: int = 1  # Layers per recompute block
    # Whether GRPO preflight should force a full Megatron runtime->HF export
    # before inference startup. Keep this on for stricter backend bringup, but
    # witnesses can disable it when the later weight-sync witness is the real
    # contract they care about and the full export is too expensive upfront.
    validate_inference_export_in_preflight: bool = True

    # Backend-neutral realization intent.
    # These strings describe denotational layout/collective intent; backends
    # validate and lower them into runtime-native configuration.
    realization_local_layouts: tuple[str, ...] = ()
    realization_collective_transitions: tuple[str, ...] = ()
    realization_packed_sequences: bool = True

    # TorchTitan-specific settings (only used when backend="torchtitan")
    # Model name registered with torchtitan (e.g., "glm", "llama3", "qwen3")
    torchtitan_model: str = "glm"
    # Model size variant (e.g., "4.7-flash", "5", "8B")
    torchtitan_model_size: str = "4.7-flash"
    # TorchTitan parallelism (TP, CP, PP handled by torchtitan)
    torchtitan_tp: int = 1
    torchtitan_cp: int = 1
    torchtitan_pp: int = 1

    def __post_init__(self) -> None:
        assert self.backend in ("pytorch", "fsdp", "fsdp2", "nmoe", "megatron", "torchtitan"), (
            f"Unknown trainer backend: {self.backend!r}. "
            "Use 'pytorch', 'fsdp', 'fsdp2', 'nmoe', 'megatron', or 'torchtitan'."
        )


@dataclass(frozen=True)
class InferenceConfig:
    """Inference server settings (SGLang/vLLM).

    Supports multiple inference engines for higher throughput (PipelineRL-style).
    Each GPU in cuda_device_ids gets its own inference server on a separate port.

    Architectural note:
    Inference backends now go through `inference_runtime_factory`, so engine
    construction and first-cut pipeline validation have one home. That runtime
    boundary is still narrower than the training-side one: launch sequencing,
    lifecycle ownership, and async publication lowering still leak into GRPO.
    Keep this config honest about concrete engine settings; do not treat it as
    a complete cross-runtime capability contract yet.

    Example:
        # Single inference engine on GPU 0
        InferenceConfig(cuda_device_ids=(0,), port=30000)

        # Two inference engines on GPUs 0 and 1 (ports 30000, 30001)
        InferenceConfig(cuda_device_ids=(0, 1), port=30000)

        # TP=2: one engine using 2 GPUs
        InferenceConfig(cuda_device_ids=(0, 1), port=30000, tensor_parallel_size=2)

    Note: cuda_device_ids is deprecated in favor of DistributedConfig.inference_gpus.
    When DistributedConfig is provided, it takes precedence.
    """

    backend: str = "sglang"  # Compatibility input; factory lowers this to a realization.
    # Preferred explicit runtime selection. Examples:
    # - "slime-sglang"
    # - "qed-vllm"
    # Keep patch/local-server choices here instead of deriving them from sync flags.
    realization: str | None = None
    # Service-scoped runtime deps for the inference process.
    # Current launchers do not realize per-service environments yet.
    deps: DepsConfig | None = None
    port: int = 30000  # Base port (engines use port, port+1, ...)
    cuda_device_ids: tuple[int, ...] = (0,)
    mem_fraction: float = 0.7
    disable_cuda_graph: bool = False
    max_total_tokens: int | None = None
    max_prefill_tokens: int | None = None
    max_running_requests: int | None = None
    chunked_prefill_size: int | None = None
    tensor_parallel_size: int = 1  # GPUs per engine (1 = each GPU is its own engine)
    expert_parallel_size: int = 1  # For MoE models (SGLang --ep-size)
    startup_timeout: float = (
        300.0  # Max seconds to wait for server to start (model download can be slow)
    )

    @property
    def num_engines(self) -> int:
        """Number of inference engines to launch."""
        return len(self.cuda_device_ids) // self.tensor_parallel_size

    @property
    def ports(self) -> tuple[int, ...]:
        """Port for each inference engine."""
        return tuple(self.port + i for i in range(self.num_engines))

    @property
    def gpu_assignments(self) -> list[tuple[int, ...]]:
        """GPU assignment for each engine (supports TP)."""
        tp = self.tensor_parallel_size
        gpus = self.cuda_device_ids
        return [tuple(gpus[i * tp : (i + 1) * tp]) for i in range(self.num_engines)]

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.backend not in ("sglang", "vllm", "engine_v2"):
            raise ValueError(
                f"Unknown inference backend: {self.backend!r}. "
                "Use 'sglang', 'vllm', or 'engine_v2'."
            )
        if len(self.cuda_device_ids) % self.tensor_parallel_size != 0:
            raise ValueError(
                f"tensor_parallel_size={self.tensor_parallel_size} must divide "
                f"len(cuda_device_ids)={len(self.cuda_device_ids)}"
            )


InferenceRole = Literal["actor", "judge", "teacher", "reference"]


@dataclass(frozen=True)
class InferenceWorkerConfig:
    """Named inference worker realized on a slice of the provisioned allocation.

    This is the missing layer between:
    - workload-level hardware allocation ("give me 4xH100")
    - role bindings in eval/RL ("actor", "judge", "teacher")

    `InferenceConfig` stays worker-local and concrete: engine/runtime settings,
    ports, GPU ids, request limits. This wrapper adds the semantic identity that
    higher-level workloads want to bind to.
    """

    worker_id: str
    model: str
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    provider: Literal["sglang", "vllm", "openai", "anthropic", "google"] | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    api_format: str | None = None

    def __post_init__(self) -> None:
        assert self.worker_id, "worker_id cannot be empty"
        assert self.model, "model cannot be empty"

    @property
    def resolved_provider(self) -> str:
        if self.provider is not None:
            return self.provider
        if self.inference.backend in {"sglang", "vllm"}:
            return self.inference.backend
        raise ValueError(
            f"InferenceWorkerConfig {self.worker_id!r} needs an explicit provider "
            f"for backend {self.inference.backend!r}."
        )


@dataclass(frozen=True)
class TrainingWorkerConfig:
    """Named training worker realized on a slice of the provisioned allocation."""

    worker_id: str
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def __post_init__(self) -> None:
        assert self.worker_id, "worker_id cannot be empty"


@dataclass(frozen=True)
class InferenceRoleBinding:
    """Bind a semantic workload role to a named inference worker."""

    role: InferenceRole
    worker_id: str


@dataclass(frozen=True)
class WorkerTopologyConfig:
    """Provisioned allocation plus named workers/services realized on top of it.

    TODO(boundary): this is the direction evals and RL both want:
    - allocate hardware once
    - carve it into named inference/training workers
    - bind semantic roles like actor/judge/teacher to those workers

    Current evals still lower this back into `endpoint + server + hardware`.
    Current RL configs still use `trainer` + `inference` directly inside
    `GRPOConfig`. Keep this topology additive until both surfaces are ready to
    consume it natively.
    """

    hardware: HardwareConfig
    inference_workers: tuple[InferenceWorkerConfig, ...] = ()
    training_workers: tuple[TrainingWorkerConfig, ...] = ()
    role_bindings: tuple[InferenceRoleBinding, ...] = ()

    def __post_init__(self) -> None:
        worker_ids = [w.worker_id for w in self.inference_workers] + [
            w.worker_id for w in self.training_workers
        ]
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("WorkerTopologyConfig worker ids must be unique")

        inference_ids = {w.worker_id for w in self.inference_workers}
        for binding in self.role_bindings:
            if binding.worker_id not in inference_ids:
                raise ValueError(
                    f"Role binding {binding.role!r} references unknown inference worker "
                    f"{binding.worker_id!r}"
                )

        allowed_gpu_ids = set(range(self.hardware.gpu_count))
        for worker in self.inference_workers:
            for gpu_id in worker.inference.cuda_device_ids:
                if gpu_id not in allowed_gpu_ids:
                    raise ValueError(
                        f"Inference worker {worker.worker_id!r} references gpu_id={gpu_id}, "
                        f"but hardware.gpu_count={self.hardware.gpu_count}"
                    )
        for worker in self.training_workers:
            for gpu_id in worker.trainer.cuda_device_ids:
                if gpu_id not in allowed_gpu_ids:
                    raise ValueError(
                        f"Training worker {worker.worker_id!r} references gpu_id={gpu_id}, "
                        f"but hardware.gpu_count={self.hardware.gpu_count}"
                    )

    def get_inference_worker(self, worker_id: str) -> InferenceWorkerConfig:
        for worker in self.inference_workers:
            if worker.worker_id == worker_id:
                return worker
        raise KeyError(f"Unknown inference worker: {worker_id!r}")

    def get_worker_for_role(self, role: InferenceRole) -> InferenceWorkerConfig:
        for binding in self.role_bindings:
            if binding.role == role:
                return self.get_inference_worker(binding.worker_id)
        raise KeyError(f"Unknown inference role: {role!r}")


@dataclass(frozen=True)
class RolloutConfig:
    """Rollout generation settings."""

    batch_size: int = 8  # Unique prompts per step
    n_samples_per_prompt: int = 8  # Completions per prompt (the "G" in GRPO)
    max_seq_len: int = 1024
    max_tokens: int = 512
    temperature: float = 0.8
    max_turns: int = 1  # For multi-turn environments
    # Trajectory strategy for multi-turn rollouts
    # "interleaved": Full conversation as one sequence (efficient, prefix sharing)
    # "branching": Each assistant turn is a separate sample (safer, mirrors deployment)
    trajectory_strategy: str = "interleaved"
    # Extra params merged into inference requests (e.g. SGLang/vLLM sampling config).
    # For Qwen3 no-think: {"chat_template_kwargs": {"enable_thinking": False}}
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint, logging, and weight sync settings."""

    num_steps: int = 100
    log_every: int = 1
    checkpoint_every: int = 20  # Save to disk (for recovery/resuming)
    # Whether checkpoints should include optimizer/scheduler state.
    # Disable this when a backend's optimizer serializer is known-broken and
    # the run only needs model-weight recovery.
    save_optimizer_state: bool = True
    sync_weights_every: int = 1  # Sync to inference engine (for on-policy vs off-policy)
    # Weight sync mode: "disk" (save to /dev/shm, reload) or "nccl" (direct tensor broadcast).
    # "nccl" chooses the transport only. Whether updates are blocking or inflight
    # still depends on pipeline_mode and the concrete inference realization.
    weight_sync_mode: str = "disk"
    # Concrete inference-side sync adapter. This is the realized engine/backend
    # contract, not the high-level semantic policy.
    inference_sync_realization: str | None = None
    # NCCL master port for weight sync (only used if weight_sync_mode="nccl")
    nccl_master_port: int = 29500
    # Pipeline mode:
    #   "sync" - generate batch, train, sync weights, repeat (stop-and-go, default)
    #   "async" - background sampling, blocking weight sync (training waits for sync)
    #   "true_pipeline" - experimental trainer/sampler overlap with versioned stale-sample
    #                     filtering. Current direct-receive realizations still block at the
    #                     inference weight-application boundary, so new admissions should be
    #                     treated as paused while sync is in progress.
    #
    # Architectural note:
    # This flag is still interpreted mostly by GRPO orchestration code. It is
    # not yet an honestly lowered cross-backend capability: training and
    # inference backends are not both validating/realizing it through a shared
    # runtime plan today.
    pipeline_mode: str = "sync"
    # Maximum weight version lag for async pipeline (samples older than this are discarded)
    # Only used if pipeline_mode="async". Set to 0 for strict on-policy.
    max_lag: int = 2
    # Sample queue size for async pipeline.
    # Set to 0 for stream-style production semantics without producer-side blocking.
    pipeline_queue_size: int = 0


@dataclass(frozen=True)
class OutputConfig:
    """Training output directory and experiment naming."""

    output_dir: str = "results"
    experiment_name: str = "experiment"
