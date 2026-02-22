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

# =============================================================================
# Hardware & Distributed Configs (provisioning + parallelism)
# =============================================================================


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
        # Single A100 on Modal
        HardwareConfig(gpu_type="A100", gpu_count=1, provider="modal")

        # 2x H100 on RunPod
        HardwareConfig(gpu_type="H100", gpu_count=2, provider="runpod")

        # Local execution (no provisioning)
        HardwareConfig(provider="local")
    """

    gpu_type: str = "A100"
    gpu_count: int = 1
    provider: Literal["modal", "runpod", "lambdalabs", "vast", "local"] = "runpod"

    # Auto-derived from gpu_type if None (for known GPUs)
    gpu_memory_gb: int | None = None
    compute_capability: str | None = None

    def __post_init__(self) -> None:
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

        # 4 GPUs: TP=2 for inference, DP=2 for training
        DistributedConfig(
            inference_gpus=(0, 1),
            inference_tp=2,
            trainer_gpus=(2, 3),
            trainer_fsdp=True,
        )
    """

    # Inference parallelism (for SGLang/vLLM)
    inference_gpus: tuple[int, ...] = (0,)
    inference_tp: int = 1  # Tensor parallel size (must divide len(inference_gpus))

    # Training parallelism
    trainer_gpus: tuple[int, ...] = (0,)
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
        return tuple(sorted(set(self.inference_gpus) | set(self.trainer_gpus)))


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


# =============================================================================
# Training Sub-Configs (algorithm-specific settings)
# =============================================================================


@dataclass(frozen=True)
class TrainerConfig:
    """Optimizer and gradient settings.

    Note: cuda_device_ids is deprecated in favor of DistributedConfig.trainer_gpus.
    When DistributedConfig is provided, it takes precedence.
    """

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


@dataclass(frozen=True)
class InferenceConfig:
    """Inference server settings (SGLang/vLLM).

    Supports multiple inference engines for higher throughput (PipelineRL-style).
    Each GPU in cuda_device_ids gets its own inference server on a separate port.

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

    backend: str = "sglang"  # "sglang", "vllm", or "engine_v2"
    port: int = 30000  # Base port (engines use port, port+1, ...)
    cuda_device_ids: tuple[int, ...] = (0,)
    mem_fraction: float = 0.7
    tensor_parallel_size: int = 1  # GPUs per engine (1 = each GPU is its own engine)

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
        if len(self.cuda_device_ids) % self.tensor_parallel_size != 0:
            raise ValueError(
                f"tensor_parallel_size={self.tensor_parallel_size} must divide "
                f"len(cuda_device_ids)={len(self.cuda_device_ids)}"
            )


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
    sync_weights_every: int = 1  # Sync to inference engine (for on-policy vs off-policy)
    # Weight sync mode: "disk" (save to /dev/shm, reload) or "nccl" (GPU-to-GPU broadcast)
    # "nccl" enables PipelineRL-style in-flight updates (faster, non-blocking)
    weight_sync_mode: str = "disk"
    # NCCL master port for weight sync (only used if weight_sync_mode="nccl")
    nccl_master_port: int = 29500
    # Pipeline mode:
    #   "sync" - generate batch, train, sync weights, repeat (stop-and-go, default)
    #   "async" - background sampling, blocking weight sync (training waits for sync)
    #   "true_pipeline" - PipelineRL-style: both sampling AND weight sync non-blocking
    #                     (inference never stops, accepts slightly stale weights)
    pipeline_mode: str = "sync"
    # Maximum weight version lag for async pipeline (samples older than this are discarded)
    # Only used if pipeline_mode="async". Set to 0 for strict on-policy.
    max_lag: int = 2
    # Sample queue size for async pipeline
    pipeline_queue_size: int = 1024


@dataclass(frozen=True)
class OutputConfig:
    """Training output directory and experiment naming."""

    output_dir: str = "results"
    experiment_name: str = "experiment"
