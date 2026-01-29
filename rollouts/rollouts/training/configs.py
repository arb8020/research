"""Shared training sub-configs.

Composable building blocks for GRPO, SFT, and future training loops.
Each is a frozen dataclass with sensible defaults. Compose via dataclass fields.
Override via replace().

These live here (not in grpo.py or sft/) so any training loop can import
without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass(frozen=True)
class TrainerConfig:
    """Optimizer and gradient settings."""

    cuda_device_ids: tuple[int, ...] = (0,)
    lr: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_minibatches: int = 8


@dataclass(frozen=True)
class InferenceConfig:
    """Inference server settings (SGLang/vLLM)."""

    backend: str = "sglang"  # "sglang" or "vllm"
    port: int = 30000
    cuda_device_ids: tuple[int, ...] = (0,)
    mem_fraction: float = 0.7


@dataclass(frozen=True)
class RolloutConfig:
    """Rollout generation settings."""

    batch_size: int = 8  # Unique prompts per step
    n_samples_per_prompt: int = 8  # Completions per prompt (the "G" in GRPO)
    max_seq_len: int = 1024
    max_tokens: int = 512
    temperature: float = 0.8
    max_turns: int = 1  # For multi-turn environments
    # TI/TO (Tokens-In/Tokens-Out) - avoids retokenization collapse
    use_tito: bool = False
    # Trajectory strategy for multi-turn rollouts
    # "interleaved": Full conversation as one sequence (efficient, prefix sharing)
    # "branching": Each assistant turn is a separate sample (safer, mirrors deployment)
    trajectory_strategy: str = "interleaved"


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint, logging, and weight sync settings."""

    num_steps: int = 100
    log_every: int = 1
    checkpoint_every: int = 20  # Save to disk (for recovery/resuming)
    sync_weights_every: int = 1  # Sync to inference engine (for on-policy vs off-policy)


@dataclass(frozen=True)
class OutputConfig:
    """Training output directory and experiment naming."""

    output_dir: str = "results"
    experiment_name: str = "experiment"
