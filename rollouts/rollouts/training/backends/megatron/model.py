"""Megatron model and optimizer factory.

Creates Megatron models from HuggingFace checkpoints using mbridge/AutoBridge.
This is the key integration point - we load HF models but train with Megatron.

Two loading modes:
1. AutoBridge (standard): For supported architectures (Llama, Qwen, etc.)
   - Direct HF -> Megatron conversion
   - Uses mbridge package from https://github.com/ISEEKYAN/mbridge

2. Custom bridges (GLM models): For unsupported architectures
   - Register custom bridges via @register_model decorator
   - See mbridge/ subdirectory for GLM4, GLM4MoE bridges

Based on SLIME's model.py but simplified to just the factory function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MegatronModelConfig:
    """Configuration for Megatron model creation.

    Combines model identity with training hyperparameters.
    """

    # Model
    model_name: str  # HuggingFace model name or path
    trust_remote_code: bool = True

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Optimizer
    lr: float = 1e-6
    min_lr: float = 1e-7
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    clip_grad: float = 1.0

    # Batch size
    micro_batch_size: int = 1
    global_batch_size: int = 8

    # Sequence
    seq_length: int = 4096

    # MoE (for GLM-4.7-Flash and similar)
    num_experts: int | None = None
    moe_router_topk: int = 4
    moe_ffn_hidden_size: int | None = None

    # MLA (Multi-Latent Attention) for GLM models
    multi_latent_attention: bool = False
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_head_dim: int | None = None
    v_head_dim: int | None = None

    # Architecture overrides (for manual model construction)
    # These are used when AutoBridge doesn't support the model
    architecture_args: dict[str, Any] = field(default_factory=dict)


def setup_megatron_model(
    config: MegatronModelConfig,
    checkpoint_path: Path | None = None,
) -> tuple[list[Any], Any, Any]:
    """Create Megatron model with optimizer from HuggingFace checkpoint.

    Uses mbridge/AutoBridge to convert HF models to Megatron format.
    This handles the weight mapping automatically.

    Supported architectures:
    - Standard (via mbridge): Llama, Qwen2, Qwen3, Qwen3Moe, DeepseekV3
    - Custom bridges: GLM-4, GLM-4.7-Flash (via our mbridge/ module)

    Args:
        config: Model and training configuration
        checkpoint_path: Optional path to load checkpoint from

    Returns:
        Tuple of (model_chunks, optimizer, scheduler)
        - model_chunks: List of model chunks (for pipeline parallelism)
        - optimizer: MegatronOptimizer instance
        - scheduler: Learning rate scheduler

    Raises:
        ImportError: If mbridge is not installed
        ValueError: If model architecture is not supported

    Example:
        >>> config = MegatronModelConfig(
        ...     model_name="THUDM/GLM-4.7-Flash",  # Now supported via custom bridge
        ...     lr=1e-6,
        ... )
        >>> model, optimizer, scheduler = setup_megatron_model(config)
    """
    # Register our custom bridges for GLM models before importing AutoBridge
    try:
        from rollouts.training.backends.megatron import mbridge as _  # noqa: F401
    except ImportError:
        logger.debug("Custom mbridge bridges not available")

    try:
        from mbridge import AutoBridge
    except ImportError as e:
        raise ImportError(
            "mbridge is required for Megatron backend. "
            "Install via: pip install git+https://github.com/ISEEKYAN/mbridge.git"
        ) from e

    try:
        from megatron.core.enums import ModelType
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
        from megatron.training.training import get_model
    except ImportError as e:
        raise ImportError(
            "megatron-core is required for Megatron backend. "
            "See: https://github.com/NVIDIA/Megatron-LM"
        ) from e

    logger.info("Loading model via AutoBridge: %s", config.model_name)

    # Create model provider from HuggingFace model
    # AutoBridge auto-detects the model type and uses registered bridges
    try:
        bridge = AutoBridge.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as e:
        raise ValueError(
            f"AutoBridge failed to load model '{config.model_name}'. "
            f"This architecture may not be supported. "
            f"Supported: Llama, Qwen, DeepseekV3, GLM-4, GLM-4.7-Flash. "
            f"Original error: {e}"
        ) from e

    # Configure parallelism (reads from mpu which was initialized in init_megatron)
    from megatron.core import mpu

    provider = bridge.to_megatron_provider(load_weights=False)  # Load weights separately
    provider.tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
    provider.pipeline_model_parallel_size = mpu.get_pipeline_model_parallel_world_size()
    provider.expert_model_parallel_size = mpu.get_expert_model_parallel_world_size()
    provider.finalize()

    # Get model using Megatron's model factory
    model = get_model(
        model_provider_func=provider.provide,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=True,
    )

    # Load weights from HuggingFace checkpoint
    logger.info("Loading weights from: %s", config.model_name)
    bridge.load_weights(model, config.model_name, memory_efficient=True)

    logger.info("Model created: %d chunks", len(model))

    # Create optimizer
    optimizer_config = OptimizerConfig(
        bf16=config.bf16,
        fp16=config.fp16,
        lr=config.lr,
        min_lr=config.min_lr,
        weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_eps=config.adam_eps,
        clip_grad=config.clip_grad,
    )

    optimizer = get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=model,
    )

    logger.info("Optimizer created: %s", type(optimizer).__name__)

    # Create scheduler
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    scheduler = OptimizerParamScheduler(
        optimizer=optimizer,
        init_lr=config.lr,
        max_lr=config.lr,
        min_lr=config.min_lr,
        lr_warmup_steps=0,
        lr_decay_steps=1000000,  # Large number, we don't decay
        lr_decay_style="constant",
    )

    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(model, optimizer, scheduler, checkpoint_path)

    return model, optimizer, scheduler


def _load_checkpoint(
    model: list[Any],
    optimizer: Any,
    scheduler: Any,
    checkpoint_path: Path,
) -> None:
    """Load checkpoint into model/optimizer/scheduler.

    Args:
        model: Model chunks
        optimizer: Optimizer
        scheduler: LR scheduler
        checkpoint_path: Path to checkpoint directory
    """
    try:
        from megatron.training.checkpointing import load_checkpoint
    except ImportError:
        logger.warning("Megatron checkpointing not available, skipping load")
        return

    logger.info("Loading checkpoint from: %s", checkpoint_path)

    load_checkpoint(
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=scheduler,
        load_dir=str(checkpoint_path),
    )

    logger.info("Checkpoint loaded")
