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

    provider = _build_model_provider(config, bridge)

    model = get_model(
        model_provider_func=provider,
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


def _build_model_provider(config: MegatronModelConfig, bridge: Any) -> Any:
    """Build a Megatron model provider without bridge-specific provider APIs.

    Current mbridge pip package does not expose `to_megatron_provider` on
    per-model bridge objects. In that case, we build the model directly from the
    bridge-generated Megatron/Transformer config and rely on mbridge for weight
    loading only.
    """
    from megatron.core import mpu
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_block_spec,
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from transformers import AutoConfig

    # Backward-compatible path when the old API is available.
    if hasattr(bridge, "to_megatron_provider"):
        provider = bridge.to_megatron_provider(load_weights=False)  # type: ignore[attr-defined]
        provider.tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
        provider.pipeline_model_parallel_size = mpu.get_pipeline_model_parallel_world_size()
        provider.expert_model_parallel_size = mpu.get_expert_model_parallel_world_size()
        provider.finalize()
        return provider.provide

    logger.info("mbridge does not provide to_megatron_provider; using raw model-provider path")
    if not hasattr(bridge, "_build_config"):
        raise AttributeError(
            "Bridge object does not expose `to_megatron_provider` or `_build_config`. "
            "This mbridge version cannot construct Megatron models directly. "
            "Consider using a megatron.bridge build instead."
        )

    hf_config = AutoConfig.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    transformer_config = bridge._build_config()

    # Apply lightweight overrides from rollouts config.
    _apply_architecture_overrides(transformer_config, config)

    if hasattr(bridge, "_get_gptmodel_args"):
        gpt_kwargs = dict(bridge._get_gptmodel_args())  # type: ignore[attr-defined]
    else:
        gpt_kwargs = {}

    # Fill args required by GPTModel defaults.
    gpt_kwargs.setdefault("vocab_size", getattr(hf_config, "vocab_size", None))
    max_sequence_length = gpt_kwargs.get("max_sequence_length")
    if max_sequence_length is None:
        max_sequence_length = getattr(hf_config, "max_position_embeddings", None)
    if max_sequence_length is None:
        max_sequence_length = getattr(hf_config, "max_seq_len", None)
    if max_sequence_length is None:
        raise ValueError(
            f"Unable to infer max sequence length from model config: {config.model_name}"
        )
    gpt_kwargs["max_sequence_length"] = max_sequence_length

    gpt_kwargs.setdefault("position_embedding_type", "rope")
    gpt_kwargs.setdefault("rotary_percent", 1.0)
    if gpt_kwargs.get("rotary_base") is None:
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = getattr(hf_config, "rope_scaling", {}).get("rope_theta")
        gpt_kwargs["rotary_base"] = rope_theta if rope_theta is not None else 10000.0

    if gpt_kwargs.get("share_embeddings_and_output_weights") is None:
        gpt_kwargs["share_embeddings_and_output_weights"] = True
    if gpt_kwargs.get("fp16_lm_cross_entropy") is None:
        gpt_kwargs["fp16_lm_cross_entropy"] = config.fp16
    if gpt_kwargs.get("parallel_output") is None:
        gpt_kwargs["parallel_output"] = True

    gpt_kwargs.update(config.architecture_args)

    # Determine expert settings.
    num_experts = config.num_experts
    if num_experts is None:
        num_experts = getattr(hf_config, "n_routed_experts", 0) or getattr(
            hf_config, "num_experts", 0
        )

    use_te = False

    def model_provider(
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
        **kwargs: Any,  # Megatron's get_model() passes config=... and other args
    ) -> GPTModel:
        if num_experts:
            layer_kwargs: dict[str, bool | int] = {"use_transformer_engine": use_te}
            if vp_stage is not None:
                layer_kwargs["vp_stage"] = vp_stage
            transformer_layer_spec = get_gpt_decoder_block_spec(transformer_config, **layer_kwargs)
        else:
            if use_te:
                layer_spec_kwargs = {
                    "num_experts": num_experts,
                    "moe_grouped_gemm": getattr(transformer_config, "moe_grouped_gemm", False),
                    "qk_layernorm": getattr(transformer_config, "qk_layernorm", False),
                    "multi_latent_attention": getattr(
                        transformer_config, "multi_latent_attention", False
                    ),
                    "moe_use_legacy_grouped_gemm": getattr(
                        transformer_config, "moe_use_legacy_grouped_gemm", False
                    ),
                }
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    **layer_spec_kwargs,
                )
            else:
                layer_spec_kwargs = {
                    "num_experts": num_experts,
                    "moe_grouped_gemm": getattr(transformer_config, "moe_grouped_gemm", False),
                    "qk_layernorm": getattr(transformer_config, "qk_layernorm", False),
                    "multi_latent_attention": getattr(
                        transformer_config, "multi_latent_attention", False
                    ),
                    "moe_use_legacy_grouped_gemm": getattr(
                        transformer_config, "moe_use_legacy_grouped_gemm", False
                    ),
                }
                transformer_layer_spec = get_gpt_layer_local_spec(**layer_spec_kwargs)

        kwargs = dict(gpt_kwargs)
        kwargs.update({
            "config": transformer_config,
            "transformer_layer_spec": transformer_layer_spec,
            "pre_process": pre_process,
            "post_process": post_process,
        })
        if vp_stage is not None and "vp_stage" not in kwargs:
            kwargs["vp_stage"] = vp_stage
        return GPTModel(**kwargs)

    return model_provider


def _apply_architecture_overrides(transformer_config: Any, config: MegatronModelConfig) -> None:
    """Apply explicit MegatronModelConfig fields to a generated Megatron config."""
    if not hasattr(transformer_config, "__dict__"):
        return

    overrides = {
        "num_moe_experts": config.num_experts,
        "moe_router_topk": config.moe_router_topk,
        "moe_ffn_hidden_size": config.moe_ffn_hidden_size,
        "multi_latent_attention": config.multi_latent_attention,
        "q_lora_rank": config.q_lora_rank,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_head_dim": config.qk_head_dim,
        "v_head_dim": config.v_head_dim,
    }
    for key, value in overrides.items():
        if value is not None and hasattr(transformer_config, key):
            setattr(transformer_config, key, value)

    if not config.architecture_args:
        return

    for key, value in config.architecture_args.items():
        if hasattr(transformer_config, key):
            setattr(transformer_config, key, value)


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
