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
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from rollouts.training.backends.megatron.qwen import build_qwen3_transformer_config
from rollouts.training.backends.megatron.qwen3_5 import get_qwen3_5_spec
from rollouts.training.backends.megatron.qwen3_next import get_qwen3_next_spec
from rollouts.training.models import (
    HFModelSource,
    ModelConstructionAdapter,
    ModelDenotation,
    lower_model_to_megatron,
    normalize_hf_model_denotation,
)
from rollouts.training.models.backend_lowering import MegatronModelLowering

logger = logging.getLogger(__name__)


class MegatronModelAdapter(ModelConstructionAdapter, Protocol):
    """Megatron-specific model construction adapter.

    This is separate from `RealizationPlan` lowering. These adapters lower model
    denotation into Megatron-native provider/model construction.
    """

    def normalize_hf_config(self, hf_config: Any) -> Any:
        """Normalize HF config into the shape this adapter expects."""
        ...

    def build_provider(
        self,
        *,
        denotation: ModelDenotation,
        runtime_config: MegatronModelConfig,
        bridge: Any,
    ) -> Any:
        """Construct a Megatron-native model provider."""
        ...

    def load_weights(self, *, bridge: Any, model: list[Any], denotation: ModelDenotation) -> None:
        """Load weights for a constructed Megatron model."""
        ...


class BridgeProviderMegatronAdapter:
    """Megatron adapter for bridge/provider-capable models."""

    adapter_name = "provider"

    def normalize_denotation(self, denotation: ModelDenotation) -> ModelDenotation:
        return denotation

    def validate_support(self, denotation: ModelDenotation, *, backend_name: str) -> None:
        assert denotation.source, f"{backend_name} model source must be non-empty"

    def normalize_hf_config(self, hf_config: Any) -> Any:
        return hf_config

    def build_provider(
        self,
        *,
        denotation: ModelDenotation,
        runtime_config: MegatronModelConfig,
        bridge: Any,
    ) -> Any:
        del denotation, runtime_config
        from megatron.core import mpu

        provider = bridge.to_megatron_provider(load_weights=False)
        provider.tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
        provider.pipeline_model_parallel_size = mpu.get_pipeline_model_parallel_world_size()
        provider.expert_model_parallel_size = mpu.get_expert_model_parallel_world_size()
        provider.finalize()
        return provider.provide

    def load_weights(self, *, bridge: Any, model: list[Any], denotation: ModelDenotation) -> None:
        bridge.load_weights(
            model,
            _materialize_hf_checkpoint_snapshot(denotation.source),
            memory_efficient=True,
        )


class RawGPTMegatronAdapter:
    """Megatron adapter for raw GPTModel construction from bridge internals.

    This is still backend-native and intentionally conservative. It exists for
    mbridge builds that do not expose `to_megatron_provider`, but it is not a
    claim that every HF model can be lowered through generic GPT defaults.
    """

    adapter_name = "raw_gpt"

    def normalize_denotation(self, denotation: ModelDenotation) -> ModelDenotation:
        return denotation

    def validate_support(self, denotation: ModelDenotation, *, backend_name: str) -> None:
        assert denotation.source, f"{backend_name} model source must be non-empty"
        if (
            denotation.architecture.family in {"qwen3", "qwen3_moe"}
            and denotation.architecture.norm == "rmsnorm"
        ):
            raise ValueError(
                f"{backend_name} raw_gpt lowering does not honestly support "
                f"{denotation.architecture.family} with RMSNorm. "
                "Use a provider-capable bridge or add a custom_spec adapter."
            )

    def normalize_hf_config(self, hf_config: Any) -> Any:
        return _normalize_hf_config_for_megatron_bridge(hf_config)

    def build_provider(
        self,
        *,
        denotation: ModelDenotation,
        runtime_config: MegatronModelConfig,
        bridge: Any,
    ) -> Any:
        del denotation
        if not hasattr(bridge, "_build_config"):
            raise AttributeError(
                "Bridge object does not expose `to_megatron_provider` or `_build_config`. "
                "This mbridge version cannot construct Megatron models directly. "
                "Consider using a megatron.bridge build instead."
            )

        from megatron.core.models.gpt import GPTModel
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_block_spec,
            get_gpt_layer_local_spec,
            get_gpt_layer_with_transformer_engine_spec,
        )

        hf_config = self.normalize_hf_config(bridge.hf_config)
        transformer_config = bridge._build_config()
        _apply_architecture_overrides(transformer_config, runtime_config)

        if hasattr(bridge, "_get_gptmodel_args"):
            gpt_kwargs = dict(bridge._get_gptmodel_args())
        else:
            gpt_kwargs = {}

        _populate_gpt_model_defaults(
            gpt_kwargs=gpt_kwargs,
            hf_config=hf_config,
            runtime_config=runtime_config,
        )

        num_experts = _infer_num_experts(runtime_config, hf_config)
        use_te = False

        # TODO: add an explicit custom-spec adapter path for model families that
        # are not honestly supported by generic raw GPT construction.
        def model_provider(
            pre_process: bool = True,
            post_process: bool = True,
            config: Any = None,
            pg_collection: Any = None,
            vp_stage: int | None = None,
        ) -> GPTModel:
            del config, pg_collection
            if num_experts:
                layer_kwargs: dict[str, bool | int] = {"use_transformer_engine": use_te}
                if vp_stage is not None:
                    layer_kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    transformer_config, **layer_kwargs
                )
            else:
                layer_spec_kwargs = _build_layer_spec_kwargs(
                    transformer_config=transformer_config,
                    num_experts=num_experts,
                )
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        **layer_spec_kwargs,
                    )
                else:
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

    def load_weights(self, *, bridge: Any, model: list[Any], denotation: ModelDenotation) -> None:
        bridge.load_weights(
            model,
            _materialize_hf_checkpoint_snapshot(denotation.source),
            memory_efficient=True,
        )


class Qwen3CustomSpecMegatronAdapter:
    """Megatron adapter for plain Qwen3 explicit model construction."""

    adapter_name = "custom_spec:qwen3"

    def normalize_denotation(self, denotation: ModelDenotation) -> ModelDenotation:
        return denotation

    def validate_support(self, denotation: ModelDenotation, *, backend_name: str) -> None:
        assert denotation.source, f"{backend_name} model source must be non-empty"
        if denotation.architecture.family != "qwen3":
            raise ValueError(
                f"{backend_name} qwen3 custom spec only supports family='qwen3', got "
                f"{denotation.architecture.family!r}"
            )
        if denotation.architecture.norm != "rmsnorm":
            raise ValueError(
                f"{backend_name} qwen3 custom spec expects RMSNorm, got {denotation.architecture.norm!r}"
            )

    def normalize_hf_config(self, hf_config: Any) -> Any:
        return _normalize_hf_config_for_megatron_bridge(hf_config)

    def build_provider(
        self,
        *,
        denotation: ModelDenotation,
        runtime_config: MegatronModelConfig,
        bridge: Any,
    ) -> Any:
        from megatron.core import mpu
        from megatron.core.models.gpt import GPTModel
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
            get_gpt_layer_with_transformer_engine_spec,
        )

        tensor_parallel_size = mpu.get_tensor_model_parallel_world_size()
        pipeline_parallel_size = mpu.get_pipeline_model_parallel_world_size()
        expert_parallel_size = mpu.get_expert_model_parallel_world_size()

        transformer_config = build_qwen3_transformer_config(
            denotation,
            seq_length=runtime_config.seq_length,
            micro_batch_size=runtime_config.micro_batch_size,
            global_batch_size=runtime_config.global_batch_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
            sequence_parallel=runtime_config.sequence_parallel,
            bf16=runtime_config.bf16,
            fp16=runtime_config.fp16,
        )
        _apply_architecture_overrides(transformer_config, runtime_config)

        hf_config = self.normalize_hf_config(bridge.hf_config)
        gpt_kwargs = {}
        if hasattr(bridge, "_get_gptmodel_args"):
            gpt_kwargs = dict(bridge._get_gptmodel_args())
        gpt_kwargs["max_sequence_length"] = runtime_config.seq_length
        _populate_gpt_model_defaults(
            gpt_kwargs=gpt_kwargs,
            hf_config=hf_config,
            runtime_config=runtime_config,
        )

        layer_spec_kwargs = _build_layer_spec_kwargs(
            transformer_config=transformer_config,
            num_experts=None,
        )
        use_te = getattr(transformer_config, "transformer_impl", "local") == "transformer_engine"
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(**layer_spec_kwargs)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(**layer_spec_kwargs)

        def model_provider(
            pre_process: bool = True,
            post_process: bool = True,
            config: Any = None,
            pg_collection: Any = None,
            vp_stage: int | None = None,
        ) -> GPTModel:
            del config, pg_collection
            kwargs = dict(gpt_kwargs)
            kwargs.update({
                "config": transformer_config,
                "transformer_layer_spec": transformer_layer_spec,
                "pre_process": pre_process,
                "post_process": post_process,
            })
            if vp_stage is not None:
                kwargs["vp_stage"] = vp_stage
            return GPTModel(**kwargs)

        return model_provider

    def load_weights(self, *, bridge: Any, model: list[Any], denotation: ModelDenotation) -> None:
        bridge.load_weights(
            model,
            _materialize_hf_checkpoint_snapshot(denotation.source),
            memory_efficient=True,
        )


class BridgeCustomSpecMegatronAdapter:
    """Megatron adapter for families that need a custom transformer layer spec."""

    adapter_name = "custom_spec"
    expected_families: tuple[str, ...] = ()
    spec_builder: Any = None

    def normalize_denotation(self, denotation: ModelDenotation) -> ModelDenotation:
        return denotation

    def validate_support(self, denotation: ModelDenotation, *, backend_name: str) -> None:
        assert denotation.source, f"{backend_name} model source must be non-empty"
        if denotation.architecture.family not in self.expected_families:
            raise ValueError(
                f"{backend_name} {self.adapter_name} expects one of {self.expected_families!r}, got "
                f"{denotation.architecture.family!r}"
            )
        if self.spec_builder is None:
            raise ValueError(f"{backend_name} {self.adapter_name} has no spec_builder configured")

    def normalize_hf_config(self, hf_config: Any) -> Any:
        return _normalize_hf_config_for_megatron_bridge(hf_config)

    def build_provider(
        self,
        *,
        denotation: ModelDenotation,
        runtime_config: MegatronModelConfig,
        bridge: Any,
    ) -> Any:
        from megatron.core.models.gpt import GPTModel

        if not hasattr(bridge, "_build_config"):
            raise AttributeError(
                "Bridge object does not expose `_build_config` for custom-spec Megatron loading."
            )

        transformer_config = bridge._build_config()
        _apply_architecture_overrides(transformer_config, runtime_config)

        hf_config = self.normalize_hf_config(bridge.hf_config)
        gpt_kwargs = {}
        if hasattr(bridge, "_get_gptmodel_args"):
            gpt_kwargs = dict(bridge._get_gptmodel_args())
        _populate_gpt_model_defaults(
            gpt_kwargs=gpt_kwargs,
            hf_config=hf_config,
            runtime_config=runtime_config,
        )

        spec_args = SimpleNamespace(
            hf_checkpoint=denotation.source.name_or_path,
            num_experts=runtime_config.num_experts,
            sequence_parallel=runtime_config.sequence_parallel,
        )

        def model_provider(
            pre_process: bool = True,
            post_process: bool = True,
            config: Any = None,
            pg_collection: Any = None,
            vp_stage: int | None = None,
        ) -> GPTModel:
            del config, pg_collection
            transformer_layer_spec = self.spec_builder(spec_args, transformer_config, vp_stage)
            kwargs = dict(gpt_kwargs)
            kwargs.update({
                "config": transformer_config,
                "transformer_layer_spec": transformer_layer_spec,
                "pre_process": pre_process,
                "post_process": post_process,
            })
            if vp_stage is not None:
                kwargs["vp_stage"] = vp_stage
            return GPTModel(**kwargs)

        return model_provider

    def load_weights(self, *, bridge: Any, model: list[Any], denotation: ModelDenotation) -> None:
        bridge.load_weights(
            model,
            _materialize_hf_checkpoint_snapshot(denotation.source),
            memory_efficient=True,
        )


class Qwen35CustomSpecMegatronAdapter(BridgeCustomSpecMegatronAdapter):
    adapter_name = "custom_spec:qwen3_5"
    expected_families = ("qwen3_5", "qwen3_5_moe")
    spec_builder = staticmethod(get_qwen3_5_spec)


class Qwen3NextCustomSpecMegatronAdapter(BridgeCustomSpecMegatronAdapter):
    adapter_name = "custom_spec:qwen3_next"
    expected_families = ("qwen3_next",)
    spec_builder = staticmethod(get_qwen3_next_spec)


def _adapter_for_lowering(lowering: MegatronModelLowering) -> MegatronModelAdapter:
    """Select the honest Megatron model-construction adapter for lowered intent."""
    if lowering.adapter_kind == "provider":
        return BridgeProviderMegatronAdapter()
    if lowering.adapter_kind == "raw_gpt":
        return RawGPTMegatronAdapter()
    if lowering.adapter_kind == "custom_spec":
        if lowering.denotation.architecture.family == "qwen3":
            return Qwen3CustomSpecMegatronAdapter()
        if lowering.denotation.architecture.family in {"qwen3_5", "qwen3_5_moe"}:
            return Qwen35CustomSpecMegatronAdapter()
        if lowering.denotation.architecture.family == "qwen3_next":
            return Qwen3NextCustomSpecMegatronAdapter()
    raise ValueError(f"Unsupported Megatron adapter kind: {lowering.adapter_kind!r}")


def _materialize_hf_checkpoint_snapshot(source: HFModelSource) -> str:
    """Resolve an HF checkpoint source into a local snapshot with real weight files.

    Passing a repo id directly into bridge-native loaders makes checkpoint
    staging opaque. Materialize and validate the snapshot explicitly first so
    backend model loading operates on a local, inspectable checkpoint boundary.
    """
    checkpoint_path = Path(source.name_or_path)
    if checkpoint_path.exists():
        _assert_hf_snapshot_has_weights(checkpoint_path)
        return str(checkpoint_path)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to materialize HF checkpoints for Megatron loading"
        ) from e

    snapshot_path = Path(
        snapshot_download(
            repo_id=source.name_or_path,
            revision=source.revision,
            allow_patterns=[
                "config.json",
                "*.safetensors",
                "*.safetensors.index.json",
            ],
        )
    )
    _assert_hf_snapshot_has_weights(snapshot_path)
    return str(snapshot_path)


def _assert_hf_snapshot_has_weights(snapshot_path: Path) -> None:
    weight_files = sorted(snapshot_path.glob("*.safetensors"))
    index_files = sorted(snapshot_path.glob("*.safetensors.index.json"))
    if weight_files or index_files:
        logger.info(
            "Using local HF checkpoint snapshot %s with weight files: %s",
            snapshot_path,
            [p.name for p in (*weight_files, *index_files)],
        )
        return

    available_files = (
        sorted(p.name for p in snapshot_path.iterdir()) if snapshot_path.is_dir() else []
    )
    raise ValueError(
        f"HF checkpoint snapshot {snapshot_path} does not contain safetensor weights. "
        f"Available files: {available_files}"
    )


def _infer_rope_theta(hf_config: Any) -> float | int | None:
    """Extract rope theta from common HuggingFace config layouts."""
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is not None:
        return rope_theta

    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is not None:
            return rope_theta

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_theta = rope_scaling.get("rope_theta")
        if rope_theta is not None:
            return rope_theta

    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        return _infer_rope_theta(text_config)

    return None


def _normalize_hf_config_for_megatron_bridge(hf_config: Any) -> Any:
    """Normalize HF config shape to match Megatron bridge expectations.

    Some model families store rotary metadata under nested config fields such as
    `rope_parameters` or `rope_scaling`, while mbridge still reads a top-level
    `hf_config.rope_theta`. Normalize that at the backend-native model boundary
    before touching bridge internals.
    """
    rope_theta = _infer_rope_theta(hf_config)
    if rope_theta is None:
        return hf_config

    if getattr(hf_config, "rope_theta", None) is None:
        hf_config.rope_theta = rope_theta

    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None and getattr(text_config, "rope_theta", None) is None:
        text_config.rope_theta = rope_theta

    return hf_config


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
    sequence_parallel: bool = False

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

    # Normalize published HF semantics first, then lower that denotation into
    # honest Megatron-native model-construction intent.
    source = HFModelSource(
        name_or_path=config.model_name,
        trust_remote_code=config.trust_remote_code,
    )
    denotation = normalize_hf_model_denotation(source)
    logger.info("Loading model via AutoBridge: %s", denotation.source.name_or_path)

    try:
        bridge = AutoBridge.from_pretrained(
            denotation.source.name_or_path,
            trust_remote_code=denotation.source.trust_remote_code,
        )
    except Exception as e:
        raise ValueError(
            f"AutoBridge failed to load model '{denotation.source.name_or_path}'. "
            f"This architecture may not be supported. "
            f"Supported: Llama, Qwen, DeepseekV3, GLM-4, GLM-4.7-Flash. "
            f"Original error: {e}"
        ) from e

    model_lowering = lower_model_to_megatron(
        denotation,
        bridge_supports_provider=hasattr(bridge, "to_megatron_provider"),
        architecture_args=config.architecture_args,
    )
    for note in model_lowering.validation_notes:
        logger.warning("Megatron model lowering note: %s", note)

    adapter = _adapter_for_lowering(model_lowering)
    denotation = adapter.normalize_denotation(denotation)
    adapter.validate_support(denotation, backend_name="megatron")
    logger.info(
        "Using Megatron model adapter: %s for family=%s variant=%s",
        adapter.adapter_name,
        denotation.architecture.family,
        denotation.variant,
    )
    provider = adapter.build_provider(
        denotation=denotation,
        runtime_config=config,
        bridge=bridge,
    )

    model = get_model(
        model_provider_func=provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=True,
    )

    # Load weights from HuggingFace checkpoint
    logger.info("Loading weights from: %s", denotation.source.name_or_path)
    adapter.load_weights(bridge=bridge, model=model, denotation=denotation)

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


def _populate_gpt_model_defaults(
    *,
    gpt_kwargs: dict[str, Any],
    hf_config: Any,
    runtime_config: MegatronModelConfig,
) -> None:
    """Populate GPTModel kwargs from model denotation plus runtime config."""
    gpt_kwargs.setdefault("vocab_size", getattr(hf_config, "vocab_size", None))
    max_sequence_length = gpt_kwargs.get("max_sequence_length")
    if max_sequence_length is None:
        max_sequence_length = getattr(hf_config, "max_position_embeddings", None)
    if max_sequence_length is None:
        max_sequence_length = getattr(hf_config, "max_seq_len", None)
    if max_sequence_length is None:
        raise ValueError(
            f"Unable to infer max sequence length from model config: {runtime_config.model_name}"
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
        gpt_kwargs["fp16_lm_cross_entropy"] = runtime_config.fp16
    if gpt_kwargs.get("parallel_output") is None:
        gpt_kwargs["parallel_output"] = True

    gpt_kwargs.update(runtime_config.architecture_args)


def _infer_num_experts(runtime_config: MegatronModelConfig, hf_config: Any) -> int | None:
    """Infer MoE expert count from explicit runtime config or HF config."""
    if runtime_config.num_experts is not None:
        return runtime_config.num_experts
    return getattr(hf_config, "n_routed_experts", 0) or getattr(hf_config, "num_experts", 0)


def _build_layer_spec_kwargs(*, transformer_config: Any, num_experts: int | None) -> dict[str, Any]:
    """Build shared Megatron GPT layer-spec kwargs."""
    return {
        "num_experts": num_experts,
        "moe_grouped_gemm": getattr(transformer_config, "moe_grouped_gemm", False),
        "qk_layernorm": getattr(transformer_config, "qk_layernorm", False),
        "multi_latent_attention": getattr(transformer_config, "multi_latent_attention", False),
        "moe_use_legacy_grouped_gemm": getattr(
            transformer_config, "moe_use_legacy_grouped_gemm", False
        ),
    }


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
