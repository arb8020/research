"""Inference backend/runtime construction.

This module owns the honest inference-side setup story for training consumers.
It centralizes concrete engine construction and validates capability-dependent
pipeline semantics before GRPO starts launching servers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rollouts.eval.configs import EndpointCapabilities, OwnedEndpoint

from .configs import CheckpointConfig, InferenceConfig, ModelConfig, RolloutConfig
from .inference_realizations import (
    InferenceEngineSpec,
    get_inference_engine_spec,
)
from .weight_sync import EngineV2Engine, InferenceBackend, SGLangEngine, VLLMEngine
from .weight_sync_protocol import (
    InferenceSyncRealization,
    get_inference_sync_realization,
    resolve_inference_sync_realization,
)


@dataclass(frozen=True)
class InferenceRuntimePlan:
    """Concrete inference runtime plus validated sync semantics."""

    engines: tuple[InferenceBackend, ...]
    spec: InferenceEngineSpec
    sync_realization: InferenceSyncRealization | None = None


def build_owned_endpoint(
    *,
    spec: InferenceEngineSpec,
    model: str,
    cuda_device_ids: tuple[int, ...],
    port: int,
    output_dir: Path,
    capabilities: EndpointCapabilities,
    **engine_kwargs: object,
) -> OwnedEndpoint:
    if spec.name in ("slime-sglang", "mini-sglang", "harvest-sglang"):
        engine = SGLangEngine(
            model_name=model,
            port=port,
            cuda_device_ids=cuda_device_ids,
            output_dir=output_dir,
            dtype=str(engine_kwargs.get("dtype", "bfloat16")),
            mem_fraction=float(engine_kwargs.get("mem_fraction", 0.7)),
            disable_cuda_graph=bool(engine_kwargs.get("disable_cuda_graph", False)),
            max_total_tokens=engine_kwargs.get("max_total_tokens"),
            max_prefill_tokens=engine_kwargs.get("max_prefill_tokens"),
            max_running_requests=engine_kwargs.get("max_running_requests"),
            chunked_prefill_size=engine_kwargs.get("chunked_prefill_size"),
            harvest_layers=engine_kwargs.get("harvest_layers"),
            harvest_output_dir=engine_kwargs.get("harvest_output_dir"),
            timeout=float(engine_kwargs.get("startup_timeout", 300.0)),
            realization_name=spec.name,
            launch_module=spec.launch_module,
            capability_notes=spec.capability_notes,
            available_sync_realizations=spec.supported_sync_realizations,
            default_sync_realization=capabilities.weight_sync or spec.default_sync_realization,
        )
        return engine.as_owned_endpoint()
    if spec.name in ("vllm", "qed-vllm", "trtllm"):
        engine = VLLMEngine(
            model_name=model,
            port=port,
            cuda_device_ids=cuda_device_ids,
            output_dir=output_dir,
            dtype=str(engine_kwargs.get("dtype", "bfloat16")),
            gpu_memory_utilization=float(engine_kwargs.get("mem_fraction", 0.7)),
            timeout=float(engine_kwargs.get("startup_timeout", 300.0)),
            realization_name=spec.name,
            launch_module=spec.launch_module,
            capability_notes=spec.capability_notes,
            available_sync_realizations=spec.supported_sync_realizations,
            default_sync_realization=capabilities.weight_sync or spec.default_sync_realization,
        )
        return OwnedEndpoint(
            spec=spec.name,
            model=model,
            cuda_device_ids=cuda_device_ids,
            port=port,
            capabilities=capabilities,
            output_dir=output_dir,
            launch_cmd=engine.build_launch_cmd(),
            mem_fraction=float(engine_kwargs.get("mem_fraction", 0.7)),
            startup_timeout=float(engine_kwargs.get("startup_timeout", 300.0)),
        )
    raise ValueError(f"Cannot construct owned endpoint for spec {spec.name!r}")


def _validate_sync_contract(
    *,
    spec: InferenceEngineSpec,
    checkpoint: CheckpointConfig,
) -> None:
    requested = checkpoint.inference_sync_realization
    if requested is None:
        return
    sync_realization = get_inference_sync_realization(requested)
    if sync_realization.requires_custom_server_patch and not spec.supported_sync_realizations:
        raise ValueError(
            f"Inference sync realization {requested!r} requires a patched inference "
            f"server, but engine spec {spec.name!r} does not advertise "
            "a patched sync surface."
        )


def _create_engine(
    *,
    spec: InferenceEngineSpec,
    model: ModelConfig,
    inference: InferenceConfig,
    rollout: RolloutConfig,
    checkpoint: CheckpointConfig,
    output_dir: Path,
    gpus: tuple[int, ...],
    port: int,
) -> InferenceBackend:
    # Dispatch on spec.name so each named spec maps to exactly one engine class.
    # New forks: add a new InferenceEngineSpec and a branch here.
    #
    # MIGRATION: This function and _build_engine() in eval/endpoint_realization.py
    # are the same logic in two places. Once OwnedEndpoint is fully implemented
    # (see eval/configs.py), consolidate both into a single factory here:
    #
    #   def build_owned_endpoint(
    #       spec: InferenceEngineSpec,
    #       model: str,
    #       cuda_device_ids: tuple[int, ...],
    #       port: int,
    #       output_dir: Path,
    #       capabilities: EndpointCapabilities,
    #       **engine_kwargs,
    #   ) -> OwnedEndpoint: ...
    #
    # Then eval/endpoint_realization._build_engine() calls this directly,
    # and _create_engine() wraps it to add weight_sync to capabilities.
    # That removes the duplication and makes OwnedEndpoint the single
    # implementation for both eval and RL lifecycle management.
    if spec.name in ("slime-sglang", "mini-sglang", "harvest-sglang"):
        return SGLangEngine(
            model_name=model.name,
            port=port,
            cuda_device_ids=gpus,
            output_dir=output_dir,
            dtype=model.dtype,
            mem_fraction=inference.mem_fraction,
            disable_cuda_graph=inference.disable_cuda_graph,
            max_total_tokens=inference.max_total_tokens,
            max_prefill_tokens=inference.max_prefill_tokens,
            max_running_requests=inference.max_running_requests,
            chunked_prefill_size=inference.chunked_prefill_size,
            harvest_layers=getattr(inference, "harvest_layers", None),
            harvest_output_dir=getattr(inference, "harvest_output_dir", None),
            realization_name=spec.name,
            launch_module=spec.launch_module,
            capability_notes=spec.capability_notes,
            available_sync_realizations=spec.supported_sync_realizations,
            default_sync_realization=spec.default_sync_realization,
        )
    if spec.name in ("vllm", "qed-vllm", "trtllm"):
        return VLLMEngine(
            model_name=model.name,
            port=port,
            cuda_device_ids=gpus,
            output_dir=output_dir,
            dtype=model.dtype,
            gpu_memory_utilization=inference.mem_fraction,
            realization_name=spec.name,
            launch_module=spec.launch_module,
            capability_notes=spec.capability_notes,
            available_sync_realizations=spec.supported_sync_realizations,
            default_sync_realization=spec.default_sync_realization,
        )
    if spec.name == "engine_v2":
        max_batch = rollout.batch_size * rollout.n_samples_per_prompt * 2
        return EngineV2Engine(
            model_name=model.name,
            port=port,
            cuda_device_ids=gpus,
            output_dir=output_dir,
            dtype=model.dtype,
            mem_fraction=inference.mem_fraction,
            max_batch_size=max_batch,
            max_seq_len=rollout.max_seq_len,
        )
    raise ValueError(f"Cannot construct engine for spec {spec.name!r}")


def _resolve_sync_realization(
    *,
    engines: tuple[InferenceBackend, ...],
    checkpoint: CheckpointConfig,
) -> InferenceSyncRealization | None:
    requested = checkpoint.inference_sync_realization
    if requested is None and checkpoint.pipeline_mode != "true_pipeline":
        return None
    return resolve_inference_sync_realization(engines[0].capabilities, requested)


def _validate_pipeline_mode(
    *,
    engines: tuple[InferenceBackend, ...],
    checkpoint: CheckpointConfig,
    sync_realization: InferenceSyncRealization | None,
) -> None:
    if checkpoint.pipeline_mode != "true_pipeline":
        return
    if checkpoint.weight_sync_mode != "nccl":
        raise ValueError(
            "checkpoint.pipeline_mode='true_pipeline' requires checkpoint.weight_sync_mode='nccl'."
        )

    capabilities = engines[0].capabilities
    if capabilities.supports_inflight_updates:
        return

    details = []
    if sync_realization is not None:
        details.append(f"sync realization {sync_realization.name!r}")
    if capabilities.capability_notes:
        details.append(" ".join(capabilities.capability_notes))
    detail_suffix = f" {' '.join(details)}" if details else ""
    raise ValueError(
        "checkpoint.pipeline_mode='true_pipeline' requires an inference runtime "
        "with truthful inflight weight-update semantics. "
        f"Engine spec {capabilities.backend_name!r} currently advertises only "
        f"blocking updates.{detail_suffix}"
    )


def create_inference_backend_runtime(
    *,
    model: ModelConfig,
    inference: InferenceConfig,
    rollout: RolloutConfig,
    checkpoint: CheckpointConfig,
    output_dir: Path,
) -> InferenceRuntimePlan:
    """Construct inference engines plus the validated sync realization."""

    spec = get_inference_engine_spec(inference.spec)
    _validate_sync_contract(spec=spec, checkpoint=checkpoint)
    engines = tuple(
        _create_engine(
            spec=spec,
            model=model,
            inference=inference,
            rollout=rollout,
            checkpoint=checkpoint,
            output_dir=output_dir,
            gpus=gpus,
            port=port,
        )
        for gpus, port in zip(inference.gpu_assignments, inference.ports, strict=False)
    )
    if not engines:
        raise ValueError("Inference runtime requires at least one engine assignment.")
    sync_realization = _resolve_sync_realization(engines=engines, checkpoint=checkpoint)
    _validate_pipeline_mode(
        engines=engines,
        checkpoint=checkpoint,
        sync_realization=sync_realization,
    )
    return InferenceRuntimePlan(
        engines=engines,
        spec=spec,
        sync_realization=sync_realization,
    )
