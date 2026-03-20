"""Inference backend/runtime construction.

This module owns the honest inference-side setup story for training consumers.
It centralizes concrete engine construction and validates capability-dependent
pipeline semantics before GRPO starts launching servers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .configs import CheckpointConfig, InferenceConfig, ModelConfig, RolloutConfig
from .inference_realizations import (
    ENGINE_V2,
    QED_VLLM,
    SLIME_SGLANG,
    VLLM,
    InferenceRealization,
    get_inference_realization,
)
from .weight_sync import EngineV2Engine, InferenceBackend, SGLangEngine, VLLMEngine
from .weight_sync_protocol import (
    VLLM_CUSTOM_NCCL_BROADCAST,
    InferenceSyncRealization,
    get_inference_sync_realization,
    resolve_inference_sync_realization,
)


@dataclass(frozen=True)
class InferenceRuntimePlan:
    """Concrete inference runtime plus validated sync semantics."""

    engines: tuple[InferenceBackend, ...]
    realization: InferenceRealization
    sync_realization: InferenceSyncRealization | None = None


def _resolve_realization(
    *,
    inference: InferenceConfig,
    checkpoint: CheckpointConfig,
) -> InferenceRealization:
    if inference.realization is not None:
        realization = get_inference_realization(inference.realization)
        if realization.backend != inference.backend:
            raise ValueError(
                f"Inference realization {realization.name!r} requires backend "
                f"{realization.backend!r}, got {inference.backend!r}."
            )
        return realization

    if inference.backend == "sglang":
        return SLIME_SGLANG
    if inference.backend == "engine_v2":
        return ENGINE_V2
    if inference.backend == "vllm":
        requested_sync_realization = checkpoint.inference_sync_realization
        if requested_sync_realization == VLLM_CUSTOM_NCCL_BROADCAST.name:
            return QED_VLLM
        return VLLM
    raise ValueError(f"Unknown inference backend: {inference.backend}")


def _validate_realization_sync_contract(
    *,
    realization: InferenceRealization,
    checkpoint: CheckpointConfig,
) -> None:
    requested = checkpoint.inference_sync_realization
    if requested is None:
        return
    sync_realization = get_inference_sync_realization(requested)
    if (
        sync_realization.requires_custom_server_patch
        and not realization.supported_sync_realizations
    ):
        raise ValueError(
            f"Inference sync realization {requested!r} requires a patched inference "
            f"server, but inference realization {realization.name!r} does not advertise "
            "a patched sync surface."
        )


def _create_engine(
    *,
    realization: InferenceRealization,
    model: ModelConfig,
    inference: InferenceConfig,
    rollout: RolloutConfig,
    checkpoint: CheckpointConfig,
    output_dir: Path,
    gpus: tuple[int, ...],
    port: int,
) -> InferenceBackend:
    if realization.backend == "sglang":
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
            realization_name=realization.name,
            launch_module=realization.launch_module,
            capability_notes=realization.capability_notes,
            available_sync_realizations=realization.supported_sync_realizations,
            default_sync_realization=realization.default_sync_realization,
        )
    if realization.backend == "vllm":
        return VLLMEngine(
            model_name=model.name,
            port=port,
            cuda_device_ids=gpus,
            output_dir=output_dir,
            dtype=model.dtype,
            gpu_memory_utilization=inference.mem_fraction,
            realization_name=realization.name,
            launch_module=realization.launch_module,
            capability_notes=realization.capability_notes,
            available_sync_realizations=realization.supported_sync_realizations,
            default_sync_realization=realization.default_sync_realization,
        )
    if realization.backend == "engine_v2":
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
    raise ValueError(f"Unknown inference backend: {realization.backend}")


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
        f"Inference backend {capabilities.backend_name!r} currently advertises only "
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

    realization = _resolve_realization(
        inference=inference,
        checkpoint=checkpoint,
    )
    _validate_realization_sync_contract(
        realization=realization,
        checkpoint=checkpoint,
    )
    engines = tuple(
        _create_engine(
            realization=realization,
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
        realization=realization,
        sync_realization=sync_realization,
    )
