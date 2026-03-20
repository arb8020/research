"""Named inference runtime realizations.

This is the first honest cut at keeping inference patching local to a
realization instead of smearing it across backend strings, sync flags, and
engine-class conditionals.
"""

from __future__ import annotations

from dataclasses import dataclass

from .weight_sync_protocol import (
    ENGINE_V2_HTTP_PATH_RELOAD,
    SGLANG_HTTP_PATH_RELOAD,
    VLLM_CUSTOM_NCCL_BROADCAST,
)


@dataclass(frozen=True)
class InferenceRealization:
    """Concrete inference runtime shape requested by training orchestration."""

    name: str
    backend: str
    launch_module: str
    supported_sync_realizations: tuple[str, ...] = ()
    default_sync_realization: str | None = None
    supports_blocking_updates: bool = True
    supports_inflight_updates: bool = False
    capability_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.default_sync_realization is not None:
            assert self.default_sync_realization in self.supported_sync_realizations, (
                "default_sync_realization must be included in supported_sync_realizations"
            )


SLIME_SGLANG = InferenceRealization(
    name="slime-sglang",
    backend="sglang",
    launch_module="rollouts.inference.realizations.slime_sglang",
    supported_sync_realizations=(SGLANG_HTTP_PATH_RELOAD.name,),
    default_sync_realization=SGLANG_HTTP_PATH_RELOAD.name,
    capability_notes=(
        "Current SGLang launch goes through the local slime-style wrapper module. "
        "Keep runtime patching and instrumentation scoped to this realization.",
    ),
)

VLLM = InferenceRealization(
    name="vllm",
    backend="vllm",
    launch_module="vllm.entrypoints.openai.api_server",
    capability_notes=(
        "Upstream vLLM has no truthful default live weight-sync adapter here. "
        "Use an explicit patched realization for custom update routes.",
    ),
)

QED_VLLM = InferenceRealization(
    name="qed-vllm",
    backend="vllm",
    launch_module="rollouts.inference.realizations.qed_vllm",
    supported_sync_realizations=(VLLM_CUSTOM_NCCL_BROADCAST.name,),
    default_sync_realization=VLLM_CUSTOM_NCCL_BROADCAST.name,
    capability_notes=(
        "QED-vLLM uses the patched /receive_weight_update control plane. "
        "Transport is NCCL but serving still blocks during tensor application.",
    ),
)

ENGINE_V2 = InferenceRealization(
    name="engine_v2",
    backend="engine_v2",
    launch_module="rollouts.serve.engine_v2",
    supported_sync_realizations=(ENGINE_V2_HTTP_PATH_RELOAD.name,),
    default_sync_realization=ENGINE_V2_HTTP_PATH_RELOAD.name,
)


INFERENCE_REALIZATIONS: dict[str, InferenceRealization] = {
    realization.name: realization
    for realization in (
        SLIME_SGLANG,
        VLLM,
        QED_VLLM,
        ENGINE_V2,
    )
}


def get_inference_realization(name: str) -> InferenceRealization:
    try:
        return INFERENCE_REALIZATIONS[name]
    except KeyError as exc:
        known = ", ".join(sorted(INFERENCE_REALIZATIONS))
        raise ValueError(f"Unknown inference realization {name!r}. Known: {known}") from exc
