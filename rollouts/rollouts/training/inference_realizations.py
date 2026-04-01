"""Named inference engine specs.

An InferenceEngineSpec is the complete reproducible identity of one inference
engine variant: where the code comes from (deps), how to launch it
(launch_module, launch_env), and what capabilities it exposes
(sync realizations, blocking/inflight).

This is the right place to add a new engine fork: define a new
InferenceEngineSpec with its own deps pinning and register it below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .configs import DepsConfig

from .weight_sync_protocol import (
    ENGINE_V2_HTTP_PATH_RELOAD,
    SGLANG_HTTP_PATH_RELOAD,
    VLLM_CUSTOM_NCCL_BROADCAST,
)

InferenceApiFormat = Literal["sglang", "vllm"]


@dataclass(frozen=True)
class InferenceEngineSpec:
    """Complete reproducible identity of one inference engine variant.

    Fields:
        name: Unique identifier used in configs (e.g. "slime-sglang").
        launch_module: Python module run as `python -m <launch_module>`.
        api_format: Wire format the HTTP server speaks. Used by eval/client
            code to pick the right API shape. "sglang" and "vllm" both expose
            an OpenAI-compatible /v1/chat/completions surface but differ in
            custom endpoints (/update_weights_from_disk, etc.).
        deps: Pip packages + bootstrap commands needed to install and run this
            engine. None means the engine is provided by the current repo
            environment (no extra install required).
        launch_env: Environment variables the server process requires.
        supported_sync_realizations: Names of InferenceSyncRealizations this
            engine supports for trainer->inference weight updates.
        default_sync_realization: Which sync realization to use when none is
            explicitly requested.
        supports_blocking_updates: Engine can receive weight updates while
            pausing serving (all current engines).
        supports_inflight_updates: Engine can receive weight updates without
            pausing serving (future, for true pipeline parallelism).
        capability_notes: Human-readable notes about known limitations or
            constraints. Surfaced in error messages.
    """

    name: str
    launch_module: str
    api_format: InferenceApiFormat
    deps: DepsConfig | None = None
    launch_env: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    supported_sync_realizations: tuple[str, ...] = field(default_factory=tuple)
    default_sync_realization: str | None = None
    supports_blocking_updates: bool = True
    supports_inflight_updates: bool = False
    capability_notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        assert self.name, "name cannot be empty"
        assert self.launch_module, "launch_module cannot be empty"
        if self.default_sync_realization is not None:
            assert self.default_sync_realization in self.supported_sync_realizations, (
                f"default_sync_realization {self.default_sync_realization!r} must be "
                "included in supported_sync_realizations"
            )


# ---------------------------------------------------------------------------
# Named engine specs
# ---------------------------------------------------------------------------

SLIME_SGLANG = InferenceEngineSpec(
    name="slime-sglang",
    launch_module="rollouts.inference.realizations.slime_sglang",
    api_format="sglang",
    deps=None,  # Provided by the repo environment; callers supply sglang install via DepsConfig
    launch_env=(
        ("AMEM_ENABLE", "1"),
        ("NCCL_CUMEM_ENABLE", "0"),
        ("NCCL_ASYNC_ERROR_HANDLING", "1"),
        ("ROLLOUTS_SGLANG_FORCE_SYNC_BROADCAST", "1"),
    ),
    supported_sync_realizations=(SGLANG_HTTP_PATH_RELOAD.name,),
    default_sync_realization=SGLANG_HTTP_PATH_RELOAD.name,
    capability_notes=(
        "SGLang launch via local slime-style wrapper module. "
        "Runtime patching and instrumentation are scoped to this spec.",
    ),
)

VLLM = InferenceEngineSpec(
    name="vllm",
    launch_module="vllm.entrypoints.openai.api_server",
    api_format="vllm",
    deps=None,  # Caller provides vllm install
    capability_notes=(
        "Upstream vLLM with no live weight-sync adapter. Use qed-vllm for NCCL weight updates.",
    ),
)

QED_VLLM = InferenceEngineSpec(
    name="qed-vllm",
    launch_module="rollouts.inference.realizations.qed_vllm",
    api_format="vllm",
    deps=None,  # Caller provides vllm install; qed_vllm module is in this repo
    launch_env=(
        ("VLLM_SERVER_DEV_MODE", "1"),
        ("NCCL_CUMEM_ENABLE", "0"),
        ("NCCL_ASYNC_ERROR_HANDLING", "1"),
    ),
    supported_sync_realizations=(VLLM_CUSTOM_NCCL_BROADCAST.name,),
    default_sync_realization=VLLM_CUSTOM_NCCL_BROADCAST.name,
    capability_notes=(
        "Patched vLLM with /receive_weight_update control plane. "
        "Transport is NCCL; serving still blocks during tensor application.",
    ),
)

ENGINE_V2 = InferenceEngineSpec(
    name="engine_v2",
    launch_module="rollouts.serve.engine_v2",
    api_format="sglang",  # exposes the same /update_weights_from_disk endpoint
    deps=None,  # In-repo engine, no extra install
    supported_sync_realizations=(ENGINE_V2_HTTP_PATH_RELOAD.name,),
    default_sync_realization=ENGINE_V2_HTTP_PATH_RELOAD.name,
)


INFERENCE_ENGINE_SPECS: dict[str, InferenceEngineSpec] = {
    spec.name: spec
    for spec in (
        SLIME_SGLANG,
        VLLM,
        QED_VLLM,
        ENGINE_V2,
    )
}


def get_inference_engine_spec(name: str) -> InferenceEngineSpec:
    try:
        return INFERENCE_ENGINE_SPECS[name]
    except KeyError as exc:
        known = ", ".join(sorted(INFERENCE_ENGINE_SPECS))
        raise ValueError(f"Unknown inference engine spec {name!r}. Known: {known}") from exc
