"""Honest placeholder for a future native `nmoe` training backend.

The previous `NmoeTrainingBackend` was not an `nmoe` runtime adapter. It loaded
Hugging Face models and wrapped our generic PyTorch backend with an optimizer
recipe inspired by `nmoe`. That was semantically dishonest, so the runnable path
has been removed.

The reserved `backend="nmoe"` surface remains only to fail loudly while we add
the real path:
- `NmoeLowering` for runtime/distributed semantics
- `NmoeModelLowering` for native model/checkpoint construction intent
- an actual adapter to `nmoe` runtime state, RDEP dispatch, and lockstep eval
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class NmoeBackendUnavailableError(NotImplementedError):
    """Raised when callers try to use the reserved `nmoe` backend name."""


def nmoe_backend_unavailable_message(*, context: str) -> str:
    """Return the canonical fail-loud message for the reserved `nmoe` path."""
    return (
        f"{context}: backend='nmoe' is reserved but not implemented.\n"
        "The old NmoeTrainingBackend was removed because it was only a Hugging Face "
        "wrapper plus optimizer recipe, not a real nmoe runtime adapter.\n"
        "Missing pieces for the honest path:\n"
        "- native nmoe model construction/checkpoint loading\n"
        "- runtime lowering for RDEP/expert ownership semantics\n"
        "- lockstep distributed eval/generation semantics\n"
        "- nmoe-native observability and weight publication semantics"
    )


def raise_nmoe_backend_unavailable(*, context: str) -> None:
    raise NmoeBackendUnavailableError(nmoe_backend_unavailable_message(context=context))


@dataclass(frozen=True)
class NmoeConfig:
    """Reserved config surface for the future native `nmoe` backend."""

    dispatch_kind: str = "rdep"
    loader_kind: str = "nmoe_native_or_hf_bridge"
    checkpoint_kind: str = "nmoe_native"
    requires_lockstep_eval: bool = True
    requires_lockstep_generation: bool = True
    validation_notes: tuple[str, ...] = (
        "placeholder config only; no runnable nmoe backend exists yet",
    )


@dataclass
class NmoeTrainingBackend:
    """Fail-loud placeholder preserving the old symbol name."""

    model_name: str
    checkpoint_dir: Path
    loss_fn: Callable[..., Any]
    config: NmoeConfig = field(default_factory=NmoeConfig)
    device_type: str = "cuda"
    gpu_rank: int = 0
    num_minibatches: int | None = None
    max_grad_norm: float | None = 1.0
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32

    def __post_init__(self) -> None:
        raise_nmoe_backend_unavailable(context="NmoeTrainingBackend")

    def forward_backward(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise_nmoe_backend_unavailable(context="NmoeTrainingBackend.forward_backward")

    def optim_step(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise_nmoe_backend_unavailable(context="NmoeTrainingBackend.optim_step")

    def get_weights(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise_nmoe_backend_unavailable(context="NmoeTrainingBackend.get_weights")

    def load_weights(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise_nmoe_backend_unavailable(context="NmoeTrainingBackend.load_weights")
