"""Lowering-oriented training objects.

These summarize backend provisioning choices derived from higher-level
realization semantics. They are not the semantic center of training.

Important:
`RealizationPlan` is currently denotational. Backends like TorchTitan and
Megatron use it for validation and lowering, not as an executable collective
program.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TorchTitanProvisioning:
    """TorchTitan-specific provisioning summary derived during lowering."""

    tp: int = 1
    cp: int = 1
    pp: int = 1
    ep: int = 1
    enable_loss_parallel: bool = True
    packed_sequences: bool = True


@dataclass(frozen=True)
class MegatronProvisioning:
    """Megatron-specific provisioning summary derived during lowering."""

    tp: int = 1
    pp: int = 1
    ep: int = 1
    enable_loss_parallel: bool = True
    packed_sequences: bool = True


@dataclass(frozen=True)
class RealizationPlan:
    """Seqax-inspired realization summary.

    The strings here are semantic layout/collective descriptions. Lowering code
    should derive backend provisioning requirements from them rather than using
    backend config as the source of truth.

    This object is not an executable IR. Today it describes intended semantics;
    concrete backends may only support a validated/lowered subset.
    """

    local_layouts: tuple[str, ...] = ()
    collective_transitions: tuple[str, ...] = ()
    packed_sequences: bool = True

    def required_mesh_axes(self) -> tuple[str, ...]:
        axes: list[str] = []
        for expr in (*self.local_layouts, *self.collective_transitions):
            for axis in re.findall(r"/([A-Za-z_][A-Za-z0-9_]*)", expr):
                if axis not in axes:
                    axes.append(axis)
        return tuple(axes)

    def requires_loss_parallel(self) -> bool:
        return any(
            "vocab/tp" in expr or "loss_parallel" in expr for expr in self.collective_transitions
        )


def _required_mesh_axes(realization: RealizationPlan) -> set[str]:
    return set(realization.required_mesh_axes())


def derive_torchtitan_provisioning(
    base: TorchTitanProvisioning,
    realization: RealizationPlan,
) -> TorchTitanProvisioning:
    """Derive TorchTitan provisioning from realization semantics."""

    required_axes = _required_mesh_axes(realization)
    if "d" in required_axes or "dp" in required_axes:
        raise ValueError(
            "TorchTitan lowering does not accept explicit /d or /dp realization intent yet. "
            "Keep data parallelism as a backend-native runtime choice in this path."
        )
    if "tp" in required_axes:
        assert base.tp > 1, "realization requires tp, but TorchTitanProvisioning.tp <= 1"
    if "cp" in required_axes:
        assert base.cp > 1, "realization requires cp, but TorchTitanProvisioning.cp <= 1"
    if "pp" in required_axes:
        assert base.pp > 1, "realization requires pp, but TorchTitanProvisioning.pp <= 1"
    if "ep" in required_axes:
        assert base.ep > 1, "realization requires ep, but TorchTitanProvisioning.ep <= 1"

    return TorchTitanProvisioning(
        tp=base.tp,
        cp=base.cp,
        pp=base.pp,
        ep=base.ep,
        enable_loss_parallel=base.enable_loss_parallel and realization.requires_loss_parallel(),
        packed_sequences=base.packed_sequences and realization.packed_sequences,
    )


def derive_megatron_provisioning(
    base: MegatronProvisioning,
    realization: RealizationPlan,
) -> MegatronProvisioning:
    """Derive Megatron provisioning from realization semantics."""

    required_axes = _required_mesh_axes(realization)
    if "cp" in required_axes:
        raise ValueError(
            "Megatron lowering does not support /cp realization intent in this path yet. "
            "Keep cp=1 and omit /cp from RealizationPlan."
        )
    if "d" in required_axes or "dp" in required_axes:
        raise ValueError(
            "Megatron lowering does not accept explicit /d or /dp realization intent yet. "
            "Megatron derives data parallel replicas from world size and lowered model parallel axes."
        )
    if "tp" in required_axes:
        assert base.tp > 1, "realization requires tp, but MegatronProvisioning.tp <= 1"
    if "pp" in required_axes:
        assert base.pp > 1, "realization requires pp, but MegatronProvisioning.pp <= 1"
    if "ep" in required_axes:
        assert base.ep > 1, "realization requires ep, but MegatronProvisioning.ep <= 1"

    return MegatronProvisioning(
        tp=base.tp,
        pp=base.pp,
        ep=base.ep,
        enable_loss_parallel=base.enable_loss_parallel and realization.requires_loss_parallel(),
        packed_sequences=base.packed_sequences and realization.packed_sequences,
    )


@dataclass(frozen=True)
class TorchTitanLowering:
    """TorchTitan-specific lowering summary."""

    provisioning: TorchTitanProvisioning = TorchTitanProvisioning()
    realization: RealizationPlan = RealizationPlan()

    @staticmethod
    def from_realization(
        provisioning: TorchTitanProvisioning,
        realization: RealizationPlan,
    ) -> "TorchTitanLowering":
        return TorchTitanLowering(
            provisioning=derive_torchtitan_provisioning(provisioning, realization),
            realization=realization,
        )


@dataclass(frozen=True)
class MegatronLowering:
    """Megatron-specific lowering summary.

    This path keeps `RealizationPlan` as the denotational source of truth, then
    derives the coarse partition intent Megatron can honestly consume today.
    It does not mean Megatron executes the explicit collective transitions from
    `realization`.
    """

    provisioning: MegatronProvisioning = MegatronProvisioning()
    realization: RealizationPlan = RealizationPlan()

    @staticmethod
    def from_realization(
        provisioning: MegatronProvisioning,
        realization: RealizationPlan,
    ) -> "MegatronLowering":
        return MegatronLowering(
            provisioning=derive_megatron_provisioning(provisioning, realization),
            realization=realization,
        )


def dense_supervised_realization(
    *,
    tp: int = 1,
    cp: int = 1,
    pp: int = 1,
    packed_sequences: bool = True,
) -> RealizationPlan:
    """Seqax-inspired realization for dense supervised training.

    This is not an executable IR. It is a semantic summary of the layouts and
    collective transitions the lowering is expected to preserve.
    """

    local_layouts = ["batch seq hidden"]
    collective_transitions = ["batch seq hidden -> scalar"]

    if cp > 1:
        local_layouts[0] = "batch seq/cp hidden"
    if tp > 1:
        local_layouts.append("batch seq vocab/tp")
        collective_transitions.insert(0, "batch seq vocab/tp -> batch seq vocab")
    else:
        local_layouts.append("batch seq vocab")
    if pp > 1:
        local_layouts.append("batch stage/pp hidden")

    return RealizationPlan(
        local_layouts=tuple(local_layouts),
        collective_transitions=tuple(collective_transitions),
        packed_sequences=packed_sequences,
    )


def dense_rl_realization(
    *,
    tp: int = 1,
    cp: int = 1,
    pp: int = 1,
    packed_sequences: bool = True,
) -> RealizationPlan:
    """Seqax-inspired realization for dense RL training."""

    local_layouts = ["batch seq hidden", "batch seq value"]
    collective_transitions = [
        "batch seq hidden -> scalar",
        "batch seq value -> scalar",
    ]

    if cp > 1:
        local_layouts[0] = "batch seq/cp hidden"
        local_layouts[1] = "batch seq/cp value"
    if tp > 1:
        local_layouts.append("batch seq vocab/tp")
        collective_transitions.insert(0, "batch seq vocab/tp -> batch seq vocab")
    else:
        local_layouts.append("batch seq vocab")
    if pp > 1:
        local_layouts.append("batch stage/pp hidden")

    return RealizationPlan(
        local_layouts=tuple(local_layouts),
        collective_transitions=tuple(collective_transitions),
        packed_sequences=packed_sequences,
    )
