"""Lowering-oriented training objects.

These summarize backend provisioning choices derived from higher-level
realization semantics. They are not the semantic center of training.

Important:
`RealizationPlan` is currently denotational. Backends like TorchTitan use it
for validation and lowering, not as an executable collective program.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelIntent:
    """Backend-neutral summary of parallel provisioning needs."""

    dp: int = 1
    tp: int = 1
    cp: int = 1
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


def derive_parallel_intent(base: ParallelIntent, realization: RealizationPlan) -> ParallelIntent:
    """Derive and validate lowering intent from realization semantics.

    The realization layer does not choose device counts, but it does say which
    mesh axes are semantically required. This function validates that the
    lowering intent provisions those axes and derives secondary flags like loss
    parallelism and sequence packing.
    """

    required_axes = set(realization.required_mesh_axes())
    if "tp" in required_axes:
        assert base.tp > 1, "realization requires tp, but ParallelIntent.tp <= 1"
    if "cp" in required_axes:
        assert base.cp > 1, "realization requires cp, but ParallelIntent.cp <= 1"
    if "pp" in required_axes:
        assert base.pp > 1, "realization requires pp, but ParallelIntent.pp <= 1"
    if "ep" in required_axes:
        assert base.ep > 1, "realization requires ep, but ParallelIntent.ep <= 1"
    if "dp" in required_axes or "d" in required_axes:
        assert base.dp > 1, "realization requires dp, but ParallelIntent.dp <= 1"

    return ParallelIntent(
        dp=base.dp,
        tp=base.tp,
        cp=base.cp,
        pp=base.pp,
        ep=base.ep,
        enable_loss_parallel=base.enable_loss_parallel and realization.requires_loss_parallel(),
        packed_sequences=base.packed_sequences and realization.packed_sequences,
    )


@dataclass(frozen=True)
class TorchTitanLowering:
    """TorchTitan-specific lowering summary."""

    parallel: ParallelIntent = ParallelIntent()
    realization: RealizationPlan = RealizationPlan()

    @staticmethod
    def from_realization(
        parallel: ParallelIntent,
        realization: RealizationPlan,
    ) -> "TorchTitanLowering":
        return TorchTitanLowering(
            parallel=derive_parallel_intent(parallel, realization),
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
