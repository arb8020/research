# TorchTitan as the first lowering target

TorchTitan is the first lowering witness, not the source of truth.

The core model remains:

- `TrainingDatum`
- `ForwardProducts`
- `StepResult`
- `TrainableParameterPolicy`
- `PrecisionPolicy`
- explicit witness loops for supervised / RL / distillation / MoE

TorchTitan is allowed to be the first backend that realizes these semantics in a
distributed system. It is not allowed to define the ontology.

## Why TorchTitan first

- It is closer to PyTorch-native semantics than Megatron.
- It is a better fit for an eventual named-axis / DTensor / mesh-aware
  realization language.
- It puts pressure on parallelism and precision without immediately forcing the
  core model to speak process-group algebra.

## What the current backend proves

The current TorchTitan backend now accepts the contract-native training path:

- `forward_backward(datum: TrainingDatum, loss_fn: LossFnLike) -> StepResult`

This proves that TorchTitan can at least serve as an outer lowering target for
the current witness loops.

## What is still missing

The current TorchTitan backend is still too operational and too backend-shaped.
In particular:

- it still builds a fake `_Cfg` object to mimic JobConfig
- TP / CP / PP are largely configuration surfaces, not semantic lowerings
- the realization language is still implicit
- there is no explicit bridge from our internal realization semantics to
  `ParallelDims`, mesh placement, or loss-parallel behavior

TODO:

- TorchTitan does not currently expose enough explicit control to directly
  execute the realization language we want.
- For now, treat TorchTitan as a lowering target that can validate and realize a
  derived `ParallelIntent`, not as the interpreter of the full seqax-style IR.
- Keep pushing the core model through dense witness paths first; only deepen the
  realization language once the end-to-end TorchTitan run is clean.

## Minimal lowering object we need

Before Megatron, we should introduce a small lowering object for TorchTitan.

This object should answer:

- what logical axes are active?
- what parallel intent is requested?
- what realization assumptions are required for this witness path?
- what precision policy should be lowered into TorchTitan job config?
- what trainable-parameter policy is active?

It should not answer:

- exact process-group initialization details
- raw mesh/rank arithmetic in the core model
- backend-specific checkpoint format choices

Status:

- Added backend-neutral [ParallelIntent](/Users/chiraagbalu/research/rollouts/rollouts/training/lowering.py).
- Added seqax-inspired [RealizationPlan](/Users/chiraagbalu/research/rollouts/rollouts/training/lowering.py).
- Added TorchTitan-specific [TorchTitanLowering](/Users/chiraagbalu/research/rollouts/rollouts/training/lowering.py).
- The TorchTitan backend now reads parallel provisioning from that lowering
  object instead of treating backend config as the primary source of truth.
- `TrainerConfig` can now optionally carry explicit TorchTitan realization
  strings; the SFT and RL TorchTitan entry points convert those into a
  `RealizationPlan` and pass them into the factory.

## Likely shape

Something like:

```python
@dataclass(frozen=True)
class RealizationPlan:
    local_layouts: tuple[str, ...]
    collective_transitions: tuple[str, ...]
    packed_sequences: bool = True


@dataclass(frozen=True)
class ParallelIntent:
    dp: int = 1
    tp: int = 1
    cp: int = 1
    pp: int = 1
    ep: int = 1
    enable_loss_parallel: bool = True
    packed_sequences: bool = True


@dataclass(frozen=True)
class TorchTitanLowering:
    parallel: ParallelIntent
    realization: RealizationPlan
```

That is not final, but it is the right category of object.

Concrete pattern:

```python
realization = RealizationPlan(
    local_layouts=(
        "batch/d seq hidden/tp",
        "batch/d seq vocab/tp",
    ),
    collective_transitions=(
        "batch/d seq vocab/tp -> batch seq vocab",
    ),
    packed_sequences=True,
)

parallel = ParallelIntent(
    dp=2,
    tp=4,
    ep=1,
)

lowering = TorchTitanLowering.from_realization(parallel, realization)
```

The important ordering is:

1. express realization semantics
2. choose backend provisioning counts
3. derive/validate lowering
4. feed the backend

## What to do next

1. Keep `RealizationPlan` as the semantic source for layout/collective intent.
2. Keep `ParallelIntent` backend-neutral and derived from realization semantics.
3. Lower that into TorchTitan `ParallelDims`.
4. Keep sequence packing / loss-parallel choices explicit in the
   lowering object rather than scattering them through backend code.
5. Only after that, start pushing on the richer seqax-style realization language.

## Acceptance standard

TorchTitan is a good first lowering if:

- the witness loops stay core-shaped
- the backend code gets uglier, not the core model
- parallelism and precision live in a small lowering object
- the next backend (Megatron) would have to implement the same lowering
  interface rather than invent a new ontology
