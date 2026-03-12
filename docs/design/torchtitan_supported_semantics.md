# TorchTitan Supported Semantics

This document states the currently supported TorchTitan path in our vocabulary.

It exists to avoid lying to users about what our seqax-inspired semantics mean
today.

## The distinction we need to preserve

There are three different layers here:

1. `TrainingDatum` / `ForwardProducts` / `StepResult`
   - the objective-facing training contract
2. `RealizationPlan`
   - intended layout and collective semantics
   - seqax-inspired, denotational, not executable by itself
3. `TorchTitanLowering`
   - the backend-specific supported realization path

TorchTitan currently implements `3`, not `2`.

That means:

- users may specify intended layout / collective semantics through
  `RealizationPlan`
- TorchTitan may validate those semantics and derive provisioning requirements
- TorchTitan does **not** currently interpret `RealizationPlan` as an
  executable collective program

## What TorchTitan actually realizes today

The currently supported TorchTitan path is:

- build a registered TorchTitan model package
- provision a `DeviceMesh` through `ParallelDims`
- apply model-specific parallelization functions
- rely on PyTorch-native distributed mechanisms:
  - `DTensor`
  - `fully_shard(...)`
  - `parallelize_module(...)`
  - model-specific TP / EP / PP wiring

So the real realized semantics today are:

- mesh-based provisioning
- model-specific rewrites
- backend-native distributed execution

not:

- direct execution of user-authored collective operations

## Supported subset in our language

### Supported

- contract-native training step boundary
  - `TrainingDatum`
  - `ForwardProducts`
  - `StepResult`
- trainable parameter policy
  - full-weight
  - LoRA-only
  - router-only / experts-only policy filtering in our backend glue
- precision policy as configuration intent
- parallel provisioning summary via `ParallelIntent`
- validation of required mesh axes from `RealizationPlan`
- loss-parallel intent derived from `RealizationPlan`
- packed-sequence intent derived from `RealizationPlan`

### Validated / lowered only

- seqax-style local layout strings
- seqax-style collective transition strings
- explicit layout/collective intent in `RealizationPlan`

These are used to:

- validate that required mesh axes are provisioned
- derive `ParallelIntent`
- drive backend configuration choices

They are **not** executed as explicit collective steps by TorchTitan.

### Not implemented yet

- executable `materialize(...)` / `reduce(...)` / `dispatch(...)` IR
- direct interpretation of explicit collectives on TorchTitan
- our explicit MoE routed-dispatch semantics
- a generic backend that natively executes our seqax-inspired collective program

## Concrete examples

### Example: dense supervised vocab materialization

If the user writes a realization intent like:

```python
RealizationPlan(
    local_layouts=("batch seq vocab/tp",),
    collective_transitions=("batch seq vocab/tp -> batch seq vocab",),
)
```

then TorchTitan currently treats that as:

- validation that `tp > 1` is required
- derived `enable_loss_parallel=True`
- lower into TorchTitan `ParallelDims` / model-specific parallelization

It does **not** mean:

- our code will emit and interpret an explicit `all_gather("tp")` operation

### Example: expert parallel dispatch

TorchTitan does have real expert-parallel dispatch behavior, including
all-to-all, permutation, and combine steps. But that behavior is currently:

- backend-native
- model-specific
- expressed through TorchTitan / PyTorch distributed code

not:

- represented in our explicit realization language yet

## User-facing contract

The honest thing to say today is:

- `RealizationPlan` is a specification of intended layout and collective
  semantics
- TorchTitan supports a constrained realization of that specification
- the current backend uses `RealizationPlan` for validation and lowering, not
  as an executable collective program

## Future path

If we want explicit collectives to actually run in our semantics, we need a
separate executable layer such as:

- `CollectiveProgram`
- `MaterializeOp`
- `ReduceOp`
- `DispatchOp`
- `CombineOp`

Then:

- a reference backend can interpret that directly
- TorchTitan can implement a supported subset if it becomes possible
- a future custom backend can make those semantics native

That future path does not exist yet. The current supported TorchTitan path is
the constrained lowering described above.
