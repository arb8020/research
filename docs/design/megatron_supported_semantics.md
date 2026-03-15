# Megatron Supported Semantics

This document states the currently supported Megatron path in our vocabulary.

It exists to avoid lying about what our seqax-inspired semantics mean today.

## The distinction we need to preserve

There are three different layers here:

1. `TrainingDatum` / `ForwardProducts` / `StepResult`
   - the objective-facing training contract
2. `RealizationPlan`
   - intended layout and collective semantics
   - seqax-inspired, denotational, not executable by itself
3. `MegatronLowering`
   - the backend-specific supported realization path

Megatron currently implements `3`, not `2`.

That means:

- users may specify intended layout / collective semantics through
  `RealizationPlan`
- Megatron may validate those semantics and derive coarse partition intent
- Megatron does **not** currently interpret `RealizationPlan` as an executable
  collective program

## What Megatron actually realizes today

The currently supported Megatron path is:

- derive a validated `MegatronLowering` from `RealizationPlan`
- initialize Megatron process groups from the lowered TP / PP / EP partition intent
- build Megatron-native wrapped models and optimizers
- rely on Megatron-Core distributed execution

So the real realized semantics today are:

- coarse partition provisioning
- Megatron-native process-group setup
- Megatron-native distributed execution

not:

- direct execution of user-authored collective operations

## Supported subset in our language

### Supported

- contract-native training step boundary
- backend-specific Megatron provisioning summary via `MegatronProvisioning`
- validation of required TP / PP / EP axes from `RealizationPlan`
- loss-parallel intent derived from `RealizationPlan`
- packed-sequence intent derived from `RealizationPlan`

### Validated / lowered only

- seqax-style local layout strings
- seqax-style collective transition strings
- explicit layout/collective intent in `RealizationPlan`

These are used to:

- validate that required lowered axes are provisioned
- derive coarse partition intent for Megatron setup
- reject unsupported realization requirements such as `/cp` in this path

They are **not** executed as explicit collective steps by Megatron.

### Backend-native only

- `sequence_parallel`

This remains a Megatron runtime flag. The current `RealizationPlan` vocabulary
does not model it honestly yet.

### Not implemented yet

- executable `materialize(...)` / `reduce(...)` / `dispatch(...)` IR
- direct interpretation of explicit collectives on Megatron
- honest `/cp` realization support in this path

## User-facing contract

The honest thing to say today is:

- `RealizationPlan` is the denotational source of layout and collective intent
- Megatron supports a constrained lowering of that intent
- the current backend uses `RealizationPlan` for validation and lowering, not
  as an executable collective program
