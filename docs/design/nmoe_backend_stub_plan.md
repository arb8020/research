# `nmoe` Backend Status

`backend="nmoe"` is currently a reserved name, not a runnable backend.

The old `NmoeTrainingBackend` was removed because it was not an `nmoe` runtime
adapter. It loaded Hugging Face models and wrapped the generic PyTorch backend
with an optimizer recipe inspired by `nmoe`. That was useful as an experiment,
but dishonest as a backend identity.

## Honest current state

What exists now:

- `NmoeTrainingBackend`: fail-loud placeholder
- `NmoeConfig`: reserved placeholder config
- `NmoeLowering`: reserved runtime-lowering shape
- `NmoeModelLowering`: reserved model-lowering shape

What does not exist yet:

- native `Transformer(cfg)` model construction
- native `nmoe` checkpoint import/export
- native optimizer/runtime state integration
- lockstep rollout/eval integration
- native `nmoe` observability and weight publication path

## Intended real path

The future native backend should be built around three honest layers:

1. `NmoeModelLowering`
- lowers internal model denotation into native `nmoe` construction intent
- names checkpoint/loader semantics honestly

2. `NmoeLowering`
- lowers `RealizationPlan` into `nmoe` runtime constraints
- expresses expert ownership and RDEP-style dispatch assumptions
- records lockstep eval/generation requirements

3. `NmoeTrainingBackend`
- adapts to actual `nmoe` runtime state
- owns checkpointing, optimizer state, metrics, and weight publication through
  native `nmoe` semantics

## Minimum acceptance bar

We should not call the backend implemented until it can do all of:

1. Build or import a native `nmoe` model honestly.
2. Run training with native optimizer/runtime/checkpoint semantics.
3. Run distributed eval and generation in lockstep.
4. Expose router-health and systems observability.
5. Publish weights through an honest train-to-infer boundary.
