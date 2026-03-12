# Argus / Rollouts event boundary

This is the ownership split we want between the control plane and the
workload package.

## Core rule

- `rollouts` owns workload semantics
- `argus` owns run identity, event transport, and log/journal storage

In practice:

- `rollouts` defines preflights
- `rollouts` defines stage names
- `rollouts` defines what those stages mean
- `rollouts` decides what invariants make a run valid

- `argus` launches runs
- `argus` records events
- `argus` stores and streams logs/journals
- `argus` provides generic monitoring surfaces over those events

## What Argus should know

Argus should know only generic event structure, e.g.:

- `ts`
- `run_id`
- `attempt_id`
- `source`
- `kind`
- `payload`

Argus may index generic fields and expose generic projections like:

- active launches
- recent events
- latest event
- failed/completed attempts

But Argus should not define the workload ontology.

## What Rollouts should own

Examples of workload-owned semantics:

- `TRAIN_BACKEND_IMPORT_OK`
- `FIRST_FORWARD_OK`
- `FIRST_ALLREDUCE_OK`
- `WEIGHT_PUBLISHED`
- `STALENESS_POLICY_APPLIED`

Argus may record those strings, but it should treat them as opaque workload
payload.

## Why this split matters

If Argus starts owning workload stages:

- the control plane becomes training-specific
- different workloads become harder to support cleanly
- event transport gets mixed with event meaning

If Rollouts owns semantics and Argus owns the journal:

- workload evolution stays local
- the control plane stays generic
- monitoring can still render useful views without defining the ontology

## Practical rule for implementation

When adding new observability:

- if the question is "what happened to the run?", it probably belongs in `argus`
- if the question is "what does this stage/check/invariant mean?", it belongs in `rollouts`

Argus should carry the facts.
Rollouts should define the meaning.
