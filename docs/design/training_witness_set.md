# Training witness set

This is the acceptance target for the training refactor.

The point is not to enumerate every future training mode. The point is to prove
that the core training language is real, that the semantic boundaries are
correct, and that the system composes along the axes we care about.

If these witness shapes are clean, the spec is probably real. If these witness
shapes require hacks, hidden state, or backend leakage, the spec is wrong.

## What this witness set is trying to prove

- Objective semantics are explicit.
- Trainable-parameter policy is orthogonal to objective semantics.
- Weight sync and training state are real semantic concepts.
- MoE and low-precision semantics are first-class.
- The core API is not accidentally specialized to dense supervised training.

## Witness set

### 1. Dense supervised, full-weight

This proves:

- basic `TrainingDatum -> ForwardProducts -> StepResult`
- supervised loss semantics
- full-weight update path
- checkpoint and optimizer-state ownership for normal dense training

Ideal usage:

```python
datum = TrainingDatum(
    model_input=ModelInput(tokens=tokens),
    objective_inputs={
        "labels": labels,
        "loss_mask": loss_mask,
    },
)

result_fut = backend.forward_backward(
    datum,
    loss_fn=supervised_loss,
)
step_fut = backend.optim_step()
```

Inner step-language intent:

```python
def supervised_step(state: TrainingState, batch: SFTBatch) -> tuple[TrainingState, StepResult]:
    logits_local = forward_local(
        state.model,
        "batch/d seq -> batch/d seq vocab/tp",
        tokens=batch.tokens,
    )
    logits = materialize(
        "batch/d seq vocab/tp -> batch seq vocab",
        logits_local,
    )
    xent = masked_cross_entropy(logits, batch.labels, batch.loss_mask)
    loss = reduce_mean("batch/d -> scalar", xent)
    backward(loss)
    state, opt_metrics = optimizer_step(state)
    return state, StepResult(
        losses={"total": loss.item(), "xent": loss.item()},
        other_metrics=opt_metrics,
    )
```

### 2. Dense distillation, LoRA-only

This proves:

- teacher-signal path
- multi-loss path
- `ForwardProducts` and `StepResult.losses` are not single-loss special cases
- adapter-only training is orthogonal to objective semantics

Ideal usage:

```python
datum = TrainingDatum(
    model_input=ModelInput(tokens=tokens),
    objective_inputs={
        "teacher_logprobs": teacher_logprobs,
        "labels": labels,
        "loss_mask": loss_mask,
    },
)

result_fut = backend.forward_backward(
    datum,
    loss_fn=distillation_loss,
)
step_fut = backend.optim_step(trainable="lora")
```

Inner step-language intent:

```python
def distill_lora_step(state: TrainingState, batch: DistillBatch) -> tuple[TrainingState, StepResult]:
    logits_local = forward_local(
        state.model,
        "batch/d seq -> batch/d seq vocab/tp",
        tokens=batch.tokens,
    )
    student_logits = materialize(
        "batch/d seq vocab/tp -> batch seq vocab",
        logits_local,
    )
    kl = distillation_kl(student_logits, batch.teacher_logprobs, batch.loss_mask)
    ce = masked_cross_entropy(student_logits, batch.labels, batch.loss_mask)
    total = reduce_mean("batch/d -> scalar", kl + ce)
    backward(total, trainable="lora")
    state, opt_metrics = optimizer_step(state, trainable="lora")
    return state, StepResult(
        losses={
            "total": total.item(),
            "kl": reduce_mean("batch/d -> scalar", kl).item(),
            "ce": reduce_mean("batch/d -> scalar", ce).item(),
        },
        other_metrics=opt_metrics,
    )
```

### 3. Dense RL, full-weight

This proves:

- rollout logprob / advantage semantics
- RL is a real objective family, not a supervised special case
- full-weight optimizer state and weight sync semantics are real
- training state and weight visibility are not accidentally LoRA-shaped
- weight publication cadence/mode are explicit policy, not loop-local modulo logic

We use full-weight here on purpose even if LoRA may be more common in practice.
This is the stronger systems witness.

Ideal usage:

```python
datum = TrainingDatum(
    model_input=ModelInput(tokens=tokens),
    objective_inputs={
        "rollout_logprobs": rollout_logprobs,
        "advantages": advantages,
        "returns": returns,
        "group_ids": group_ids,
    },
)

result_fut = backend.forward_backward(
    datum,
    loss_fn=rl_loss,
)
step_fut = backend.optim_step()
```

Inner step-language intent:

```python
def rl_step(state: TrainingState, batch: RLBatch) -> tuple[TrainingState, StepResult]:
    outputs = forward_local(
        state.model,
        "batch/d seq -> batch/d seq vocab/tp",
        tokens=batch.tokens,
        require=["logits", "values"],
    )
    logits = materialize("batch/d seq vocab/tp -> batch seq vocab", outputs.logits)
    new_logprobs = gather_token_logprobs(logits, batch.tokens)
    policy = policy_objective(
        new_logprobs=new_logprobs,
        old_logprobs=batch.rollout_logprobs,
        advantages=batch.advantages,
        group_ids=batch.group_ids,
    )
    value = value_objective(outputs.values, batch.returns)
    total = reduce_mean("batch/d -> scalar", policy + value)
    backward(total)
    state, opt_metrics = optimizer_step(state)
    return state, StepResult(
        losses={
            "total": total.item(),
            "policy": reduce_mean("batch/d -> scalar", policy).item(),
            "value": reduce_mean("batch/d -> scalar", value).item(),
        },
        other_metrics=opt_metrics,
    )
```

### 4. MoE low-precision supervised, full-weight

This proves:

- MoE is not just a dense-model variant hidden behind a model factory
- precision policy is semantic
- TP and EP pressure the layout language
- the API is not accidentally unaware of routing / dispatch / expert state

We do not require ragged routed-token dispatch to be fully first-class in the
first pass, but this witness should make it obvious where the extension point
belongs.

Ideal usage:

```python
datum = TrainingDatum(
    model_input=ModelInput(tokens=tokens),
    objective_inputs={
        "labels": labels,
        "loss_mask": loss_mask,
    },
)

result_fut = backend.forward_backward(
    datum,
    loss_fn=moe_supervised_loss,
)
step_fut = backend.optim_step()
```

Inner step-language intent:

```python
def moe_supervised_step(
    state: TrainingState,
    batch: SFTBatch,
) -> tuple[TrainingState, StepResult]:
    outputs = forward_local(
        state.model,
        "batch/d seq -> batch/d seq hidden/tp",
        tokens=batch.tokens,
        precision=PrecisionPolicy(
            param="bf16",
            compute="bf16",
            router="fp32",
            expert_compute="lowp",
            reduce="fp32",
        ),
        require=["logits", "router_aux"],
    )
    logits = materialize("batch/d seq vocab/tp -> batch seq vocab", outputs.logits)
    xent = masked_cross_entropy(logits, batch.labels, batch.loss_mask)
    total = reduce_mean("batch/d -> scalar", xent + outputs.router_aux)
    backward(total)
    state, opt_metrics = optimizer_step(state)
    return state, StepResult(
        losses={
            "total": total.item(),
            "xent": reduce_mean("batch/d -> scalar", xent).item(),
            "router_aux": reduce_mean("batch/d -> scalar", outputs.router_aux).item(),
        },
        other_metrics=opt_metrics,
    )
```

## What is intentionally deferred

- Pipeline-parallel schedule semantics as first-class source-level constructs.
- Full ragged routed-token layout algebra.
- Large feature matrices like `RL + LoRA + MoE + lowp`.

Those are composition tests for later. The witness set above is the basis.

## Acceptance standard

The refactor is on track if these witness shapes:

- read cleanly in source
- do not require overloaded data bags
- do not leak backend ontology into the core types
- make objective requirements explicit
- make trainable-parameter policy explicit
- make distributed materialization/reduction points explicit where they matter

After these are clean, we can mix and match.

## Audit: what is still missing

The code now has real witness paths for:

- dense supervised
- dense RL
- dense distillation
- MoE low-precision supervised

But the witness set is still incomplete in several important ways.

### 1. Trainable-parameter policy is not first-class yet

The witness set says:

- dense distillation, LoRA-only

But the current contract-native code does not make trainable-parameter policy a
real semantic object. We still lack an explicit way to say:

- full-weight
- LoRA-only
- router-only
- experts-only

This is the most obvious gap between the written witness set and the actual
 code.

What to add:

- `TrainableParameterPolicy` or equivalent
- explicit use of that policy in at least one witness path
- metrics/events that make it obvious which policy is active

Status:

- Added `TrainableParameterPolicy` to the core training contract.
- Witness loops now set policy explicitly:
  - dense supervised: `full_weight`
  - dense RL: `full_weight`
  - dense distillation: `lora_only`
  - MoE low-precision supervised: `full_weight`
- PyTorch contract path now enforces the active policy by dropping gradients for
  parameters outside the selected family.

### 2. Training state and weight sync are still implicit

The witness set says:

- dense RL, full-weight

This was chosen partly because it should stress:

- optimizer state ownership
- checkpoint semantics
- weight publication/versioning

But current witness code still relies on backend-internal mutable state and the
legacy checkpoint/sync path. We do not yet have a first-class contract for:

- current training step
- weight version
- publication of new weights
- visibility to inference workers

What to add:

- explicit `TrainingState` / `WeightVersion` / `WeightSyncPolicy`
- at least one RL witness path that surfaces weight publication explicitly

Status:

- Added `TrainingRuntimeState`, `WeightVersion`, and `WeightPublication`.
- The RL witness loop now carries explicit runtime state and records weight
  publications when checkpoints are pushed to inference engines.
- Step metrics now surface:
  - `weight_version`
  - `published`
  - `published_weight_version`
  - `published_checkpoint_path`
  - `published_engine_count`

### 3. Distributed realization semantics are not in the executable witness path

The design docs and sketches say we want:

- explicit local compute
- explicit materialization
- explicit reduction
- seqax-like layout/collective semantics

But the code witnesses currently prove only the objective/data contract.
They do not yet prove the inner realization language.

What to add:

- a second witness layer for realization/lowering
- or explicit `RealizationPlan` / layout-transition annotations attached to a
  witness path

Without this, the witness set proves the outer contract but not the distributed
training worldview.

### 4. `ForwardProducts` is still too weak for the hard cases

Right now it has:

- `logits`
- `values`
- `hidden_states`
- `aux`

This is enough for the first pass, but it is still too loose for:

- typed MoE router aux
- routing decisions / keep-routing metadata
- distillation-specific forward products
- future attention/cache or value-head variants

What to add:

- more explicit fields for hard semantic cases
- or typed payload objects inside `aux` with clear ownership

### 5. MoE routing/dispatch is still only hinted at

The MoE witness path now carries:

- `PrecisionPolicy`
- optional `router_aux_loss`

That is good, but it does not yet make routed-token dispatch a first-class
concept.

Missing concepts:

- routing result
- dispatch plan
- capacity policy
- padding policy
- possibly keep-routing metadata for RL

This should remain a targeted MoE extension, not a global tensor concept, but
it is still missing from the witness set.

### 6. The `Sample` split is not yet reflected in the training boundary

The loops now mostly build `TrainingDatum` directly, which is good.
But the upstream pipeline still uses over-shared shapes like:

- `TrainingSample`
- `AttemptRow`
- `RolloutBatch`

We have not yet defined the clean phase-specific objects that should exist
before a `TrainingDatum` is built.

What to add:

- explicit pre-training/training/scoring/rollout phase objects
- one-way conversions at boundaries instead of bridge objects with many
  convenience accessors

## Proposed next witness extensions

The next additions to the witness set should be:

1. Add `TrainableParameterPolicy` to every witness shape.
2. Add an explicit RL weight-publication/version witness.
3. Add a realization/lowering witness so the seqax-style inner language is
   tested, not just described.
4. Add a MoE routing/dispatch witness object, even if it starts minimal.
