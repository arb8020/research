# Async and pipeline RL semantics

This is the minimal semantic model we want before hardening more runtime code.

The goal is not to fully design a pipeline manager. The goal is to make the
important async RL distinctions explicit:

- when rollouts are considered stale
- when a newly trained weight version becomes visible
- whether new admissions are allowed during sync
- what runtime state the system is actually in
- how batches are tagged with the version that produced them
- what service transition is happening between training and inference

If these are not explicit, sync mode and async mode will drift into ad hoc
control flow instead of remaining one coherent training model.

## Core objects

### `StalenessPolicy`

This answers:

- how old may a rollout batch be relative to the current training version?
- should stale batches be dropped, blocked, or tolerated?

Likely shape:

```python
@dataclass(frozen=True)
class StalenessPolicy:
    max_version_lag: int = 0
    drop_stale: bool = True
    require_exact_version: bool = False
```

Interpretation:

- `max_version_lag=0` means exact-version or synchronous semantics
- `max_version_lag=1` means one-version stale batches are allowed
- `drop_stale=True` means stale batches are discarded rather than trained on
- `require_exact_version=True` is the strongest setting and should dominate

This is the semantic object behind phrases like:

- sync training
- bounded staleness
- off-policy tolerance

### `WeightVisibilityPolicy`

This answers:

- when does a trained version become eligible for inference?
- does visibility happen atomically?
- do we drain in-flight work before a new version is served?

Likely shape:

```python
@dataclass(frozen=True)
class WeightVisibilityPolicy:
    publish_mode: str = "checkpoint"
    atomic_visibility: bool = True
    drain_before_publish: bool = True
```

Interpretation:

- `publish_mode` might later be `checkpoint`, `filesystem`, `nccl`, `in_memory`
- `atomic_visibility=True` means inference should not see half-applied versions
- `drain_before_publish=True` means no new admissions until the serving side is
  safe to flip to the new version

This is the semantic object behind:

- checkpoint publication
- weight sync
- version visibility

This object is intentionally semantic, not transport-specific. It answers when
a version becomes visible, not whether visibility is realized by:

- filesystem checkpoint publication
- custom HTTP control plane + NCCL data plane
- direct in-memory process-local update

### `AdmissionPolicy`

This answers:

- can new rollouts start while sync is happening?
- are in-flight batches allowed to finish?
- what is the admission behavior around sync boundaries?

Likely shape:

```python
@dataclass(frozen=True)
class AdmissionPolicy:
    pause_on_sync: bool = True
    allow_inflight_completion: bool = True
    max_inflight_batches: int | None = None
```

Interpretation:

- `pause_on_sync=True` gives a clean drain-and-publish default
- `allow_inflight_completion=True` means already-started work may complete
- `max_inflight_batches` is an optional implementation guardrail, not the
  primary async semantic model

This is the semantic object behind:

- pause admissions
- drain before sync
- admission behavior at sync boundaries

### `OverloadPolicy`

This answers:

- what safety valves are allowed when stream-style production outruns training?
- what should happen under sustained pressure?

Likely shape:

```python
@dataclass(frozen=True)
class OverloadPolicy:
    cancel_stale_inflight: bool = False
    spill_to_disk: bool = False
    block_generation_as_last_resort: bool = False
    queue_pressure_threshold: int | None = None
```

Interpretation:

- stream-style backlog remains the default semantic model
- overload response is a separate explicit policy surface
- queue thresholds are guardrails, not the main async ontology

### `PipelineRuntimeState`

This answers:

- what versions are training and serving right now?
- is sync in progress?
- are admissions paused?
- how much work is currently in flight?

Likely shape:

```python
@dataclass(frozen=True)
class PipelineRuntimeState:
    current_train_version: int
    current_serving_version: int
    sync_in_progress: bool = False
    admissions_paused: bool = False
    inflight_batches: int = 0
```

This is the explicit runtime state machine surface. It should be observable and
loggable, not reconstructed from scattered booleans.

For simple modes, that coarse state is enough. For live train/infer systems, we
also need the service-transition state between the trainer and inference
service.

### `UpdateChannelState`

This answers:

- has the train/infer update channel been initialized?
- is inference still serving, or has it quiesced for update?
- is a new version actively being published right now?
- has serving resumed on the new version?

Likely shape:

```python
@dataclass(frozen=True)
class UpdateChannelState:
    channel_ready: bool = False
    quiescing_for_update: bool = False
    update_in_progress: bool = False
    last_published_version: int | None = None
    serving_resumed: bool = False
```

Interpretation:

- `channel_ready=True` means the runtime update path exists and can be used
- `quiescing_for_update=True` means serving is draining or pausing admissions
- `update_in_progress=True` means trainer->inference publication is active
- `serving_resumed=True` means inference has returned to normal serving on the
  new version

Without this, "sync in progress" is too coarse to explain failures in systems
that rely on explicit quiesce/update/resume transitions.

### `VersionedRolloutBatch`

This answers:

- which weight version produced this batch?
- how old is it relative to the current train version?
- when was it created?

Likely shape:

```python
@dataclass(frozen=True)
class VersionedRolloutBatch:
    batch: RolloutBatch
    weight_version: int
    created_at_step: int
    version_lag: int = 0
```

This is the semantic object that makes stale-sample handling honest.

Without this, async RL becomes “some queue somewhere” instead of a versioned
training system.

## One model, three modes

We do not want separate ontologies for sync, async, and true pipeline. We want
one model with different policy settings.

### Synchronous mode

Strongest safety, simplest reasoning.

Example settings:

```python
StalenessPolicy(
    max_version_lag=0,
    drop_stale=True,
    require_exact_version=True,
)

WeightVisibilityPolicy(
    publish_mode="checkpoint",
    atomic_visibility=True,
    drain_before_publish=True,
)

AdmissionPolicy(
    pause_on_sync=True,
    allow_inflight_completion=True,
    max_inflight_batches=1,
)
```

Interpretation:

- only exact-version batches are trained on
- no new admissions during publish
- the system drains to a clean boundary before serving a new version
- the update channel behaves like:
  - `channel_ready -> quiescing_for_update -> update_in_progress ->
    serving_resumed`

This should be the default worldview for the first real dense RL run.

### Async mode

Bounded overlap, but still conservative.

Example settings:

```python
StalenessPolicy(
    max_version_lag=1,
    drop_stale=True,
    require_exact_version=False,
)

WeightVisibilityPolicy(
    publish_mode="checkpoint",
    atomic_visibility=True,
    drain_before_publish=False,
)

AdmissionPolicy(
    pause_on_sync=False,
    allow_inflight_completion=True,
    max_inflight_batches=None,
)

OverloadPolicy(
    cancel_stale_inflight=False,
    spill_to_disk=False,
    block_generation_as_last_resort=False,
    queue_pressure_threshold=None,
)
```

Interpretation:

- one-version stale batches may be tolerated
- already-started work can continue during publish
- new admissions may continue under stream-style production
- overload handling remains a separate explicit policy surface
- service transitions still exist, but may not fully drain before visibility
  changes

This is the natural next step after a clean sync run.

### True pipeline mode

Most aggressive overlap. Highest semantic bar.

Example settings:

```python
StalenessPolicy(
    max_version_lag=2,
    drop_stale=True,
    require_exact_version=False,
)

WeightVisibilityPolicy(
    publish_mode="nccl",
    atomic_visibility=True,
    drain_before_publish=False,
)

AdmissionPolicy(
    pause_on_sync=False,
    allow_inflight_completion=True,
    max_inflight_batches=None,
)

OverloadPolicy(
    cancel_stale_inflight=True,
    spill_to_disk=False,
    block_generation_as_last_resort=True,
    queue_pressure_threshold=None,
)
```

Interpretation:

- multiple versions may coexist transiently
- serving and training overlap more aggressively
- the runtime must track sync state and version lag explicitly
- the update channel is a first-class long-lived runtime subsystem, not an ad
  hoc helper call

If this mode needs different concepts entirely, the model is wrong.

## What this means for the code

Before pushing harder on async/pipeline runtime machinery, the code should be
able to name and log:

- current training version
- current serving version
- whether sync is in progress
- whether admissions are paused
- each batch’s producing version
- each batch’s version lag relative to training
- what policy caused a batch to be accepted or dropped
- the current update-channel transition:
  - ready
  - quiescing
  - publishing
  - resumed

This is the minimum semantic bar.

## Immediate next implementation target

For the first real RL run, we do not need all of this operationalized. We do
want the objects in place so the runtime has somewhere honest to grow.

Recommended order:

1. synchronous dense RL run with explicit policies
2. version-tagged rollout batches
3. runtime state observability
4. explicit update-channel state observability
5. bounded async mode
6. true pipeline only after the above are clean

The next honest runtime boundary is:

1. semantic policy objects
2. lowering into a concrete train/infer sync realization
3. a long-lived update channel that owns:
   - initialize
   - publish
   - close

That channel should sit between:

- a `TrainingBackend` that knows how to prepare/send updates
- an `InferenceBackend` that knows how to quiesce/receive/resume

This keeps:

- policy semantics
- transport/runtime realization
- service lifecycle

separate and observable.
