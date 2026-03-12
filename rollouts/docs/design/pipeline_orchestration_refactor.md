# Pipeline Orchestration Refactor

**DRI:**
**Claude:** [this conversation]

## Done When

We are done when the repo has:

1. One official eval entrypoint.
2. One official training entrypoint.
3. A small set of explicit stage/artifact boundaries that let us compose:
   - standalone eval runs
   - standalone RL/SFT runs
   - periodic held-out evals during training
   - post-checkpoint evals
4. No duplicate task-specific wrappers/configs left in active use when shared code can express the same thing.

Concrete future flows we want to support:

```python
Pipeline(
    stages=[
        SFTStage(
            name="sft_main",
            every_steps=None,
            outputs=["checkpoint"],
        ),
        EvalStage(
            name="heldout_kernelbench",
            trigger=AfterCheckpoint("sft_main"),
            input="checkpoint",
            evaluator="kernelbench",
        ),
    ]
)
```

```python
Pipeline(
    stages=[
        RLStage(
            name="grpo_main",
            every_steps=100,
            outputs=["checkpoint", "endpoint"],
        ),
        EvalStage(
            name="heldout_eval",
            trigger=EveryKSteps(1000),
            input="checkpoint",
            evaluator="heldout_suite",
        ),
    ]
)
```

## Context

Right now the repo has the right low-level pieces but not a single clean orchestration model.

We already have:

- A shared eval core: [`rollouts/rollouts/eval/native.py`](../../rollouts/eval/native.py)
- A higher-level eval runner: [`rollouts/rollouts/eval_runner.py`](../../rollouts/eval_runner.py)
- A unified training/eval launcher: [`rollouts/rollouts/run.py`](../../rollouts/run.py)
- A generic training loop with explicit cadence boundaries: [`rollouts/rollouts/training/train.py`](../../rollouts/training/train.py)
- Weight-sync boundaries between training and inference: [`rollouts/rollouts/training/weight_sync.py`](../../rollouts/training/weight_sync.py)
- A shared `Sample` type used across training/eval/rollouts: [`rollouts/rollouts/training/types.py`](../../rollouts/training/types.py)

We also have too many entrypoints and wrapper styles:

- `rollouts/evals/run_eval.py`
- `rollouts/rollouts/run_eval.py`
- task-specific eval wrappers under `examples/`
- task-specific training wrappers and config shapes

This is acceptable for experiments but not for the next phase, where we want to express pipelines like:

- "run this held-out eval every k RL steps"
- "run this eval after every checkpoint during SFT"
- "evaluate a live endpoint during training"
- "evaluate a saved checkpoint offline after training"

This is a research codebase. We are allowed to break compatibility in favor of simplicity and flexibility.

## Problem Statement

We currently mix together three different concerns:

1. Leaf execution:
   - run one eval
   - run one training job
2. Task-specific setup:
   - KernelBench sandboxing
   - environment factories
   - dataset loading
3. Cross-stage orchestration:
   - when to checkpoint
   - when to sync weights
   - when to run held-out evals
   - what artifact an eval should consume

The refactor should separate these concerns so we can compose them cleanly.

## Non-Goals

- Backward compatibility with old CLI wrappers or config shapes
- Preserving every current path under `examples/` or `rollouts/rollouts/`
- Building a full workflow engine up front
- Solving distributed scheduling for all future cluster backends in this pass
- Rewriting `run_agent()` or low-level training internals unless needed for the stage boundary

## Core Decision

The shared boundary is not the RL data buffer.

The shared boundary is the artifact contract between stages.

The RL data buffer is a training-side implementation detail. Evals should not depend on it unless an eval explicitly evaluates rollout batches as training data.

### Why

The data buffer in [`rollouts/rollouts/training/datasets/data_buffer.py`](../../rollouts/training/datasets/data_buffer.py) is just one way to iterate over `Sample`s for training. It is not the right unit for orchestration.

The real cross-stage artifacts are things like:

- task collections
- checkpoints
- live endpoints
- rollout/sample batches
- eval reports

This lets us support:

- offline eval from checkpoint
- online eval against live endpoint
- RL with inference weight-sync
- SFT with periodic checkpoint-based eval

without forcing eval code to know about RL internals.

## Target Model

We will standardize on three layers.

### 1. Leaf Executors

These do one thing only.

- `run_eval.py`
  - input: tasks + evaluator config + model source
  - output: eval report
- `run.py`
  - input: training config + model source + rollout source
  - output: checkpoints, optional live endpoint updates, training metrics
- future `run_sft.py` or a generalized training entrypoint
  - input: offline dataset + training config
  - output: checkpoints, metrics

Leaf executors must not encode cross-stage policy like "run held-out eval every k steps".

### 2. Shared Stage Specs

We introduce explicit configs for stage boundaries.

Minimal conceptual model:

```python
@dataclass(frozen=True)
class CheckpointRef:
    path: str
    step: int
    source_stage: str


@dataclass(frozen=True)
class EndpointRef:
    base_url: str
    model: str
    weight_version: int | None
    source_stage: str


@dataclass(frozen=True)
class EvalReportRef:
    output_dir: str
    summary_metrics: dict[str, float]
    source_stage: str
```

And stage shapes like:

```python
class Stage(Protocol):
    name: str


@dataclass(frozen=True)
class EvalStage:
    name: str
    evaluator: str
    input_kind: Literal["checkpoint", "endpoint", "tasks"]
    trigger: TriggerSpec


@dataclass(frozen=True)
class TrainStage:
    name: str
    trainer: Literal["rl", "sft", "pretrain"]
    trigger: TriggerSpec | None = None
```

This does not need to be over-designed. The important part is that stages have explicit input/output kinds.

### 3. Pipeline Coordinator

A separate orchestration layer coordinates stages.

Responsibilities:

- subscribe to train events
- react to checkpoint save / step complete / weight sync events
- launch eval stages at configured cadence
- pass artifact refs into downstream stages
- persist pipeline-level metadata and provenance

Non-responsibilities:

- implementing eval logic
- implementing training logic
- implementing environment logic

## Official Entrypoints

After the refactor, the official CLI surface should be:

- `rollouts/evals/run_eval.py`
- `rollouts/rollouts/run.py`
- future `rollouts/pipeline/run_pipeline.py` or similar

Everything else is either:

- deleted
- converted into thin shims that immediately call the official path
- or moved under the new shared runner shape

We should not keep multiple active entrypoints that do the same thing.

## Config Direction

Configs should move toward a small number of shapes with clear I/O boundaries.

### Eval Config

Eval config should answer:

- what tasks are we evaluating?
- what model source are we evaluating?
- what environment/evaluator do we use?
- where do outputs go?

Model source should be explicit:

- `endpoint`
- `checkpoint`
- `model_id`

instead of being hidden in task-specific wrappers.

### Training Config

Training config should answer:

- what model are we training from?
- what data/rollout source do we consume?
- what backend/inference setup do we use?
- what checkpoint/sync cadence do we use?

### Pipeline Config

Pipeline config should answer:

- which stages exist?
- what artifacts flow between them?
- what triggers downstream stages?

Example:

```python
pipeline = PipelineConfig(
    stages=[
        RLStageConfig(
            name="train",
            config_path="...",
        ),
        EvalStageConfig(
            name="heldout_eval",
            config_path="...",
            input_from="train.checkpoint",
            trigger=EveryKSteps(1000),
        ),
    ]
)
```

## Existing Abstractions We Should Reuse

### Reuse

- `Sample` as the shared per-example artifact shape
- `EvalConfig` / `evaluate()` as the eval execution core
- `training/train.py` cadence boundaries for checkpoint/sync hooks
- weight-sync interfaces for live-endpoint evaluation during RL
- provenance/fingerprint utilities

### Do Not Promote

- training `DataBuffer` as the universal orchestration abstraction
- task-specific `run_eval.py` wrappers
- duplicated ad hoc config loaders

## Refactor Plan

### Phase 1: Entrypoint Cleanup

1. Declare official eval entrypoint.
2. Declare official training entrypoint.
3. Move KernelBench onto the official eval path.
4. Delete duplicate wrappers/configs when shared code can replace them.

Deliverable:

- one recommended way to run evals
- one recommended way to run training

### Phase 2: Artifact and Stage Contracts

1. Add explicit artifact reference types.
2. Add explicit model-source config for evals:
   - checkpoint
   - endpoint
   - direct model/provider
3. Add explicit stage config dataclasses.

Deliverable:

- evals and training can be described as stages with typed inputs/outputs

### Phase 3: Hook Training into Stage Events

1. Expose structured train events at:
   - step complete
   - checkpoint saved
   - weight sync complete
2. Allow downstream orchestration to subscribe.
3. Keep leaf executors simple; do not bury orchestration inside them.

Deliverable:

- "run held-out eval every k steps" can be expressed without forking training code

### Phase 4: Add Pipeline Runner

1. Implement a small coordinator over leaf executors.
2. Accept stage graph / ordered pipeline config.
3. Persist pipeline-level provenance and outputs.

Deliverable:

- single command for multi-stage workflows

## Deletion Policy

This refactor is not complete unless it removes duplication.

Whenever we migrate a task onto a shared entrypoint, we should also:

1. delete the old wrapper, or
2. leave a tiny shim with a clear TODO and no custom logic

The repo should not end this effort with:

- multiple equivalent `run_eval.py` files
- multiple config shapes for the same concept
- copy-pasted provenance/fingerprinting code in task-specific runners

## Risks

### Risk: Overbuilding the pipeline layer

Mitigation:

- keep the first pipeline runner small
- build only around concrete artifacts we already have: checkpoint, endpoint, report

### Risk: Forcing evals to depend on RL concepts

Mitigation:

- keep data buffer / rollout manager as training internals
- stage inputs must be artifact refs, not trainer implementation details

### Risk: Conflating live-endpoint eval and checkpoint eval

Mitigation:

- model source must be explicit in eval config
- provenance must record whether the eval ran against a checkpoint or endpoint

## Open Questions

1. Should eval leaf execution live under `rollouts/evals/` or under `rollouts/rollouts/` as a library-first path with a thin CLI wrapper?
2. Should `run.py` remain RL-specific, or should we rename it to a more general training runner once SFT shares the same shape?
3. Do we want one generic `PipelineStageConfig`, or separate `EvalStageConfig` / `TrainStageConfig` dataclasses for clarity?
4. Should pipeline orchestration be synchronous/local-only at first, or immediately support remote providers?

## Immediate Next Step

Use KernelBench as the first migration target.

That means:

1. move KernelBench onto the centralized eval runner shape
2. preserve the new provenance/fingerprint work in shared code
3. delete the bespoke wrapper path once the centralized version is working
4. use that migration to define the minimum stage/artifact interfaces we need next
