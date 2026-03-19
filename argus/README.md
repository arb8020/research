# Argus

`argus` is the run supervisor layer for this monorepo.

Its job is to model detached experiment execution honestly:

- a `RunSpec` is user intent
- a `RunRecord` is a durable logical run
- an `AttemptRecord` is one concrete execution of that run
- an `AllocationRef` is the leased compute bound to an attempt
- a `Command` asks the runtime to do something
- an `Event` is an append-only fact
- a `RunSnapshot` is a materialized view derived from events

## Non-goals

`argus` is not:

- a cloud broker
- a workspace deploy tool
- a process transport layer
- a log file server

Those belong in `broker`, `bifrost`, and `miniray`.

## Intended Boundary

The stack should look like:

1. `broker` allocates runtime-ready compute
2. `bifrost` operates on prepared remote workspaces and processes
3. `miniray` provides narrow worker / stream transport primitives
4. `argus` owns run lifecycle, commands, events, and projections

## Current Scope

This initial package provides:

- core supervisor datatypes
- an in-memory event journal
- snapshot materialization from append-only events
- a first-pass control-plane CLI:
  - `python -m argus run ...`
  - `python -m argus monitor ...`

It does not yet include:

- remote supervisor processes
- network protocols
- persistence beyond process memory
- retry policy executors
- provider-aware monitoring or control-plane transport

## Current CLI shape

Today, the Argus CLI is the public control-plane entrypoint:

- `argus run` is implemented by [run.py](/Users/chiraagbalu/research/argus/run.py)
- `argus monitor` is implemented by [monitor.py](/Users/chiraagbalu/research/argus/monitor.py)

This is intentional as a first compression step:

- public control-plane ownership moves to Argus now
- SSH/Modal execution guts remain separate for the moment
- deeper execution/session unification belongs in Broker/Bifrost/Argus follow-up work

## Current monitoring model

The intended split is:

- `argus monitor`
  - resolve run identity
  - attach/sync remote artifacts when needed
  - expose launcher/control-plane-only status such as active launchers
  - hand off run viewing to a viewer
- `rollouts monitor`
  - render the workload-aware TUI over local artifacts

So the local run directory is the seam between the control plane and the UI.

## Running evaluations

`argus run` now accepts both training configs and eval configs.

For eval configs, the current launcher path is:

1. classify the config as `evaluation`
2. allocate a stable local run directory under `rollouts/results/eval/run_<timestamp>/`
3. launch `python -m rollouts.eval.run ... --output-dir <run_dir>` as a detached subprocess
4. optionally hand off to `argus monitor <run_dir>` for live viewing

Examples:

```bash
# Fire-and-forget eval
python -m argus run --config rollouts/configs/prime_ci/reverse_text/eval_api.py

# Launch and watch in the TUI
python -m argus run --config rollouts/configs/prime_ci/reverse_text/eval_api.py --tui

# Launch and tail to stdout
python -m argus run --config rollouts/configs/prime_ci/reverse_text/eval_api.py --tail
```

Current limitation:

- this Argus eval launch path is local orchestration only
- provider-specific runtime lifecycle for evals should still live in the eval workload itself
  - for example, a Modal or RunPod workspace resource inside the eval environment

## Writing an eval config

An eval config must satisfy the `rollouts.config_contracts.validate_eval_config_module(...)` contract.

It must export:

- `tasks` or `tasks_path`
- `run_spec: AgentRunSpec` or `prepare_messages`
- `score_fn` or `sample_scorer`

It may also export:

- `run: EvalRunConfig`
- `output: EvalOutputConfig`
- `hardware: HardwareConfig`
- `server: InferenceServerConfig`

Minimal example:

```python
from rollouts.core import Message, Metric, Score
from rollouts.eval import AgentRunSpec, EndpointConfig
from rollouts.training.scoring import FunctionSampleScorer

tasks = [{"text": "hello"}]

run_spec = AgentRunSpec(
    endpoint=EndpointConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        temperature=0.0,
        max_tokens=128,
    ),
    prepare_messages=lambda sample: [
        Message(role="user", content=f"Reverse this text: {sample['text']}")
    ],
)

def score_attempt(sample):
    expected = sample.input["text"][::-1]
    correct = sample.response.strip() == expected
    return Score(metrics=(Metric("exact_match", 1.0 if correct else 0.0),))

sample_scorer = FunctionSampleScorer(score_attempt)
```

Then run it with:

```bash
python -m argus run --config /absolute/path/to/eval_config.py --tui
```
