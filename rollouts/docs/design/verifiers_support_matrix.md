# Verifiers Support Matrix

Repo inspected: `PrimeIntellect-ai/verifiers` at commit `3a421c03`.

For the side-by-side comparison against local `rollouts`, see `docs/design/verifiers_vs_rollouts.md`.

## Core read

`verifiers` is best modeled as:

- dataset
- rollout state machine
- rubric

It does not have a separate first-class harness abstraction. The harness mostly lives inside an environment subclass via:

- `setup_state`
- `get_prompt_messages`
- `env_response`
- `update_tool_args`
- cleanup / teardown hooks

The main boundaries are:

- `Environment` in `/tmp/verifiers/verifiers/envs/environment.py`
- `MultiTurnEnv` in `/tmp/verifiers/verifiers/envs/multiturn_env.py`
- `Rubric` in `/tmp/verifiers/verifiers/rubrics/rubric.py`

This means it supports many niche cases, but often via subclassing and `state` mutation rather than via explicit compositional interfaces.

## Support by pattern

### Context management

Support level: strong, but ad hoc.

Built-in prompt/system/few-shot handling exists in `Environment`. Multi-turn context accumulation exists in `MultiTurnEnv.get_prompt_messages`. If you want unusual context semantics, the intended move is to override `get_prompt_messages`, `render_completion`, or mutate `state`.

What exists:

- system prompt / few-shot / prompt normalization
- per-turn prompt reconstruction
- custom completion rendering

What does not exist:

- a typed context object
- a dedicated context manager layer
- middleware-like prompt transforms as a first-class interface

Conclusion: supported, but not factored cleanly.

### User sims

Support level: yes.

`MultiTurnEnv.env_response()` is the user/environment-simulation hook. The environment can emit follow-up messages after each model action. `DoubleCheckEnv` is the simplest example. `TextArenaEnv` is a fuller interactive simulation.

Conclusion: this is a real extension point, even if it is just "subclass and implement env_response".

### Native tool parsing

Support level: yes for provider-native structured tool calls; no for text-emitted pseudo-tool syntax.

`ToolEnv` assumes the model response already contains structured `tool_calls`. Client adapters convert provider-native tool schemas into the internal `Tool` and `ToolCall` types and back.

What exists:

- provider-agnostic `Tool` and `ToolCall`
- provider adapters that map to native tool APIs
- runtime tool execution over parsed JSON args

What does not exist:

- fallback parsing for XML / markdown / plain-text tool syntax
- a pluggable "tool-call parser" separate from model client behavior

Conclusion: if your model emits native tool calls, good. If it emits text that should be interpreted as tool calls, you have to build that layer yourself.

### Harness-in-sandbox

Support level: strong.

`SandboxEnv` creates a persistent sandbox per rollout and exposes a hidden-arg `bash` tool. `CliAgentEnv` goes further and runs a full agent process inside the sandbox while intercepting its model API calls. `OpenEnvEnv` also runs through Prime Sandboxes.

Conclusion: this is one of the framework's strongest areas.

### Harness-outside-of-sandbox

Support level: yes.

Plain `MultiTurnEnv`, `ToolEnv`, and `StatefulToolEnv` all run in-process without container isolation. If you want worker isolation without sandboxing, the env server / worker model provides that.

Conclusion: supported, though not separated into a named harness kind.

### No sandbox at all

Support level: yes, default.

This is the normal path for the basic environment types. Sandboxing is additive, not mandatory.

### Groupwise rewards

Support level: first-class.

`Rubric` detects group reward functions from plural argument names such as `completions`, `prompts`, `answers`, `infos`, or from a `list[...]` return shape. Group rewards run inside `score_group`, and grouped generation is built around `example_id`.

Conclusion: genuinely supported, not a hack.

### Intermediate rewards

Support level: partial.

There is an explicit `TrajectoryStep` type with `reward` and `advantage` slots. `MultiTurnEnv.add_trajectory_step()` is the intended override point. `OpenEnvEnv` writes step rewards into the active trajectory step.

What exists:

- per-step reward fields
- per-step advantage fields
- the ability to mutate them during rollout

What does not exist:

- a more principled stepwise credit assignment interface
- explicit framework semantics for how intermediate rewards should aggregate

Conclusion: possible and partially intended, but still mostly manual.

### Multiple environments

Support level: yes, in two senses.

`EnvGroup` is a first-class mixture of environments keyed by task. Multi-environment eval configs also exist in the eval CLI / TOML flow.

What this does support:

- one dataset composed from multiple environments
- routing rollouts and scoring by task
- multi-environment benchmark runs

What it does not directly support:

- one rollout that simultaneously interacts with multiple environments as peers

Conclusion: heterogeneous mixtures are supported; multi-env interaction inside one rollout is still custom logic.

### Resource management

Support level: moderate.

There are explicit cleanup and teardown hooks, stop-condition-triggered cleanup, sandbox lifecycle management, env-server worker management, and executor autoscaling.

What exists:

- `@vf.cleanup`
- `@vf.teardown`
- per-rollout cleanup
- process / worker lifecycle management
- sandbox bulk teardown

What is missing:

- a unified resource ownership model
- a single abstraction that says who owns which external effects

Conclusion: operational support is decent, but the abstraction boundary is smeared across env subclasses.

### Custom metrics / error handling

Support level: yes, with one important caveat.

Metrics are just reward functions with zero weight, plus monitor rubrics. Aggregate eval metrics are computed incrementally from rollout outputs. Tool error handling is configurable through `stop_errors` and `error_formatter`. Error chains are preserved for reporting.

The caveat:

`Rubric` catches reward-function exceptions and turns them into `0.0`. That is operationally convenient, but semantically weak if you want scoring failures to crash loudly.

Conclusion: custom metrics are well supported; error handling is flexible but sometimes too forgiving.

### Offline evals

Support level: partial.

Saved outputs, metadata, checkpointing, resume, and output export are all solid. There is clear support for resuming interrupted evaluations from `results.jsonl` plus `metadata.json`.

What exists:

- incremental save
- resume from partial runs
- loading saved rollout outputs
- aggregate metrics over saved outputs
- dataset export

What I do not see:

- a first-class "rescore this offline completions dataset without rerunning rollouts" API
- a clear completion-dataset evaluation path wired into `evaluate()` / `generate()`

There is an `_format_completion_dataset()` hook, but I do not see it used by the main eval path.

Conclusion: offline result persistence is strong; offline rescoring is not obviously first-class.

## Overall judgment

The important distinction is:

- many things are possible
- fewer things are modeled explicitly

`verifiers` is more expressive than frameworks that hard-code one agent loop, one sandbox assumption, one tool protocol, or one reward shape. But a lot of that expressiveness comes from "subclass the environment and mutate `state`" rather than from cleanly separated interfaces.

So the honest summary is:

- context management: yes, but ad hoc
- user sims: yes
- native tool parsing: yes for native structured tools only
- harness-in-sandbox: yes
- harness-outside-of-sandbox: yes
- no sandbox: yes
- groupwise rewards: yes, first-class
- intermediate rewards: partial
- multiple environments: yes
- resource management: moderate
- custom metrics / error handling: yes, with caveats
- offline evals: partial

## Main caveats

The places where the design still feels fundamentally preclusive are:

1. Harness is not a first-class algebraic boundary.
2. Tool parsing assumes provider-native structured tool calls.
3. Intermediate rewards exist, but mostly as mutable trajectory bookkeeping.
4. Offline rescoring does not look like a fully surfaced workflow.
5. `state` is the escape hatch for nearly everything, which buys flexibility at the cost of local reasoning.

That is the real tradeoff in `verifiers`: it avoids some of the common over-commitments in RL framework design, but it pays for that by pushing a lot of semantics into subclass code and shared mutable state.
