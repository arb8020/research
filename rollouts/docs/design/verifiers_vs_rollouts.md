# Verifiers Vs Rollouts

Repos inspected:

- `PrimeIntellect-ai/verifiers` at commit `3a421c03`
- local `rollouts` at commit `1cb9fda7`

This is not a generic framework comparison. It is specifically about the design pressure from "niche but not that niche" rollout patterns:

- context management
- user sims
- native tool parsing
- harness-in-sandbox
- harness-outside-of-sandbox
- no sandbox
- groupwise rewards
- intermediate rewards
- multiple environments
- resource management
- custom metrics / error handling
- offline evals

## Short read

`verifiers` is stronger when the center of gravity is:

- grouped RL rollouts
- reward composition
- environment-controlled interaction loops
- built-in sandboxed evaluation modes

`rollouts` is stronger when the center of gravity is:

- honest environment state
- serializability / resumability
- explicit runtime ownership
- composing tool surfaces
- projecting into external agents or attempt executors

The cleanest summary is:

- `verifiers` says: environment + rollout state machine + rubric
- `rollouts` says: environment protocol + serialized state + explicit scorer / executor stages

That difference matters a lot. `verifiers` is more RL-loop-native. `rollouts` is more boundary-native.

## Architectural difference

`verifiers` does not really have a first-class harness algebra. The harness mostly lives inside environment subclass methods such as `setup_state`, `get_prompt_messages`, `env_response`, and `update_tool_args`, with reward semantics centralized in `Rubric`. In practice, a lot of extension power comes from mutating shared rollout `state`.

`rollouts` is trying to make the denotation more explicit. The core `Environment` protocol says the construction path should be `row_to_state(row) -> Environment.deserialize(state)`, and it explicitly separates environment ownership from scoring ownership and from external attempt execution. The docs also state the intended split between `ResourceState`, `ResourceHandle`, and `Environment`, which is a much cleaner answer to "what owns resources versus task semantics?" than anything I saw in `verifiers`.

That said, `rollouts` is not pretending this is finished. The `Environment` protocol itself has a TODO noting that too many semantically different things still hide behind one surface. So this is not "rollouts solved it"; it is "rollouts names the boundary problem more honestly". [dtypes.py](/Users/chiraagbalu/research/rollouts/rollouts/dtypes.py#L1469) [resource_state_and_environment_boundary.md](/Users/chiraagbalu/research/rollouts/docs/design/resource_state_and_environment_boundary.md#L26)

## Pattern-by-pattern

### Context management

`verifiers`: strong but ad hoc. Prompt assembly and multi-turn context are environment responsibilities, but mostly through override points and mutable state.

`rollouts`: more explicit and narrower. The environment can provide `get_system_prompt()`, and eval config owns `prepare_messages`. That is a better ownership story for prompt/task construction, but it is less centralized than `verifiers`' rollout state machine. If you want highly custom context mutation during runtime, `on_assistant_message()` is the escape hatch, not a dedicated context layer. [creating_environment.md](/Users/chiraagbalu/research/rollouts/docs/creating_environment.md#L93) [eval.py](/Users/chiraagbalu/research/rollouts/rollouts/core/eval.py#L44)

Verdict: `rollouts` has the cleaner boundary; `verifiers` has the more built-in rollout-context machinery.

### User sims

`verifiers`: yes, via `MultiTurnEnv.env_response()`.

`rollouts`: yes, via `on_assistant_message()`. `DialogueEnvironment` is the direct analog: after the primary assistant speaks, the environment calls a responder model, records its turn, and injects the responder reply back as a user message. [dialogue.py](/Users/chiraagbalu/research/rollouts/rollouts/environments/dialogue.py#L108)

Verdict: both support this well. `rollouts` is slightly more compositional because user simulation is an optional hook on the environment protocol instead of a subclass family.

### Native tool parsing

`verifiers`: good support for provider-native structured tools, weak support for text-emitted pseudo-tools.

`rollouts`: stronger overall. The internal message model has first-class `ToolCallContent`, `ToolCall`, parse-error fields, and provider adapters for OpenAI, Anthropic, Google, and SGLang-style text parsers. The runtime also feeds malformed tool calls back as tool errors rather than crashing. This means `rollouts` supports both native structured tool calls and at least some text-to-tool parsing paths. [dtypes.py](/Users/chiraagbalu/research/rollouts/rollouts/dtypes.py#L639) [dtypes.py](/Users/chiraagbalu/research/rollouts/rollouts/dtypes.py#L692) [runtime.py](/Users/chiraagbalu/research/rollouts/rollouts/agents/runtime.py#L555) [sglang.py](/Users/chiraagbalu/research/rollouts/rollouts/providers/sglang.py#L916)

Verdict: `rollouts` wins here.

### Harness-in-sandbox

`verifiers`: one of its strongest areas. `SandboxEnv`, `CliAgentEnv`, and `OpenEnvEnv` make sandboxed harness execution a real built-in mode.

`rollouts`: supported, but less singularly central. `SandboxedWorktreeEnvironment` adds OS-level sandboxing on top of a git worktree environment, and there are remote/workspace-backed paths for external runtimes. The difference is that sandboxing in `rollouts` feels like one environment family among several, not the central evaluation abstraction. [sandboxed_worktree.py](/Users/chiraagbalu/research/rollouts/rollouts/environments/sandboxed_worktree.py#L28) [external_attempts.py](/Users/chiraagbalu/research/rollouts/rollouts/eval/external_attempts.py#L294)

Verdict: `verifiers` is stronger if your main question is "how do I run the harness inside the sandbox?" `rollouts` supports it, but it is not the center of the framework.

### Harness-outside-of-sandbox

Both support this.

In `verifiers`, this is the normal in-process path for basic env types.

In `rollouts`, it is also normal: environments are plain protocol objects with `exec_tool()`, and the runtime invokes them directly. [dtypes.py](/Users/chiraagbalu/research/rollouts/rollouts/dtypes.py#L1536) [runtime.py](/Users/chiraagbalu/research/rollouts/rollouts/agents/runtime.py#L624)

Verdict: tie.

### No sandbox at all

Both support this by default.

Verdict: tie.

### Groupwise rewards

`verifiers`: this is first-class. `Rubric` explicitly detects group reward functions and computes group rewards plus rollout-level advantages in `score_group()`. [rubric.py](/tmp/verifiers/verifiers/rubrics/rubric.py#L82) [rubric.py](/tmp/verifiers/verifiers/rubrics/rubric.py#L278)

`rollouts`: much weaker as a surfaced eval abstraction today. The eval layer has explicit `Score`/`Metric` products and can group result lists for analysis, and the training/docs clearly care about advantages and GRPO-style grouping, but I did not find a `verifiers`-style first-class group reward API in the runtime/eval boundary itself. Grouping exists more as downstream analysis/training structure than as a rubric-level scoring primitive. [eval.py](/Users/chiraagbalu/research/rollouts/rollouts/core/eval.py#L15) [native.py](/Users/chiraagbalu/research/rollouts/rollouts/eval/native.py#L1647)

Verdict: `verifiers` clearly wins here.

### Intermediate rewards

`verifiers`: partial but real. Trajectory steps can carry reward and advantage; environments can write them mid-rollout.

`rollouts`: not comparably first-class in the agent runtime. The canonical execution artifact is a `Trajectory`, and scoring happens as an explicit eval stage over an `AttemptResult`. That is cleaner for end-of-attempt scoring, but weaker for native stepwise credit assignment inside the framework boundary. [training/types.py](/Users/chiraagbalu/research/rollouts/rollouts/training/types.py#L204) [native.py](/Users/chiraagbalu/research/rollouts/rollouts/eval/native.py#L322)

Verdict: `verifiers` is ahead for intermediate reward semantics.

### Multiple environments

`verifiers`: strong for heterogeneous benchmark mixtures via `EnvGroup`, weaker for simultaneous multi-environment interaction in one rollout.

`rollouts`: stronger for tool-surface composition. `ComposedEnvironment` merges multiple environments, routes tools by owner, chains system prompts and `on_assistant_message()`, and serializes sub-environments independently. That is much closer to a first-class "multiple environments in one rollout" story, even if the file itself honestly documents unresolved collision and ordering problems. [compose.py](/Users/chiraagbalu/research/rollouts/rollouts/environments/compose.py#L79)

Verdict: `verifiers` is stronger for multi-env eval mixtures; `rollouts` is stronger for composing multiple environment/tool surfaces into one live session.

### Resource management

`verifiers`: moderate support, but ownership is decentralized across env subclasses, sandbox classes, and worker systems.

`rollouts`: conceptually stronger. The code and docs keep insisting that serialization and resource ownership are part of the environment boundary, and the runtime serializes environment state before tool execution, rehydrates a fresh environment per tool call, then reserializes after effects. This is a very opinionated state-ownership model. It is heavier, but it makes resumability and resource provenance much more explicit. [runtime.py](/Users/chiraagbalu/research/rollouts/rollouts/agents/runtime.py#L548) [runtime.py](/Users/chiraagbalu/research/rollouts/rollouts/agents/runtime.py#L596) [resource_state_and_environment_boundary.md](/Users/chiraagbalu/research/rollouts/docs/design/resource_state_and_environment_boundary.md#L28)

Verdict: `rollouts` has the cleaner model; `verifiers` has the looser one.

### Custom metrics / error handling

`verifiers`: metrics are reward functions with weight `0.0`, but rubric exceptions are swallowed into zero scores.

`rollouts`: metrics are explicit `Metric` values inside `Score`, and eval config requires an explicit scorer. Runtime and eval errors are still often converted into recorded failure artifacts rather than crashing the whole run, but that behavior is more explicit at the eval boundary. Score computation failures become an `"error"` metric rather than silently pretending success. [eval.py](/Users/chiraagbalu/research/rollouts/rollouts/core/eval.py#L15) [eval.py](/Users/chiraagbalu/research/rollouts/rollouts/core/eval.py#L44) [native.py](/Users/chiraagbalu/research/rollouts/rollouts/eval/native.py#L322)

Verdict: `rollouts` is semantically cleaner. `verifiers` is operationally convenient but more likely to hide scoring failures.

### Offline evals

`verifiers`: strong persistence and resume for generated outputs; weaker first-class rescoring of arbitrary offline completion datasets.

`rollouts`: stronger explicit offline story. The eval runner writes canonical per-sample artifacts plus `report.json` and partial reports for crash recovery, `EvalConfig` includes `resume_dir`, and the execution result type explicitly stores `environment_state`, `trajectory`, and `evaluation`. The framework is also built around an explicit scorer stage, which is a better fit for offline rescoring than `verifiers`' rubric-inside-rollout model. [eval.py](/Users/chiraagbalu/research/rollouts/rollouts/core/eval.py#L73) [native.py](/Users/chiraagbalu/research/rollouts/rollouts/eval/native.py#L574) [native.py](/Users/chiraagbalu/research/rollouts/rollouts/eval/native.py#L627) [training/types.py](/Users/chiraagbalu/research/rollouts/rollouts/training/types.py#L204)

Verdict: `rollouts` wins for offline eval architecture.

## Where Rollouts Is Actually Better

These are the places where `rollouts` feels materially less preclusive than `verifiers`.

1. Environment construction is explicit.
   `row_to_state -> deserialize` is a much more honest reconstruction boundary than ambient environment initialization.

2. Environment state is first-class.
   `serialize()` and `deserialize()` are mandatory, not optional nice-to-haves.

3. External runtimes are real citizens.
   `attempt_executor` and external-runtime adapters give a direct path for Codex / Claude Code / OpenHands style execution without lying that they are ordinary native tool loops. [external_attempts.py](/Users/chiraagbalu/research/rollouts/rollouts/eval/external_attempts.py#L233) [drivers/protocol.py](/Users/chiraagbalu/research/rollouts/rollouts/drivers/protocol.py#L17)

4. Multiple environment composition is more compositional.
   `ComposedEnvironment` is imperfect, but it is a real object with real semantics.

5. Tool parsing is more pluralistic.
   Native tool calls and text-derived tool calls both exist inside the same message/event model.

## Where Verifiers Is Actually Better

These are the places where `verifiers` currently has the more native answer.

1. Groupwise rewards are first-class.
2. Intermediate reward plumbing exists inside the rollout model.
3. Built-in sandboxed harness modes are more central and mature.
4. Multi-turn environment-driven RL loops are more obviously the main use case.

## Honest bottom line

If your complaint is:

- "this framework hard-codes one rollout shape and makes grouped RL awkward"

then `verifiers` is the better fit.

If your complaint is:

- "this framework lies about who owns resources, state restoration, scoring, or external runtime projection"

then `rollouts` is the better fit.

So for your original list, the compressed judgment is:

- `rollouts` is less fundamentally preclusive on environment/resource/runtime boundaries
- `verifiers` is less fundamentally preclusive on RL reward semantics

That is the real tradeoff. `verifiers` is more natively an RL environment-and-rubric system. `rollouts` is more natively a serializable execution-and-evaluation substrate.

## Can Rollouts Do The Original List?

Short answer: yes, but not uniformly first-class.

- context management: yes
- user sims: yes
- native tool parsing: yes
- harness-in-sandbox: yes
- harness-outside-of-sandbox: yes
- no sandbox at all: yes
- groupwise rewards: partial
- intermediate rewards: partial
- multiple environments: yes
- resource management: yes
- custom metrics / error handling: yes
- offline evals: yes

### Notes

- `context management`: supported via `prepare_messages`, `get_system_prompt()`, and `on_assistant_message()`. This is a cleaner ownership story than `verifiers`, but it is not a dedicated context layer.
- `user sims`: supported directly through `on_assistant_message()`. `DialogueEnvironment` is the clearest built-in example.
- `native tool parsing`: supported for provider-native tool calls, with parse errors preserved in the message model and surfaced back into the loop. `rollouts` also has text-derived tool parsing in some provider paths such as SGLang/Hermes.
- `harness-in-sandbox`: supported through sandboxed environment families and remote/workspace-backed execution paths.
- `harness-outside-of-sandbox`: supported as the ordinary environment `exec_tool()` path.
- `no sandbox at all`: the normal default.
- `groupwise rewards`: only partial as a surfaced eval abstraction. Grouping and advantages clearly matter in training, but I do not see a `verifiers`-style first-class group-rubric scoring interface in the eval/runtime boundary.
- `intermediate rewards`: only partial. End-of-attempt scoring is explicit and clean; native stepwise reward shaping is not equally first-class.
- `multiple environments`: supported via `ComposedEnvironment`, which merges tools, chains hooks, and serializes sub-environments.
- `resource management`: strong. `serialize()` / `deserialize()` are mandatory environment boundaries, and the runtime explicitly rehydrates environment state around tool execution.
- `custom metrics / error handling`: strong. Metrics are explicit `Metric`s inside `Score`, and scoring is an explicit stage rather than being ambient inside the rollout loop.
- `offline evals`: strong. `rollouts` has explicit scorers, resumable eval reports, per-sample artifacts, and stored `environment_state`.

### Bottom line

`rollouts` can do almost all of the original list.

Its weak spots, relative to `verifiers`, are:

- first-class groupwise reward semantics
- first-class intermediate reward semantics

Its strong spots, relative to `verifiers`, are:

- environment/resource/runtime boundary clarity
- serializability and resumability
- external-runtime projection
- offline evaluation architecture
