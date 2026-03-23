# Canonical Environment Examples

These are stubbed example shapes for the environment families we currently want
people to reason with.

They are intentionally illustrative, not yet blessed runnable configs.

TODO(run-and-bless): turn each example below into a real shaken-out config and
point to the concrete file to copy.

## 1. Immediate Eval + Native Agent

Use this shape when:
- the task is primarily "produce a submission / answer"
- there is no persistent workspace as the main interaction object
- the environment owns verification or immediate judged feedback

Sketch:

```python
config = BenchmarkConfig(...)
environment = ImmediateEvalEnvironmentConfig(...)
agent = NativeAPIAgentConfig(endpoint=EndpointConfig(...))
run = EvalRunConfigSpec(...)
```

Current forcing example:
- KernelBench immediate-eval `write_kernel` / `codeblock` mode

## 2. Workspace + Native Agent

Use this shape when:
- the task's main denotation is a persistent coding workspace
- the agent edits files / runs commands / finalizes a submission
- the environment owns workspace semantics and scoring/finalization

Sketch:

```python
config = BenchmarkConfig(...)
environment = WorkspaceEnvironmentConfig(resource=SandboxResourceConfig(...))
agent = NativeAPIAgentConfig(endpoint=EndpointConfig(...))
run = EvalRunConfigSpec(...)
```

Current forcing example:
- KernelBench SDK workspace mode

## 3. Workspace + External Agent

Use this shape when:
- the environment is still a persistent workspace
- but execution is delegated to an external runtime like Claude Code or Codex
- benchmark semantics stay fixed; only agent execution mode changes

Sketch:

```python
config = BenchmarkConfig(...)
environment = WorkspaceEnvironmentConfig(resource=SandboxResourceConfig(...))
agent = ClaudeCodeAgentConfig(...)  # or CodexAgentConfig(...)
run = EvalRunConfigSpec(...)
```

Current forcing examples:
- KernelBench Claude Code workspace mode
- KernelBench Codex workspace mode

## 4. Immediate Eval + External Scorer

Use this shape when:
- the environment stages the interaction
- but a separate scorer/verifier consumes the result afterward
- scoring is not owned by the environment itself

Sketch:

```python
config = BenchmarkConfig(...)
environment = ImmediateEvalEnvironmentConfig(...)
agent = NativeAPIAgentConfig(endpoint=EndpointConfig(...))
scorer = ExplicitScorer(...)
run = EvalRunConfigSpec(...)
```

This shape is included because it is a real semantic distinction even when the
runtime protocol looks similar.

## Notes

- `workspace` vs `immediate_eval` is the primary environment-family split.
- `native` vs `external` is an execution-mode split, not an environment-family
  split.
- `environment-owned` vs `external` scoring is another orthogonal distinction.
- We do not currently make `single_turn` vs `multi_turn` a top-level family.
  That may matter later, but it does not appear to be the main ownership split
  in today's code.
