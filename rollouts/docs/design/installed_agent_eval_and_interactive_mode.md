# Installed-Agent Evals And Interactive Mode

This is a stub design note for two related directions:

1. adopt Terminal-Bench / Harbor-style task-resource patterns for KernelBench
2. add an interactive/manual entry mode for eval tasks

## Why

Right now the codebase has three different denotations that are only partly unified:

- API-model evals via `rollouts.eval.run` / `argus run`
- installed-agent task runs via `rollouts --env tbench --driver claude|codex|...`
- manual interactive task sessions via the ordinary Rollouts CLI

The first path is reasonably productized.
The second and third paths exist in code, but they are not yet a clear first-class eval story.

## Workstream 1: TerminalKernelBench

Adopt the important Terminal-Bench / Harbor boundary for a KernelBench task environment:

- task resource owns workspace/container/session/test execution
- environment exposes the tool contract used by the agent
- scorer/verifier consumes the task result, not arbitrary ad hoc state

Desired shape:

- `TerminalKernelBenchTaskResource`
  - provision workspace
  - expose terminal / file / test primitives
  - own cleanup and logging layout
- `TerminalKernelBenchEnvironment`
  - expose the coding surface to the agent
  - expose task-complete / submit-result semantics
- `TerminalKernelBenchResult`
  - score
  - correctness
  - speedup / benchmark facts
  - failure reason

Why this is better than the current KernelBench-only multi-turn environment:

- installed-agent CLIs like Claude Code / Codex fit this boundary naturally
- manual interactive use fits the same boundary naturally
- it is closer to Harbor's "run black-box agent in prepared task container" model
- it keeps task semantics separate from agent/runtime choice

Initial TODOs:

- define the task-resource interface for TerminalKernelBench
- decide whether the first surface is `terminal`, `coding`, or both
- normalize the result payload so scoring does not depend on environment internals
- preserve the existing KernelBench runtime contract and backend requirements

## Workstream 2: Interactive Evals

Support a user entering the same eval task manually, not only through a batch evaluator.

Important distinction:

- this is not "pause an arbitrary API eval and magically take over"
- this is "materialize the same task/workspace and let a human or installed agent operate on it"

Desired product surface:

- launch a task from an eval/problem set
- choose agent mode:
  - API model
  - installed agent CLI
  - manual interactive session
- keep the same task resource / logging / verification layout
- optionally save the resulting trajectory as an eval artifact

Possible CLI shape:

```bash
# Autonomous installed-agent task run
rollouts --env terminal_kernelbench --driver codex --task-id ...

# Manual interactive task run
rollouts --env terminal_kernelbench --task-id ...

# Eventually: launch from an eval/problem-set config
argus run --config ... --mode interactive
```

Initial TODOs:

- define what "interactive eval" means as a product type
- decide how task/problem-set selection maps into an interactive run
- decide whether interactive runs should produce normal eval artifacts or a sibling format
- decide whether "reattach to running task" is in scope now or later

## Likely Architecture

The honest split still looks like:

- `argus`
  - launch / attach / sync / run identity
- `rollouts`
  - task semantics
  - installed-agent drivers
  - interactive/manual TUI
  - artifact emission
- task repo such as `charisma`
  - KernelBench-specific task resource, environment, scoring, trusted configs

So the interactive/manual mode should likely be layered over the same local artifact stream rather than inventing a separate side channel.

## Open Questions

- Should installed-agent evals become a first-class `EvalSpec` sum branch, separate from API-model evals?
- Should `TerminalKernelBench` live in `charisma`, with only generic task-resource protocols in `rollouts`?
- How much of Harbor's trial/result layout do we want to adopt directly?
- Do we want "manual drop-in" only at task start, or also mid-run attach/takeover?

## Immediate Next Step

Do the smallest honest slice first:

- design `TerminalKernelBenchTaskResource`
- make one single-task manual run work
- then add one autonomous Codex/Claude Code run on top of the same task resource
- only after that, wire it into batch eval orchestration
