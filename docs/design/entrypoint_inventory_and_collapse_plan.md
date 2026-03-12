# Entrypoint Inventory and Collapse Plan

## Goal

Get to:

- few user-facing entrypoints
- clear ownership by layer
- compact launch and monitoring codepaths

This is an inventory of the current `rollouts` entrypoints and a proposed
collapse plan.

## Current entrypoints that matter

### Keep as primary product entrypoints

- `python -m rollouts`
  - implemented by [__main__.py](/Users/chiraagbalu/research/rollouts/rollouts/__main__.py)
  - dispatches into the main CLI in [cli/main.py](/Users/chiraagbalu/research/rollouts/rollouts/cli/main.py)
- `python -m argus run`
  - implemented by [run.py](/Users/chiraagbalu/research/rollouts/rollouts/run.py)
  - this is the public control-plane launch entrypoint for now
- `python -m argus monitor`
  - implemented by [tui/monitor_cli.py](/Users/chiraagbalu/research/rollouts/rollouts/tui/monitor_cli.py)
  - this is the right user-facing monitoring entrypoint for now

### Keep, but likely as domain-specific subcommands later

- `python -m rollouts.eval`
  - implemented by [eval/__main__.py](/Users/chiraagbalu/research/rollouts/rollouts/eval/__main__.py)
  - currently delegates to [eval/run.py](/Users/chiraagbalu/research/rollouts/rollouts/eval/run.py)
- inference/frontends servers
  - [inference/server.py](/Users/chiraagbalu/research/rollouts/rollouts/inference/server.py)
  - [frontend/server.py](/Users/chiraagbalu/research/rollouts/rollouts/frontend/server.py)
  - these are real service entrypoints, not duplicates of the training launcher

### Collapse / de-emphasize

- [run_rl.py](/Users/chiraagbalu/research/rollouts/rollouts/run_rl.py)
  - duplicated `run.py` semantics with a narrower training-only wrapper
  - should be removed in favor of `rollouts.run`
- [modal_runner.py](/Users/chiraagbalu/research/rollouts/rollouts/modal_runner.py) standalone `main()`
  - useful as implementation detail
  - should not remain a first-class user-facing launcher
  - Modal should be a provider/runtime under `rollouts.run`
- [tui/monitor.py](/Users/chiraagbalu/research/rollouts/rollouts/tui/monitor.py) direct `main()`
  - should remain implementation detail behind `python -m argus monitor`
- [tui/__main__.py](/Users/chiraagbalu/research/rollouts/rollouts/tui/__main__.py)
  - redundant if `python -m argus monitor` is the real monitor entrypoint
- [tui/remote_runner.py](/Users/chiraagbalu/research/rollouts/rollouts/tui/remote_runner.py)
  - implementation tool, not a product entrypoint

### Ignore for this compression pass

- tool/debug scripts under `tools/functional_extractor/`
- ad hoc environment demos/tests
- pretrain smoke config CLIs
- session index CLI

These are not the main launch surface problem.

## Duplicated codepaths

### Launch duplication

1. `run.py` vs `run_rl.py`
- both load config
- both override hardware/provider
- both dispatch local/modal/SSH
- `run_rl.py` was historical duplication

2. `run.py` vs `modal_runner.py`
- before the recent refactor, both partially owned source/deploy policy
- still split between:
  - top-level dispatch in `run.py`
  - provider-specific execution in `modal_runner.py`

3. `monitor_cli.py` vs `monitor.py`
- one is the actual CLI
- one is the TUI implementation
- this split is fine as long as only the CLI is treated as public

## Target entrypoint shape

### User-facing

- `python -m rollouts`
  - interactive/agent CLI
- `python -m argus run`
  - launch training/eval/benchmark jobs
- `python -m argus monitor`
  - attach, observe, cancel, inspect runs

Everything else should either:
- become a subcommand behind those
- or become an implementation detail

## Target layer ownership

### rollouts.run

Should own:
- config loading
- turning config into runtime/materialization requests
- dispatch to lower layers
- workload-launch implementation behind `argus run`

Should not own:
- provider-specific resource logic
- duplicated dirty-source policy
- run supervision semantics

### broker

Should own:
- runtime-ready allocation/resource contract
- provider-specific provisioning

### bifrost

Should own:
- transport-agnostic execution/session operations on prepared resources
- source sync
- workspace materialization
- exec / detached exec / download / log streaming

### argus

Should own:
- detached run truth
- run / attempt / allocation / command / event model
- snapshot + subscribe
- projection-friendly event journal

## Collapse order

### Phase 1

- make `argus run` and `argus monitor` the public control-plane entrypoints
- keep `rollouts.run` as implementation, not public surface
- remove `run_rl.py`
- keep `modal_runner.py` only as implementation detail

### Phase 2

- make Modal and SSH look like the same execution/session layer from `rollouts.run`
- continue pushing runtime/materialization/source-sync semantics downward

### Phase 3

- move detached run semantics out of `rollouts` launch code and into `argus`
- keep `python -m argus monitor` as the user-facing observer/command tool

## Immediate conclusion

The shortest path to "clear/few entrypoints and compact codepaths" is:

1. standardize on `argus run`
2. kill `run_rl.py` as a separate launcher
3. treat `modal_runner.py` as backend implementation, not public API
4. keep only one public monitor entrypoint: `python -m argus monitor`
