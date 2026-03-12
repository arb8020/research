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

`argus monitor` now uses an honest local model:

- resolve a run from a path, `--latest`, or the local jobs registry
- consume `run.jsonl` and `monitor.jsonl`
- render a terminal view or a static HTML snapshot

It does not query providers or tmux directly. The monitor should consume
durable run artifacts instead of reconstructing truth from substrate-specific
state.
