# Argus Supervisor Architecture

## Context

The repo already has most of the lower layers:

- `broker` provisions compute
- `bifrost` operates on remote workspaces and processes
- `miniray` provides narrow process / transport primitives

What is missing is a supervisor layer that models detached runs honestly.

The important clarification from discussion is:

- the laptop should not be the synchronization point
- runs must keep going if the laptop sleeps or disconnects
- the laptop should be able to reattach later and reconstruct state
- the UI should consume `snapshot + diff events`, like an orderbook

So the laptop is better described as a command origin and subscriber than as
the sole control plane.

## Problem

Today, truth about a remote run is split across:

- local job metadata
- remote tmux sessions
- provider instance state
- synced log files
- monitor-side reconstruction logic

That works for launching and tailing, but it does not give one honest model of:

- what run exists
- what attempt is executing
- what allocation is bound to it
- what facts have happened so far
- how a fresh client should reattach

## Goals

- Fire-and-forget launch semantics.
- Remote execution survives laptop disconnect, sleep, and monitor exit.
- Reattach from fresh local state.
- One append-only event stream per run.
- `snapshot + subscribe(from_cursor)` as the primary observation contract.
- UI and CLI issue commands; they do not mutate process state directly.
- Run identity stays stable across retries and node replacement.

## Non-Goals

- Replacing `broker`, `bifrost`, or `miniray`.
- Building a full multi-tenant cluster scheduler.
- Making the laptop the sole durable database for all runtime facts.
- Hiding infrastructure details behind fake generic abstractions.

## Core Entities

### RunSpec

User intent:

- "train this model this way"
- "launch this eval"
- "run this kernel benchmark"

This is the semantic object the user cares about.

### Run

A durable logical run created from a `RunSpec`.

This should survive:

- retry
- reallocation
- reconnect
- multiple UI sessions

### Attempt

One concrete execution of a run.

If a node dies and the same run is tried again, that is a new attempt of the
same run, not a new run.

### Allocation

The leased infrastructure bound to an attempt:

- provider
- node ID
- image/runtime contract
- mounted volumes
- workspace path

### Command

A request to change runtime state indirectly:

- `launch_run`
- `cancel_run`
- `retry_run`
- `publish_artifact`

Commands are not facts. They are requests whose results become events.

### Event

An append-only fact:

- `run_created`
- `attempt_created`
- `allocation_bound`
- `attempt_started`
- `stage_updated`
- `metrics_reported`
- `artifact_recorded`
- `heartbeat`
- `attempt_failed`
- `attempt_finished`

### Projection

A materialized view derived from events:

- active runs
- startup ladder
- latest metrics
- instability timeline
- kernel leaderboard

## Observation Model

The primary runtime protocol should be:

1. fetch current snapshot
2. subscribe from the snapshot cursor
3. apply diffs locally

Sketch:

```python
snapshot = supervisor.get_snapshot(run_id)
cursor = snapshot.cursor

for event in supervisor.subscribe(run_id, from_cursor=cursor):
    apply(event)
```

This is a better fit than "tail files until the UI guesses the current state."

## Ownership

### What Lives On Remote Workers

Remote workers should own the runtime facts needed for detached execution:

- the actual supervised process
- the authoritative append-only event journal for the run
- local stdout/stderr capture
- checkpoint / artifact announcements
- heartbeat emission
- enough local state to rebuild the latest snapshot

This is what allows runs to continue with no laptop connected.

### What Lives On The Laptop

The laptop should own:

- command creation
- subscriptions to remote snapshots/events
- optional local cache / index for cross-run queries
- TUI / web rendering
- derived projections used for comparison and dashboards

The laptop may cache and index remote facts, but it should not be required for
the run to make progress.

## Package Boundaries

### broker

`broker` should own infrastructure allocation and runtime-contract satisfaction.

This boundary is wider than "return an arbitrary SSH string". It should
normalize provider-specific setup that affects correctness, including:

- image selection
- attached volumes
- cache mounts
- exposed ports
- readiness/bootstrap contract

The output of `broker` should be closer to "runtime-ready allocation" than
"machine you can SSH into".

### bifrost

`bifrost` should own remote workspace and process operations on a prepared node:

- connect to prepared node
- materialize workspace
- run commands
- run detached commands
- stream logs
- upload/download files
- generic remote agent management, if needed

It should not own run semantics like retries, attempts, projections, or event
schemas.

### miniray

`miniray` should remain the narrow worker / transport substrate inspired by the
Heinrich note:

- local `fork + socketpair`
- remote TCP worker handles
- simple servers
- optional narrow helpers like `LogsServer`

It should not become an experiment manager.

### argus

`argus` is the supervisor layer above those packages.

It should own:

- run / attempt / allocation / command / event model
- remote supervisor protocol
- event journal semantics
- snapshot materialization
- command handling
- first-class subscription surface for UIs

## Remote Supervisor Shape

The smallest honest remote supervisor is a long-lived process that:

- starts the actual workload
- captures lifecycle transitions as events
- writes an append-only journal
- answers `get_snapshot(run_id)`
- answers `subscribe(run_id, from_cursor)`
- optionally accepts `submit_command(run_id, cmd)`

This is the natural home for Argus.

`miniray.logs_server` is a useful transport/data-serving primitive, but not a
supervisor by itself. Serving files is not the same thing as owning run state.

## First Event Families

### Runtime Health

- allocation acquired
- workspace prepared
- bootstrap started/finished
- process started/exited
- heartbeat

### Training Health

- step metrics
- NaN / overflow / divergence signals
- throughput
- checkpoint written
- weight sync visibility

### Kernel Work

- benchmark started/finished
- correctness pass/fail
- latency / throughput metrics
- leaderboard entry update

### Artifact Lifecycle

- checkpoint created
- artifact published
- weights visible at path / alias

## First Projections

- `run_summary`
  Current status, stage, attempt, allocation, last heartbeat.
- `startup_ladder`
  Provisioning -> workspace -> bootstrap -> launch -> healthy runtime.
- `instability_timeline`
  Ordered training-health events and warnings.
- `kernel_leaderboard`
  Correctness-gated performance ranking.
- `compare_runs`
  Side-by-side summary from normalized run metadata and latest metrics.

## MVP

The smallest useful first implementation is:

1. keep `broker`, `bifrost`, and `miniray` mostly as they are
2. add a remote Argus supervisor process per run
3. store the authoritative run journal remotely as JSONL first
4. materialize a current snapshot from that journal
5. expose `snapshot + subscribe(from_cursor)` to one TUI client
6. optionally mirror the remote stream into a local SQLite index later

This is enough to support:

- detached distributed training launches
- detached kernel benchmark runs
- reattachable monitoring after laptop sleep
- projection-driven UI without pretending file tailing is the state model

## Why This Is Honest

This architecture makes the real distinctions explicit:

- infrastructure allocation is not the same as a run
- a run is not the same as an attempt
- commands are not facts
- logs are not lifecycle truth
- UI is not process ownership
- projections are not raw state

That is the main value of Argus.
