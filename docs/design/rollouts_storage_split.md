# Rollouts Storage Split

## Claim

`rollouts` should use a hybrid storage model:

- `SQLite` as the canonical store for mutable local session state and queryable metadata
- `JSONL/files` for append-only artifacts, trajectories, spans, logs, and eval event streams

It should not treat the current file store as the long-term canonical representation for everything.

## Why

The repo already contains two different kinds of data with different denotations:

1. Append-only artifacts
2. Mutable shared state with invariants

Those want different storage.

### Append-only artifacts want files

These are naturally logs, traces, or replayable outputs:

- eval progress is derived from `events.jsonl`
- trajectories are written as per-sample `.jsonl`
- request spans are appended to `spans.jsonl`
- debug/error logs are written as JSONL files

This is honest file-shaped data. Human readability and replay matter more than indexed updates.

Relevant code:

- [progress_display.py](/Users/chiraagbalu/research/rollouts/rollouts/progress_display.py#L1)
- [native.py](/Users/chiraagbalu/research/rollouts/rollouts/eval/native.py#L680)
- [store.py](/Users/chiraagbalu/research/rollouts/rollouts/store.py#L395)
- [store.py](/Users/chiraagbalu/research/rollouts/rollouts/store.py#L410)
- [logging_utils.py](/Users/chiraagbalu/research/rollouts/rollouts/logging_utils.py#L19)

### Session state wants a database

The current `FileSessionStore` already admits this in its own `TODO(database)` comment:

- full-text search
- complex queries
- atomic transactions
- better multi-process safety

Relevant code:

- [store.py](/Users/chiraagbalu/research/rollouts/rollouts/store.py#L6)

The reason is straightforward: session metadata is not just a blob. It is mutable state with invariants:

- one current `status` per session
- one current tag set per session
- one parent/child relationship graph
- one latest `updated_at`
- queries like latest session, sessions by tag, sessions by status

That is database-shaped data.

Today the file store handles this by:

- read-modify-write on `session.json`
- scanning directories to list sessions
- opening `messages.jsonl` just to count lines

Relevant code:

- [store.py](/Users/chiraagbalu/research/rollouts/rollouts/store.py#L358)
- [store.py](/Users/chiraagbalu/research/rollouts/rollouts/store.py#L440)

This is workable for a prototype, but it pushes concurrency, indexing, and state-machine correctness into ad hoc filesystem logic.

## Existing Proof

The repo already wrote the indexing half of the answer.

`rollouts.session_index` is explicitly:

> queryable SQLite index over Claude Code and Codex session JSONL files

It keeps raw JSONL as source of truth for imported external session logs, but uses SQLite for:

- sessions
- turns
- tool calls
- file touches
- tags
- indexed queries

Relevant code:

- [session_index.py](/Users/chiraagbalu/research/rollouts/rollouts/session_index.py#L1)
- [session_index.py](/Users/chiraagbalu/research/rollouts/rollouts/session_index.py#L50)
- [session_index.py](/Users/chiraagbalu/research/rollouts/rollouts/session_index.py#L528)

This is the honest pattern: raw event log plus relational projection.

## Recommendation

### Canonical in SQLite

Move these into a SQLite-backed `SessionStore`:

- `sessions`
  - `id`
  - `parent_id`
  - `branch_point`
  - `status`
  - `reward`
  - `created_at`
  - `updated_at`
  - `endpoint`
  - `environment_state`
- `session_tags`
- session summaries / latest-session queries
- branch / child lookups
- any waiting / queued / pending control state

This is the mutable control plane.

### Canonical as files

Keep these as files:

- `messages.jsonl`
- `spans.jsonl`
- eval `events.jsonl`
- trajectory `.jsonl`
- sample result JSON / artifacts
- debug and error logs

This is append-only data and should stay cheap to inspect and stream.

### Optional projection/indexes

If needed, build SQLite indexes over the append-only files for:

- text search over messages
- training data extraction
- aggregate reward queries
- “show all sessions that touched X”

But that index should be understood as a projection over artifacts, not as the artifact itself.

## Why Not DuckDB

DuckDB is the wrong default primary store here.

It is stronger for analytical scans over file datasets, but `rollouts` session storage is mostly:

- point lookups
- updates
- appends
- transactional metadata changes
- multi-process correctness

That is SQLite territory, not DuckDB territory.

DuckDB may still be useful later for offline analytics over exported JSONL/Parquet artifacts.

## Why Not Postgres

Postgres only becomes the default answer if `rollouts` turns into a real shared network service with:

- multiple machines/processes mutating the same state
- multi-user access
- remote control plane semantics
- operational requirements beyond one local workspace

That does not appear to be the current shape.

## Practical Rule

For `rollouts`, use:

- `JSONL/files` for events and artifacts
- `SQLite` for local mutable session/control state

Short version:

- events -> files
- state -> SQLite

## Migration Shape

1. Keep `FileSessionStore` as compatibility/debug fallback.
2. Add `DatabaseSessionStore` backed by SQLite.
3. Make SQLite canonical for session metadata and queries.
4. Continue writing append-only message/span/event files.
5. Optionally mirror or derive indexes from those files when needed.

This preserves replayability and human inspection while moving correctness-critical state into a real transactional store.
