# Event log refactor

This note turns the current logging discussion into a concrete migration plan.
It complements [argus_rollouts_event_boundary.md](./argus_rollouts_event_boundary.md):
that document explains ownership; this one explains how to make the current
event/log surfaces honest.

## Problem

The journals are already JSONL, but the event envelope is not yet canonical.

Current drift:

- some records use `event` as the discriminator
- some records still effectively use `message` as the discriminator
- forwarded log lines live in `line`
- some consumers reconstruct semantics from mixed-field heuristics
- viewers still depend on Rollouts-specific code paths instead of reading the
  journals as a generic event stream

This makes local reasoning worse:

- producers and consumers do not agree on the canonical event shape
- tailers have to compensate for schema drift
- monitor code owns interpretation work that should belong at emission time

## Target contract

Use one canonical event envelope across the monorepo:

- `event`: required type discriminator
- timestamp: required event time (`ts` / `timestamp` can be unified later, but
  the event must carry one canonical time field at emission)
- `message`: optional human text, never a type discriminator
- `line`: optional raw forwarded log line
- `log_blob`: optional multiline payload
- flat extra fields for domain-specific context

Operational rules:

- producers own schema truth
- consumers may keep compatibility fallbacks for old runs, but should not invent
  new semantics
- tail/TUI surfaces should prefer `line` when present, otherwise render compact
  event summaries
- storage layout (`run.jsonl`, `events.jsonl`, etc.) is a separate decision from
  the event envelope

## Ownership

- `rollouts` owns workload event vocabulary and emits canonical records
- `argus` owns generic journal reading, tailing, syncing, and viewing surfaces
- no viewer should need Rollouts-specific code just to read JSONL records

## Migration steps

1. Define a tiny shared event-emission helper around the canonical envelope.
   Keep it small: no framework, no consumer-side normalization layer.

2. Migrate emitters so new records always carry `event`, and `message` is only
   human text.

3. Keep compatibility in consumers only where old runs require it.
   Compatibility should be explicit and temporary.

4. Make the generic monitor/tail/TUI read JSONL directly in `argus`.
   `rollouts.monitor` is transitional and should shrink as that happens.

5. Revisit whether separate journals are still buying us anything only after
   the event envelope is honest.

## Non-goals

- building a standalone `logging/` framework first
- hiding schema drift behind a smarter tailer
- deciding all future event taxonomy up front

The first job is smaller: make the boundary honest, then keep the viewers dumb.
