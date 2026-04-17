# Session Ownership

Status: framing note. Ownership model, not implementation.
Date: 2026-04-17.

This doc defines **what owns what** in a run. It deliberately avoids implementation details, data layouts, and file formats. Those belong in follow-up docs written at implementation time.

## The framing

Four roles. Keep them separate.

### Session — the record

The authoritative append-only log of what happened. Owns the canonical sequence of *effects* (things the agent did) and *observations* (things that came back). Everything else is derived from or scoped to the session.

A session is what you read when you want to know "what happened in this run." It is the source of truth. Nothing else gets to be.

### Environment — the interpreter

Stateless with respect to history. Given an effect, it produces an observation. Given a *log* of effects, its current state is `fold(effects, initial_state)`.

This is the load-bearing invariant: **environment state is a function of the effect log, nothing else.** Any state an environment holds that isn't recomputable from the log is a bug in the environment's contract. If the environment has a Docker container, the container's state is a consequence of the effects we've applied; if we throw the container away and replay the log onto a fresh one, we're back where we were.

"Replay onto a fresh container" may be slow. That's fine. It's semantically what's happening whether we like it or not.

### Trajectory — a view of the session for training

A particular linear walk through the session, suitable to feed to a training procedure. A session with branches admits many trajectories; a session with one linear path admits one.

Trajectory is derived. It does not own state. If the session is the record and the environment is the interpreter, the trajectory is "the slice a trainer cares about."

### AgentState — resume cursor

Where the interpreter was when it last stopped. Which turn, what pending tool calls, how much budget remaining. Volatile. The minimum extra information needed to pick back up from an interrupted run without re-deriving it from the session.

AgentState is recoverable from session in principle, but it's cheap to carry explicitly, and carrying it explicitly keeps the resume path simple.

## Why this matters

The reason we're writing this down: when these four roles are kept separate, several things fall out cleanly that don't fall out cleanly today.

- **Native and external agents produce the same artifact.** Both yield effects into a session. The loop body (who decides the next effect) differs; the record does not. Downstream consumers don't branch on which harness ran.
- **Non-destructive compaction.** Compaction is a session entry, not a mutation. The context window handed to the LLM is *derived* from session + view policy. Compaction changes the derivation; it doesn't erase history.
- **Rewind and branch are cheap.** A session with effect log `[e1, e2, e3]` can be forked at `e2`: the environment is re-derived by folding `[e1, e2]` through a fresh interpreter, and a new branch of effects grows from there. No special "branching machinery" beyond "another parent pointer in the log."
- **Replay for training and analysis is trivial.** The log is authoritative and complete. Anything downstream can re-execute from it.
- **Environment serialization stops being a quirky per-environment puzzle.** Serialize = "here is the effect log up to this point." Deserialize = "fold it." Some environments will additionally provide fast snapshot serialization as an optimization — see below — but it's an optimization on top of a uniform contract, not a replacement for one.

## Warm vs cold deserialization

`deserialize(session)` can land in different places depending on what resources are available:

- **Warm.** A live interpreter already exists and is compatible with this session — e.g., we kept the container alive, or we grabbed one from a sandbox pool that matches. Deserialization is bookkeeping: point at the existing resource.
- **Cold.** No live interpreter. We provision a new one from scratch and fold the effect log through it.

The caller doesn't need to know which happened. The contract is "after deserialize, the environment reflects the session's final state, regardless of path." Warm is faster. Cold always works.

Environments that today claim "I can't be deserialized" (e.g., TerminalBench's live Docker container) are really saying "I don't have a warm path from serialized bytes." Fine — they still have a cold path: provision a fresh container, replay the effects. That's the contract.

The warm/cold distinction is an implementation concern, not a semantic one. It shows up in performance, not correctness.

## Ownership boundaries (what does NOT own what)

Explicit anti-roles, because these have been smeared in practice:

- **The record does not own the context window.** The LLM context is a *view* derived from the session. What the model sees is a function of session plus view policy (trimming, compaction, branch selection). Today `trajectory.messages` is both record and context; that's the conflation we're unwinding.
- **The environment does not own history.** It folds over history but doesn't carry it. If the environment disappears, the session is intact and the environment is re-derivable.
- **The trajectory does not own identity.** Session owns identity (session id, parent session, branch point). Trajectory is a projection; many trajectories can cite the same session.
- **AgentState does not own persistence.** It's the pause button, not the save file. The save file is the session.

## What this implies for current code (without prescribing the refactor)

Flagging the places where current ownership is smeared. Resolving these is the work; how is a later conversation.

- **`Trajectory.messages` is simultaneously record and context.** The one real conflation. Non-destructive compaction can't exist while they're the same list.
- **Effects are nested inside message content blocks.** Tool calls are `ToolCallContent` inside assistant messages; tool results are tool-role messages. They are effects, structurally, but they don't live at the log level — they live as content of messages. Native and external agents diverge here because external formats have their own effect encoding that we adapt post-hoc.
- **External-agent runs don't write effects live.** External adapters observe turns as they happen (by polling the external tool's own session file) but don't append to our session until the run ends. The discipline "runtime yields effect → session gets appended" is violated on this path.
- **Some environments claim non-serializability.** They do have effect logs; they just haven't been asked to derive state from them.
- **`events.jsonl` is a different channel.** It is a streaming monitor log, not the session. It coexists with the session and is not replaced by it. Useful for live observation; not authoritative for "what happened."

## What this doc is not

- Not a schema. Nothing here commits to JSONL, entry types, field names, serialization formats, or storage locations.
- Not a sequencing plan. No "first do X then Y."
- Not a taxonomy of entry types. That's an implementation choice that should be made against concrete use cases.
- Not a decision on whether `Trajectory` is a type, a function, or a view helper. Derived-from-session is the constraint; the mechanism is open.
- Not a commitment to any particular branching model (forked files vs. leaf cursor vs. something else). The framing supports branching; the model is an implementation choice.

## Related threads

Tracked separately; not scope for this doc.

- **External agent per-turn persistence.** Closing the "external runs don't write to session" gap. Concrete, small, on-ramp to this framing.
- **First-class effects.** Pulling tool calls and results out of message content blocks into the log level. The structural move that makes native and external look the same at the record layer.
- **Environment effect-fold audit.** Walking current environment implementations to identify which genuinely need a cold-path implementation versus which already have one in disguise.
- **Harbor integration.** Running Harbor-provided tasks (TB2, SWE-Bench, 50+ adapters) through rollouts. Wants this framing to be in place so native and external agents both compose cleanly.
- **Tree navigation / branch surgery.** Rewind, fork, replay UX. Falls out cheaply once the framing is real. Not urgent.

## Archive

Prior design work on sessions, trajectories, and branching has been moved to `docs/archive/session/` and `rollouts/docs/archive/session/`. Those docs reflect earlier thinking from prior maintainers and are preserved as historical context, not current direction. Where they conflict with this framing, this framing wins.

## References

- `docs/references/ant-managed-agents.md` — the "decouple the brain from the hands" framing. Session as append-only log. Sandbox as `execute(name, input) → string`. We adopt the decoupling and the roles; we don't commit to their exact interfaces.
- `/tmp/pi-mono/packages/coding-agent/src/core/session-manager.ts` — reference for what typed log entries and parent pointers look like in practice.
