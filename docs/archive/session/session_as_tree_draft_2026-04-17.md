# Session as Tree

Status: exploration. Nothing here is a final decision.
Date: 2026-04-17.

## What we're trying to do

We want to rework how a "run" is recorded in this codebase so that:

- Native rollouts agents and external agents (claude-code, codex, Harbor installed-agents) produce the *same artifact* as they go, not two artifacts bridged by a post-hoc adapter.
- Context management (compaction, trimming, resets) is non-destructive — we can always go back and see what actually happened, even after a compaction step.
- Rewind and branching (pi-mono `/tree` style) are first-class, not a special mode.
- Scoring, analysis, replay, and tree search read one shape of thing, regardless of which harness produced it.

The rough direction we keep circling back to: make the record an **append-only tree of typed entries**. Everything interesting becomes either "append an entry" or "move a cursor."

This doc is a place to collect what we've read, what shape we're tentatively drawn to, and what we don't know yet. It isn't a spec.

## Why this keeps coming up

A few recurring symptoms that all point at the same root:

- `Trajectory` is doing three jobs: the durable record of what happened, the derived context window for the next LLM call, and a mutable working surface the loop edits. These have different lifecycles and different consumers.
- External-agent adapters (`trajectory_from_remote_claude_code` etc.) reconstruct `Trajectory` after the run finishes. That means their session is never a live object — you can't observe it mid-run, you can't branch from it, you can only consume what they hand you at the end.
- Compaction mutates in place. If we ever want to look at what the model saw *before* compaction, we need a harness that happened to stash it. There's no structural guarantee.
- Branching (tree search, exploratory forks) requires copy-and-diverge on `Trajectory`. That's fine for small cases but doesn't compose with anything else.
- Every new external-agent type (next: Harbor installed-agents) adds another adapter. If we don't converge the shape, we write adapter N+1.

## Things we read

### Anthropic Managed Agents

`docs/references/ant-managed-agents.md`. The central move is "decouple the brain from the hands": the harness doesn't live inside the container, it calls the container the same way it calls any other tool (`execute(name, input) -> string`). Session is an append-only event log that lives outside both. Harness yields Effects and appends events. If the harness crashes, you reboot a new one and `getSession(id)` recovers state. If a container dies, the harness catches a tool-call error and Claude decides what to do.

Interfaces they name:

- Session — `getSession`, `getEvents`, `emitEvent`. Append-only log.
- Orchestration — `wake(session_id)`. Any scheduler.
- Harness — yields Effects, appends to Session.
- Sandbox — `provision({resources})`, `execute(name, input) -> string`.
- Tools — name + description + input_schema. MCP satisfies this.

The framing we're taking from this: session is a *context object that lives outside the context window*. Context-for-the-LLM is derived from session on demand, not stored there.

### badlogic/pi-mono `SessionManager`

`/tmp/pi-mono/packages/coding-agent/src/core/session-manager.ts`. Concrete implementation of the managed-agents shape, with a tree twist.

Key observations from reading it:

- Session is a JSONL file. One `SessionHeader` line, then one `SessionEntry` per line. Append-only.
- Every entry carries `id` and `parentId`. The tree structure is implicit in parent pointers — no separate tree data structure, no "branching" concept. A fork is "a parent has two children."
- "Where am I in the tree" is a `leafId` cursor. Moving the cursor = switching branches. Rewinding = moving the cursor to an ancestor. Nothing is deleted when you rewind.
- Context for the LLM is derived: walk from leaf to root, collect entries, transform per compaction/branch-summary entries encountered along the way. This is the piece that makes non-destructive compaction work.
- Entry types they ship: `message`, `model_change`, `thinking_level_change`, `compaction`, `branch_summary`, `label`, `custom`, `custom_message`, `session_info`. Compaction and branch-summary are *entries*, not operations — they sit in the log and the context builder honors them.
- Extensibility is `custom` / `custom_message` entries discriminated on a `customType` string. Extensions persist state in the same log without forking the schema.
- Migrations are in-band. They have a `migrateV1ToV2` function that walks entries and adds `id`/`parentId` to old logs that lack them.
- `ReadonlySessionManager` is a `Pick<...>` of the read methods. Readers can't mutate. Small thing, we'd probably do the same.

### Our current `Trajectory` and adapter code

Briefly, for contrast. `Trajectory` is a flat `list[Message]` plus metadata. External adapters (`trajectory_from_remote_claude_code`, `trajectory_from_remote_codex`, etc.) are adapter functions that take an external run's output and *return* a `Trajectory`. They're batch, post-hoc. The kernelbench `LocalSyncAgentHost` / `RemoteResidentAgentHost` machinery is built on top of these.

## What we're tentatively drawn to

Loose sketch, not a commitment:

- A `Session` type that's an ordered collection of typed entries with parent pointers. Probably persisted as JSONL. Probably lives under `results/.../sessions/<id>.jsonl`.
- Entry types along pi-mono's lines: `MessageEntry`, `CompactionEntry`, `BranchSummaryEntry`, `ModelChangeEntry`, `CustomEntry`, `CustomMessageEntry`, `LabelEntry`, `SessionInfoEntry`. We'd probably prune or add based on what we actually need.
- A `view(session, leaf_id) -> list[Message]` function that derives LLM context. Scorers and the loop call this instead of reading `Trajectory` directly.
- Native rollouts agent loop writes entries as it goes.
- External-agent adapters eventually stream entries into a `Session` live instead of returning a `Trajectory` at the end — but that's downstream work, not P1.

## Things we don't know yet

These are real questions, and the answers probably come from trying to build it, not from deciding in advance.

- **Does `Session` replace `Trajectory` or coexist?** Replace is cleaner long-term. Coexist is safer while consumers migrate. No strong opinion yet.
- **Where does `leaf_id` live?** Inside the session file, as a mutable header field — breaks append-only. As a separate cursor file — introduces two files per session. As its own `CursorEntry` type that gets appended when the cursor moves — stays append-only, preserves cursor history, but means reading the cursor requires scanning for the latest cursor entry. Pi-mono does something like the first; we might prefer the third. Unclear.
- **How do tool calls and tool results associate?** Two shapes: (a) one `MessageEntry` per role with tool results inside user messages (matches our current `Trajectory`), (b) separate `ToolResultEntry` keyed by tool-call-id (matches managed-agents' event-per-effect story and probably makes analysis cleaner). (a) is a smaller change; (b) is more honest. Don't know which is right.
- **What about streaming / partial entries?** If an assistant message streams tokens, is each chunk an entry? A single entry written at the end? Does it matter? Pi-mono seems to write the full message once. We'd probably do the same initially.
- **Concurrency.** Multiple samples writing to their own session files is obviously fine. What about concurrent writers to one session? Unclear if we need it. Branching within one process doesn't need it — one cursor at a time.
- **Schema versioning.** Pi-mono does `SessionHeader.version` + migration functions. That seems right. Question is how strict we are about unknown entry types in forward-compat — skip with warning, or fail loud?
- **Interaction with results/events.jsonl.** We already write `events.jsonl` during evals (per `CLAUDE.md`, this is the "ground truth for what happened"). Is `Session` a different thing from the events stream, or the same thing viewed differently? They might be the same. Or `events.jsonl` might be the flat projection of the session log. Worth thinking about before we double up.
- **How much of pi-mono's extension-entry shape do we actually need?** `CustomEntry` and `CustomMessageEntry` are a nice escape hatch, but they could also be the kind of abstraction we only need if we have extensions. Probably add them later when we have a concrete use case.

## Related work we haven't scoped here

We've talked through several adjacent threads. Naming them so they aren't forgotten, without pretending they're designed:

### Harness abstraction

The goal: both native rollouts agents and external agents (claude-code, Harbor installed-agents) become implementations of the same thing — a loop that yields Effects and appends to Session. Today these are two separate code paths. A session-as-tree structure is a precondition for this unification; without it the two harnesses would still produce subtly different artifacts.

### Environment exposes MCP

If we want external agents to use our custom tools, we probably need `Environment` to expose an MCP server that the external harness can be pointed at. Native rollouts agents would keep using `get_tools()` directly. Same environment, two surfaces. Orthogonal to session work but pairs with the harness abstraction.

### HarborEnvironment and `harbor_v0` eval package

Wrapping `harbor.environments.BaseEnvironment` so we can run Harbor tasks (TB2, SWE-Bench, 50+ others) via rollouts agents. The rollouts-native path doesn't need the harness abstraction or MCP — it just needs Harbor-as-library. The external-agent path (running claude-code-on-TB2 through Harbor and collecting the session in our format) needs both.

### Tree navigation UI

`/tree`-style CLI/TUI for listing entries, jumping the cursor, labeling, forking. Falls out of the data model cheaply if the data is tree-shaped. Not core; built when we want it.

## Rough sequencing

1. Figure out the session shape concretely by building a prototype against one use case (probably `functional_extractor`, since it's the simplest native eval we have). This is where most of the open questions above actually get answered.
2. Migrate remaining native consumers to `Session`.
3. Build HarborEnvironment and `harbor_v0` on top of `Session` with rollouts-native agents only. This exercises the session shape in anger without needing harness/MCP work.
4. Harness abstraction + MCP on Environment. External-agent parity.
5. Tree navigation UI, if useful.

Stages 3 and 4 each probably want their own design doc written at the time they're real. This doc is just about stage 1.

## References

- `docs/references/ant-managed-agents.md`
- `/tmp/pi-mono/packages/coding-agent/src/core/session-manager.ts`
- `/tmp/pi-mono/packages/coding-agent/src/core/agent-session.ts`
- Harbor framework: `https://github.com/laude-institute/harbor`
- Old TB environment (to be superseded): `rollouts/rollouts/environments/terminal_bench.py`
