# Job Groups (SkyPilot) → Where It Fits Here

Reference: SkyPilot “Job Groups” (`https://blog.skypilot.co/job-groups/`, docs: `https://docs.skypilot.co/en/latest/reference/job-groups/index.html`).

## What SkyPilot Job Groups are (in 60s)

SkyPilot’s job groups are a **lifecycle + discovery wrapper** around multiple “tasks” that should:
- Start together (often heterogeneous: service + workers + driver)
- Discover each other (service discovery)
- Shut down together (**primary** finishes ⇒ group terminates; optional termination delay)

This is “multi-process orchestration” expressed as a single unit.

## Where this fits in *our* architecture

This repo already has a clean separation:
- **broker**: provision GPU instances
- **bifrost**: deploy + run/serve processes via SSH + tmux (jobs/servers) + logs
- **miniray**: worker coordination (send/recv)
- **rollouts**: orchestration logic (eval/training loops) + trio parallelism

SkyPilot Job Groups map most naturally to **Layer 1: Job Deployment** as a higher-level primitive:

> A *JobGroup* is “multiple `bifrost` jobs/servers with shared lifecycle semantics”.

In practice this would be the missing glue for patterns we already do manually:
- start proxy/server (long-running)
- start N workers (long-running)
- run a driver/orchestrator (finite)
- teardown everything when the driver finishes (or fails)

## Proposed minimal slice (MVP)

Implement a “job group” concept on top of **bifrost**:
- Group runs on **one remote host** (single `BifrostClient`) first
- Group tasks can be:
  - `submit()` jobs (finite)
  - `serve()` servers (long-running)
- One task is the **primary** (must be a `submit()` job in MVP)
- When primary completes:
  - wait `termination_delay`
  - stop remaining tasks (kill tmux sessions)

This intentionally skips SkyPilot’s Kubernetes-only service discovery; instead we focus on
**lifecycle composition** (the part we most need today).

## Next slices (after MVP)

1. **Multi-host groups**: each task can have its own acquisition spec (`ssh` / `node_id` / `provision`)
2. **Service discovery**: write a small “group registry” file (JSON) and/or bring up a tiny HTTP registry
3. **Failure semantics**: configurable “any task fails ⇒ terminate group” vs “only primary matters”
4. **Rollouts integration**: one command that brings up “proxy + workers + driver” for remote eval

## Design notes (why bifrost is the right place)

Job groups are fundamentally about:
- starting processes (tmux-backed)
- monitoring/waiting
- structured teardown

Those are bifrost concerns (not broker, and not miniray).

## Deeper docs

- `docs/design/job_groups/01_spec.md` — proposed API + YAML schema
- `docs/design/job_groups/02_semantics.md` — lifecycle rules + failure/timeout semantics
- `docs/design/job_groups/03_bifrost_integration.md` — CLI + Python API plan
- `docs/design/job_groups/04_rollouts_use_cases.md` — concrete rollouts scenarios

