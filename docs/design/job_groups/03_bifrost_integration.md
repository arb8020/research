# Job Groups: Bifrost Integration Plan

## Why bifrost

`bifrost` already owns:
- process start (`submit()` / `serve()`)
- process identity (`JobInfo` / `ServerInfo`)
- monitoring (`job_*` / `server_*` functions)
- process termination (kill tmux sessions)

A job group is just a thin orchestrator on top of those primitives.

## Proposed Python API (MVP)

- `bifrost.job_group.start_job_group(session, spec) -> JobGroupRun`
- `bifrost.job_group.wait_job_group(session, run) -> int` (wait primary, then teardown)
- `bifrost.job_group.stop_job_group(session, run) -> None`

All functions are “functions over frozen dataclasses”, consistent with existing `bifrost.job` and `bifrost.server`.

## Proposed CLI (MVP)

Add a new CLI command:

```bash
bifrost job-group run <ssh> --file group.yaml
```

This is intentionally just a thin wrapper; the real unit is the Python API.

## Future: multi-host groups

Extend `JobGroupTaskSpec` to include a tri-modal acquisition block:
- `ssh: ...`
- `node_id: ...`
- `provision: { type: A100, count: 1, ... }`

and orchestrate acquisition with `bifrost.provision.acquire_node()` per task.

