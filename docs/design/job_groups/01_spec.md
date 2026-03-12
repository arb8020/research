# Job Groups: Proposed Spec (Repo-local)

## Goal

Represent “multiple processes that should be treated as one unit” as a single spec:
- Can be persisted in a YAML file
- Can be executed via a Python API
- Can be surfaced via a small CLI wrapper

## MVP: single-host spec

MVP assumes all tasks run on the same remote host (one `BifrostClient`).

```yaml
name: example
termination_delay_seconds: 30
primary: driver

tasks:
  proxy:
    kind: server
    port: 8080
    command: ["python", "-m", "rollouts.proxy", "--port", "8080"]

  worker0:
    kind: job
    command: ["python", "-m", "miniray.worker_server", "--port", "10000"]

  driver:
    kind: job
    command: ["python", "-m", "rollouts.scripts.remote_eval_driver", "--proxy", "http://127.0.0.1:8080"]
```

## Types (proposed)

Conceptually:
- `JobGroupSpec`: name, primary, tasks, termination delays
- `JobGroupTaskSpec`: kind (`job` vs `server`), process spec, server port/health if needed
- `JobGroupRun`: result of starting a group (task infos + group id/name)

## Non-goals (MVP)

- Multi-host provisioning (`broker` integration per task)
- Service discovery beyond “localhost and fixed ports”
- Automatic retries / resumption

