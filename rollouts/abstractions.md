# Abstractions

Notes on library boundaries and potential refactoring.

## Current State

| Library | Responsibility |
|---------|----------------|
| **broker** | GPU provisioning (create/terminate instances, provider abstraction) |
| **bifrost** | SSH primitives + code deployment (`push()` syncs git repo) |
| **kerbal** | tmux session management, log streaming, dep bootstrapping |
| **miniray** | Distributed computing (multi-node workers, NCCL setup) |

## The Leak

`bifrost.push()` does code sync, but `kerbal` does bootstrapping. Both are "get machine ready to run code" concerns split across two libs.

## Proposed Cleaner Split

| Layer | Responsibility |
|-------|----------------|
| **broker** | Get me a machine (provision/terminate) |
| **bifrost** | SSH/SFTP primitives (connect, exec, transfer) |
| **kerbal** | Deploy & run jobs (code sync, deps, docker, tmux, logs) |
| **miniray** | Distributed compute (multi-node coordination, NCCL) |

kerbal would own the full "deploy code + deps + run reliably + capture logs" story, using bifrost as its SSH layer.

## Ideal API

```python
instance = broker.create(...)
kerbal.deploy(instance, workspace, bootstrap_cmd)  # uses bifrost.push internally
kerbal.run(instance, cmd, log_file)                # tmux + script
kerbal.sync_logs(instance, remote_path, local_path)
```

## Questions

- Should `bifrost.push()` move into kerbal?
- Should kerbal also handle docker?
- How does this relate to Ray/Slurm job submission patterns?

## Related

- Ray Job Submission API solves similar problems
- Slurm handles job execution + logging automatically
- We're rebuilding these patterns for single-node GPU research use case
