# bifrost

SSH-based code deployment and remote job execution. Deploy code via git worktree, run jobs synchronously or detached in tmux, stream logs back.

## Core concept

`acquire_node` gives you a `BifrostClient` connected to a GPU. Then you push code and run commands.

```python
from bifrost import acquire_node, GPUQuery

# Provision new node
bifrost, instance = await acquire_node(
    provision=GPUQuery(type="A100", count=1, exposed_ports=(9100,))
)

# Reuse existing node
bifrost, instance = await acquire_node(node_id="runpod:abc123")

# Connect via raw SSH
bifrost, _ = await acquire_node(ssh="root@host:22")
```

## Key BifrostClient methods

```python
# Deploy code (git-based, creates ~/.bifrost/workspaces/<name>/ on remote)
workspace = bifrost.push("~/.bifrost/workspaces/my-project")

# Run command (blocking)
result = bifrost.exec("python train.py", working_dir=workspace)
print(result.stdout, result.returncode)

# Run detached in tmux (non-blocking)
job = bifrost.run_detached("python train.py", name="training", working_dir=workspace)

# Download results
bifrost.download_files("~/results/", "./local-results/", recursive=True)
```

## CLI

```bash
bifrost push <ssh-string> --bootstrap "uv sync"
bifrost exec <ssh-string> "python train.py"
bifrost run <ssh-string> "python train.py" --name job
bifrost logs <ssh-string> <job-id> --follow
bifrost download <ssh-string> <remote-path> <local-path>
```

## Key gotchas

- Code deployment is git-based — uncommitted changes won't be deployed (you'll get a warning)
- Jobs run in tmux sessions on the remote, so they persist if SSH disconnects
- `working_dir` must be set explicitly if you want to run from the deployed workspace
- `InstanceNotFoundError` when trying to reuse a terminated pod
