# Bifrost

Deploy code and run jobs on remote GPU instances via git-based deployments.

## Installation

```bash
uv pip install -e .
```

## Quick Start

```bash
# Setup SSH key path
bifrost init
# Edit .env with SSH_KEY_PATH

# Deploy code with dependencies
bifrost push root@gpu.example.com:22 --bootstrap "uv sync"

# Run command (streaming output)
bifrost exec root@gpu.example.com:22 "python train.py"

# Run in background (tmux session)
bifrost run root@gpu.example.com:22 "python train.py" --name training
```

## Python API

```python
from bifrost import BifrostClient

# Initialize client
client = BifrostClient(
    ssh_connection="root@gpu.example.com:22",
    ssh_key_path="~/.ssh/id_ed25519"
)

# Deploy code
workspace_path = client.push(bootstrap_cmd="uv sync --extra gpu")

# Execute with streaming output
for line in client.exec_stream("python train.py", working_dir=workspace_path):
    print(line)

# Run detached job in tmux
job = client.run_detached(
    command="python train.py --epochs 100",
    no_deploy=True  # already deployed
)

# Monitor job logs
for line in client.follow_job_logs(job.job_id):
    print(line, end="")

# Download results
client.download_files(
    remote_path="~/.bifrost/workspace/outputs",
    local_path="./results",
    recursive=True
)
```

## Common Patterns

### Deploy and run with environment variables

```python
# From dev/outlier-features/deploy.py
client = BifrostClient(instance.ssh_connection_string(), ssh_key_path)

# Deploy with bootstrap
workspace_path = client.push(bootstrap_cmd="uv sync --extra dev")

# Run with environment variables in tmux
hf_token = os.getenv("HF_TOKEN")
cmd = f"export HF_TOKEN='{hf_token}' && python analyze.py config.py"
tmux_cmd = f"tmux new-session -d -s analysis '{cmd}'"
client.exec(tmux_cmd)
```

### Poll for completion with markers

```python
# From dev/corpus-proximity/deploy.py
poll_interval = 30
for i in range(max_iterations):
    check_cmd = """
    test -f .pipeline_complete && echo 'COMPLETE' && exit 0
    test -f .pipeline_failed && echo 'FAILED' && exit 0
    echo 'RUNNING'
    """
    result = client.exec(check_cmd)
    status = result.stdout.strip().split('\n')[-1]

    if status == 'COMPLETE':
        break
    time.sleep(poll_interval)
```

### Upload config and run

```python
# Upload config file
remote_config_path = "~/.bifrost/workspace/configs/config.py"
client.exec(f"mkdir -p ~/.bifrost/workspace/configs")
with open(local_config_path) as f:
    config_content = f.read()
client.exec(f"cat > {remote_config_path} <<'EOF'\n{config_content}\nEOF")

# Run analysis with config
client.exec(f"python analyze.py {remote_config_path}")
```

## CLI Reference

```bash
# Deployment
bifrost push <ssh> --bootstrap "uv sync"              # Deploy code
bifrost upload <ssh> <local> <remote>                 # Upload files

# Execution
bifrost exec <ssh> "python train.py"                  # Sync execution
bifrost run <ssh> "python train.py" --name job        # Background (tmux)
bifrost deploy <ssh> "python train.py" --bootstrap    # Deploy + exec

# Job Management
bifrost jobs <ssh>                                    # List jobs
bifrost logs <ssh> <job-id> --follow                  # Stream logs

# File Transfer
bifrost download <ssh> <remote> <local>               # Download files
```

## How It Works

1. **Git-based deployment**: Creates git worktree and pushes to `~/.bifrost/workspace/`
2. **Persistent jobs**: Background jobs run in tmux sessions
3. **SSH/SFTP**: All operations via paramiko with retry logic

## Configuration

Create `.env`:
```bash
SSH_KEY_PATH=~/.ssh/id_ed25519
```
