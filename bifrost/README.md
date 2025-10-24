# Bifrost

Deploy code and manage jobs on remote GPU instances.

## CLI Reference

```bash
bifrost init                                          # Create .env template
bifrost push <ssh> --bootstrap "uv sync"              # Deploy code to remote
bifrost exec <ssh> "python train.py"                  # Execute command (sync)
bifrost deploy <ssh> "python train.py" --bootstrap    # Deploy + execute (convenience)
bifrost run <ssh> "python train.py" --name my-job     # Run in background (detached)
bifrost jobs <ssh>                                    # List all jobs
bifrost logs <ssh> <job-id> --follow                  # Stream job logs
bifrost download <ssh> <remote-path> <local-path>     # Download files
bifrost upload <ssh> <local-path> <remote-path>       # Upload files
```

**SSH format:** `user@host:port` (e.g., `root@157.157.221.29:20914`)
**Global options:** `--ssh-key`, `--quiet`, `--json`, `--debug`

## Python API Reference

```python
from bifrost import BifrostClient

# Initialize client
client = BifrostClient(
    ssh_connection="root@gpu.example.com:22",
    ssh_key_path="~/.ssh/id_ed25519"
)

# Deploy code
client.push(bootstrap_cmd=["uv sync --extra gpu"])

# Execute command (sync)
result = client.exec("python train.py")
print(result.stdout)

# Run detached job (background)
job = client.run_detached(
    command="python train.py --epochs 100",
    name="training-job",
    env={"CUDA_VISIBLE_DEVICES": "0,1"}
)

# Monitor job
status = client.get_job_status(job.job_id)
print(f"Status: {status.status.value}")  # running, completed, failed

# Stream logs
for line in client.follow_job_logs(job.job_id):
    print(line, end="")

# Wait for completion
client.wait_for_completion(job.job_id, poll_interval=5)

# Download results
client.download_files(
    remote_path="~/.bifrost/workspace/outputs",
    local_path="./results",
    recursive=True
)
```

## Setup

Create `.env` file:
```bash
SSH_KEY_PATH=~/.ssh/id_ed25519
```

## Implementation Details

- **Git-based deployment**: Uses git worktrees to deploy code while preserving uncommitted changes
- **Job management**: Background jobs run in tmux sessions for persistence
- **Workspace**: Code deployed to `~/.bifrost/workspace/` on remote
- **Retry logic**: Automatic retry with exponential backoff for network operations
- **SSH**: Uses paramiko for SSH connections and SFTP for file transfers
