# Remote Job API

**DRI:** chiraag
**Claude:** [this conversation]

## Context
We have 5+ deploy.py files that each reimplement the same 6-step pattern (provision → deploy → tmux → stream → sync → cleanup) with slight variations and accumulated bug fixes. Compress into a single unified API.

## Out of Scope
- Changing broker's GPU provisioning internals
- Multi-node distributed jobs (future work)
- Kubernetes/container orchestration

## Solution
**Input:** GPU query + command + workspace config
**Output:** Streaming logs + synced results + automatic cleanup

## Usage

```python
from bifrost import RemoteJob, GPUQuery

# Full example: provision, run, sync, cleanup
job = RemoteJob(
    # Node acquisition (exactly one required)
    provision=GPUQuery(type="A100", count=2, max_price=3.0),
    # OR: ssh="root@host:22"
    # OR: node_id="runpod:abc123"

    # Deployment
    bootstrap="uv sync --extra dev",

    # Execution
    command="python train.py",
    env={"HF_TOKEN": os.getenv("HF_TOKEN")},
    timeout=7200,
)

with job.run() as handle:
    # Stream logs (blocking)
    for line in handle.logs():
        print(line)

    # Sync results
    handle.download("results/", "./local_results/")

# Auto-cleanup on exit (unless keep_alive=True)
```

```python
# Minimal example: reuse existing node
job = RemoteJob(
    node_id="runpod:abc123",
    command="python eval.py",
    keep_alive=True,  # Don't terminate after
)

with job.run() as handle:
    handle.wait()  # Block until done, no streaming
    handle.download("output.json", "./")
```

```python
# Detached example: start and exit
job = RemoteJob(
    ssh="root@gpu:22",
    command="python long_train.py",
)

handle = job.start()  # Non-blocking
print(f"Job running in tmux: {handle.session_name}")
print(f"Reconnect: bifrost attach {handle.job_id}")
# No cleanup - detached jobs persist
```

---

## Details

### Flow
1. **Acquire node**: Use existing SSH, reuse broker instance, or provision new
2. **Deploy code**: `bifrost.push()` with optional bootstrap command
3. **Start job**: Launch in tmux session with log file capture
4. **Stream/wait**: Either stream logs or poll for completion marker
5. **Sync results**: Download specified paths to local
6. **Cleanup**: Terminate instance unless `keep_alive=True`

### Core Types

```python
@dataclass
class GPUQuery:
    """Query for GPU provisioning."""
    type: str = "A100"           # GPU type filter (contains match)
    count: int = 1               # Number of GPUs
    max_price: float | None = None  # Max $/GPU/hour
    min_vram_gb: int | None = None  # Min VRAM per GPU
    cloud_type: str = "secure"   # "secure" or "community"
    min_cuda: str = "12.0"       # Min CUDA version
    container_disk_gb: int = 100
    volume_disk_gb: int = 0


@dataclass
class RemoteJob:
    """Unified remote job configuration."""
    # Node acquisition (exactly one required)
    ssh: str | None = None
    node_id: str | None = None
    provision: GPUQuery | None = None

    # Deployment
    workspace: str = "~/.bifrost/workspace"
    bootstrap: str | list[str] | None = None

    # Execution
    command: str
    working_dir: str | None = None  # Relative to workspace
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 7200  # seconds

    # Lifecycle
    keep_alive: bool = False

    def run(self) -> "JobContext":
        """Run job with context manager for cleanup."""
        ...

    def start(self) -> "JobHandle":
        """Start job detached (no auto-cleanup)."""
        ...


@dataclass
class JobHandle:
    """Handle to a running or completed job."""
    job_id: str
    session_name: str
    log_file: str
    instance: ClientGPUInstance | None
    client: BifrostClient

    def logs(self) -> Iterator[str]:
        """Stream log lines."""
        ...

    def wait(self, timeout: int | None = None) -> int:
        """Wait for completion, return exit code."""
        ...

    def download(self, remote: str, local: str) -> None:
        """Download files/directories."""
        ...

    def terminate(self) -> None:
        """Terminate the instance."""
        ...
```

### Absorbed Patterns

| Current Location | Absorbed Into |
|------------------|---------------|
| `broker.create()` + query building | `GPUQuery` + `RemoteJob(provision=...)` |
| `bifrost.push()` + bootstrap | `RemoteJob(bootstrap=...)` |
| `kerbal.start_tmux_session()` | `JobHandle` internal |
| `kerbal.stream_log_until_complete()` | `JobHandle.logs()` |
| `bifrost.download_files()` | `JobHandle.download()` |
| `broker.terminate_instance()` | `JobContext.__exit__()` |

### Bug Fixes to Incorporate

1. **Path normalization**: Handle `./results` vs `results` in download paths
2. **SSH timeout**: Use 600s default (900s was too long, 300s too short)
3. **CUDA version**: Default `min_cuda="12.0"` for PyTorch 2.x compatibility
4. **Workspace expansion**: Expand `~` on remote before use
5. **Tmux cleanup**: Kill existing session with same name before starting
6. **HF cache**: Auto-configure when `volume_disk_gb > 0`

### Open Questions
- [ ] Should `GPUQuery` support provider preference ordering?
- [ ] How to handle multi-step bootstraps (uv sync then pip install)?
- [ ] Should we expose tmux session name for manual attach?
- [ ] How to handle job resumption after disconnect?

### Files
**Read:**
- `dev/outlier-features/deploy.py` - full provision→cleanup pattern
- `dev/speedrun/deploy.py` - exec_stream variant
- `dev/integration-evaluation/deploy.py` - kerbal setup patterns
- `examples/rl/base_config.py` - minimal run_remote()
- `tests/test_tito_correctness.py` - embedded remote scripts

**Modify:**
- `bifrost/bifrost/client.py` - add RemoteJob
- `bifrost/bifrost/types.py` - add GPUQuery, JobHandle
- (deprecate) `kerbal/` - absorb into bifrost
