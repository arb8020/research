# Infrastructure Layers: Bifrost, Miniray, and Pipeline RL

**DRI:** chiraag
**Claude:** [this conversation]

## Context

We have multiple overlapping systems for remote execution and distributed training:

| System | What it does | Current state |
|--------|--------------|---------------|
| **broker** | GPU provisioning (RunPod, Vast) | Working |
| **bifrost** | SSH + deploy + exec on remote nodes | Working |
| **kerbal** | Tmux sessions, log streaming, Python env setup | Working, being absorbed into bifrost |
| **miniray** | Local multi-GPU coordination (fork + socketpair + NCCL) | Working |
| **rollouts** | RL training loops, inference, data buffers | Active development |

Problems:
1. **5+ deploy.py files** reimplementing the same 6-step pattern (provision → deploy → tmux → stream → sync → cleanup)
2. **No clear boundary** between job deployment and coordinated RL training
3. **kerbal absorption** into bifrost is in progress but incomplete
4. **Miles/pipeline RL** will need coordinated weight sync, which none of these currently support remotely

## Out of Scope

- Kubernetes/container orchestration
- Multi-cloud provider abstraction beyond broker
- Fault tolerance / job resumption (future work)

## Solution

Three distinct layers with clear boundaries:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer 3: Pipeline RL (future)                                              │
│  ─────────────────────────────                                              │
│  Coordinated training + inference with in-flight weight sync                │
│  • PipelineRL-style: HTTP handshake + NCCL broadcast                        │
│  • Version-stamped samples, staleness control                               │
│  • Stream-based coordination (Redis/file queues)                            │
│                                                                             │
│  Components: weight_broadcast(), version tracking, stream backend           │
│  Uses: Layer 2 for worker coordination, Layer 1 for deployment              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Worker Coordination                                               │
│  ────────────────────────────                                               │
│  Heinrich-pattern workers with send/recv semantics                          │
│  • Local: fork() + socketpair (miniray today)                               │
│  • Remote: SSH + TCP socket (miniray extension)                             │
│  • Ongoing bidirectional communication, not one-shot jobs                   │
│                                                                             │
│  Components: Worker, RemoteWorker, wait_any(), NCCL setup                   │
│  Uses: Layer 1 for initial node provisioning and code deployment            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 1: Job Deployment                                                    │
│  ─────────────────────────                                                  │
│  One-shot remote job execution with lifecycle management                    │
│  • Provision GPU node (broker)                                              │
│  • Deploy code (bifrost.push)                                               │
│  • Run command (tmux + log streaming)                                       │
│  • Sync results (download)                                                  │
│  • Cleanup (terminate)                                                      │
│                                                                             │
│  Components: frozen specs + pure functions + explicit state dict            │
│  Uses: broker for provisioning, bifrost for SSH/deploy                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Job Deployment (Immediate)

**Goal:** Compress 5+ deploy.py files into one reusable API.

**Design principle:** Functions over classes (nmoe style). Classes only for:
- PyTorch API requirements (Optimizer subclass)
- Resource lifecycle (file handles, async queues, **SSH connections**)

### When to use a class

| Use case | Class? | Why |
|----------|--------|-----|
| SSH connection | ✓ `BifrostClient` | Owns socket, needs cleanup |
| Job config | ✗ frozen dataclass | Just data, no lifecycle |
| Job state | ✗ explicit dict | Visible at call sites |
| Job operations | ✗ pure functions | Take session + state, return result |

### bifrost Refactor Needed

We're adding new types and functions to bifrost. The pattern to follow (not refactoring existing code):

```python
# Wrong pattern (don't do this)
@dataclass
class JobHandle:
    _session: BifrostClient  # circular reference back to session
    name: str
    ...

    def status(self) -> str: ...      # method that just calls _session.exec()
    def wait(self) -> int: ...        # method that just calls _session.exec()
    def logs(self) -> str: ...        # method that just calls _session.exec()
```

Handles that don't own resources - just pass `_session` to every method -
should be frozen data + pure functions instead:

```python
# Refactored (correct)
@dataclass(frozen=True)
class JobInfo:
    """Immutable job identifier - just data."""
    name: str
    tmux_session: str
    log_file: str | None
    workspace: str | None


# Pure functions - session is explicit parameter
def job_status(session: BifrostClient, job: JobInfo) -> str:
    result = session.exec(f"tmux has-session -t {job.tmux_session} 2>/dev/null")
    return "running" if result.exit_code == 0 else "completed"

def job_wait(session: BifrostClient, job: JobInfo, timeout: float | None = None) -> int:
    while job_status(session, job) == "running":
        time.sleep(1.0)
    return job_exit_code(session, job)

def job_logs(session: BifrostClient, job: JobInfo, tail: int = 100) -> str:
    result = session.exec(f"tail -n {tail} {job.log_file}")
    return result.stdout

def job_kill(session: BifrostClient, job: JobInfo) -> None:
    session.exec(f"tmux kill-session -t {job.tmux_session}")
```

Same refactor for `ServerHandle` → `ServerInfo` + functions.

### Types (frozen dataclasses)

```python
@dataclass(frozen=True)
class ProcessSpec:
    """Complete specification of a process to run.

    Immutable - represents what to run, not a running process.
    Can be serialized, logged, compared.

    Env vars are passed here, not cached on Session.
    """
    command: str
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)  # dict for ergonomics
    gpu_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class GPUQuery:
    """GPU provisioning query."""
    type: str = "A100"
    count: int = 1
    max_price: float | None = None
    min_vram_gb: int | None = None
    min_cuda: str = "12.0"
    cloud_type: str = "secure"
    container_disk_gb: int = 100
    volume_disk_gb: int = 0


@dataclass(frozen=True)
class JobSpec:
    """What to run."""
    command: str
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 7200
    log_file: str = "job.log"


@dataclass(frozen=True)
class DeploySpec:
    """Where to run."""
    # Node acquisition (exactly one)
    ssh: str | None = None
    node_id: str | None = None
    provision: GPUQuery | None = None

    # Code deployment
    workspace: str = "~/.bifrost/workspace"
    bootstrap: str | tuple[str, ...] | None = None

    # Lifecycle
    keep_alive: bool = False
```

### State (explicit dict, not class)

```python
def make_job_state() -> dict:
    """Create empty job state."""
    return {
        "job_id": None,
        "session_name": None,
        "log_file": None,
        "client": None,      # BifrostClient (stateful, owns SSH connection)
        "instance": None,    # ClientGPUInstance | None
        "status": "pending",
        "exit_code": None,
    }
```

### Functions

```python
# Core lifecycle
def start_job(deploy: DeploySpec, job: JobSpec, state: dict) -> dict
def poll_status(state: dict) -> str  # "running" | "completed" | "failed"
def wait_for_completion(state: dict, poll_interval: float = 2.0) -> int
def stream_logs(state: dict, follow: bool = True) -> Iterator[str]
def download(state: dict, remote: str, local: str) -> None
def terminate(state: dict) -> None

# Helpers (called by start_job)
def acquire_node(deploy: DeploySpec) -> tuple[BifrostClient, ClientGPUInstance | None]
def provision_node(query: GPUQuery) -> ClientGPUInstance
def start_tmux(client, session_name: str, command: str, ...) -> None
```

### Usage

```python
# Explicit state threading (nmoe style)
deploy = DeploySpec(
    provision=GPUQuery(type="A100", count=2),
    bootstrap="uv sync --extra dev",
)
job = JobSpec(command="python train.py", env={"HF_TOKEN": token})

state = make_job_state()
start_job(deploy, job, state)
exit_code = wait_for_completion(state)
download(state, "results/", "./local/")
terminate(state)

# Or with context manager (convenience, not primary API)
with run_job(deploy, job) as state:
    for line in stream_logs(state):
        print(line)
    download(state, "results/", "./local/")
```

### What it absorbs

| Current | Absorbed into |
|---------|---------------|
| `bifrost.push()` + bootstrap | `start_job()` internals |
| `kerbal.start_tmux_session()` | `start_tmux()` |
| `kerbal.stream_log_until_complete()` | `stream_logs()` + `wait_for_completion()` |
| `bifrost.download_files()` | `download()` |
| `broker.terminate_instance()` | `terminate()` |

### Bug fixes to incorporate

1. Path normalization (`./results` vs `results`)
2. SSH timeout: 600s default
3. CUDA version: `min_cuda="12.0"` default
4. Workspace `~` expansion on remote
5. Tmux session cleanup before start
6. HF cache config when `volume_disk_gb > 0`

### Concrete Migration: tito test + examples/rl

Two files with duplicated patterns that should use the new API:

**Current (both files):**
```python
# Still importing from kerbal (not absorbed!)
from kerbal.job_monitor import LogStreamConfig, stream_log_until_complete
from kerbal.tmux import start_tmux_session

# Duplicated acquire_node pattern (~50 lines each)
def acquire_node(...):
    if ssh:
        return BifrostClient(ssh, ssh_key_path), None
    elif node_id:
        instance = broker.get_instance(...)
    else:
        instance = broker.create(...)
    instance.wait_until_ssh_ready(timeout=600)
    return BifrostClient(...), instance

# Duplicated tmux + streaming dance
session, err = start_tmux_session(
    client=bifrost,
    session_name=session_name,
    command=cmd,
    workspace=...,
    log_file=...,
    capture_exit_code=True,
)
stream_log_until_complete(bifrost, monitor_config)

# Duplicated result sync in finally block
bifrost.download_files(remote_path=..., local_path=...)
if not keep_alive:
    instance.terminate()
```

**After refactor:**
```python
# No kerbal imports - all from bifrost
from bifrost import BifrostClient
from bifrost.job import job_wait, job_stream_logs, job_kill
from bifrost.provision import acquire_node  # new: extracted pattern

# Single acquire_node function (moved to bifrost)
session, instance = acquire_node(
    ssh=args.ssh,
    node_id=args.node_id,
    provision=GPUQuery(type="A100", count=2) if args.provision else None,
)

# Clean job submission
job = session.submit(
    ProcessSpec(command="python", args=(script_path,), cwd=workspace),
    name="training",
)

# Stream logs (function, not method)
for line in job_stream_logs(session, job):
    print(line)

exit_code = job_wait(session, job)

# Download results
session.download_files(...)

# Cleanup
if not keep_alive and instance:
    instance.terminate()
```

### Files to change

- `bifrost/bifrost/job.py` (new) - specs, state, functions
- `bifrost/bifrost/provision.py` (new) - `acquire_node()` extracted from deploy.py files
- Deprecate `kerbal/` - functionality absorbed

### Migration order

**Phase A: Refactor bifrost (steps 1-2)**

1. **Extract `acquire_node()`** into `bifrost/provision.py`
   - Handles ssh / node_id / provision tri-modal pattern
   - Returns `(BifrostClient, ClientGPUInstance | None)`
   - Lives in bifrost (not broker) because it returns a BifrostClient

2. **Add `JobInfo` + functions** to bifrost
   - Delete legacy `JobInfo` class in `types.py` (v1 job manager API) - not used
   - `types.py`: Add new frozen `JobInfo` dataclass (name, tmux_session, log_file, workspace)
   - `types.py`: Add new frozen `ServerInfo` dataclass (same pattern)
   - `job.py` (new): Add `job_status()`, `job_wait()`, `job_stream_logs()`, `job_kill()`
   - `server.py` (new): Add `server_is_healthy()`, `server_wait_until_healthy()`, `server_stop()`
   - `client.py`: Add `submit()` method that returns `JobInfo`, add `serve()` that returns `ServerInfo`

*Validate: run `pytest rollouts/tests/test_tito_correctness.py` and test examples/rl manually.*

**Phase B: Migrate callers (steps 3-4)**

3. **Migrate `tests/test_tito_correctness.py`**
   - Replace kerbal imports with bifrost
   - Use `acquire_node()` + `submit()` + `job_*()` functions
   - Verify test still passes

4. **Migrate `examples/rl/base_config.py`**
   - Same pattern as tito test
   - This exercises the TUI streaming path

**Phase C: Cleanup (step 5)**

5. **Delete kerbal package**
   - First: `grep -r "from kerbal" --include="*.py"` should return nothing (imports removed in Phase B)
   - Then: `rm -rf kerbal/` to delete the package entirely

### Clarifications

**Backwards compatibility**: No deprecation period. All callers are internal (tito test, examples/rl, deploy.py files). Refactor and update callers in the same PR.

**BifrostClient stays a class** with methods for things it owns:
- `exec()`, `push()`, `download_files()` - use the SSH connection
- `submit()`, `serve()` - stay as methods, but return frozen `JobInfo`/`ServerInfo`

The extracted functions operate *on* jobs/servers, not *as* them:
```python
# BifrostClient methods (owns SSH connection)
job = session.submit(spec)  # returns JobInfo (frozen)
session.download_files(...)  # uses SSH connection

# Extracted functions (operate on job, don't own it)
job_wait(session, job)  # takes session explicitly
job_status(session, job)
job_stream_logs(session, job)
```

**acquire_node location**: `bifrost/provision.py`, not broker, because:
- Returns `BifrostClient` (bifrost's type)
- bifrost already imports broker for types
- Semantics: "provision a node and give me a bifrost session"

**ProcessSpec vs command strings**: `submit()` takes a `ProcessSpec` for structured command building. See `docs/design/bifrost_v2.md` lines 249-272 for the full `ProcessSpec` definition. The existing `run_detached()` takes raw command strings and can stay for simple cases. `submit()` is the new primary API.

**Relationship to bifrost_v2.md**: That doc has the full bifrost design including `ProcessSpec`, layer architecture, and kerbal absorption plan. This doc (infrastructure_layers.md) focuses on the three-layer view (bifrost → miniray → pipeline RL) and the functions-over-classes refactor. The main update to bifrost_v2.md: `JobHandle`/`ServerHandle` should be frozen `JobInfo`/`ServerInfo` + pure functions (not classes with methods).

**JobInfo is identifier-only**: The new frozen `JobInfo` contains only identifiers (name, tmux_session, log_file, workspace). Status comes from calling `job_status(session, job)`. No cached state on the dataclass.

**submit() replaces run_detached()**: Same underlying tmux mechanism, better interface. `run_detached()` can stay temporarily for backwards compat but `submit()` is the new API.

**ProcessSpec.env uses dict[str, str]**: The `frozenset[tuple[str, str]]` in bifrost_v2.md was over-engineered. Use regular dict for ergonomics - frozen dataclass + dict is fine.

**GPUQuery defined in bifrost/provision.py**: It's a bifrost concern (what you want) not a broker concern (how to get it). `acquire_node()` internally calls broker APIs.

**Phase A validation**: Phase A just adds new API. Validation = imports work without error. The tito test validation happens after Phase B migrates the test to use new API.

**Jobs and servers together**: Both `submit()` → `JobInfo` and `serve()` → `ServerInfo` are in scope for Phase A. Same pattern, both used by tito test and RL examples.

**Delete legacy methods after grep**: Run `grep -rn "get_job_status\|get_all_jobs\|submit_job" --include="*.py"` first. If only defined (not called), delete.

---

## Layer 2: Worker Coordination (Near-term)

**Goal:** Extend miniray's Heinrich pattern to work over SSH.

### Current (local only)

```python
# miniray today - fork + socketpair
workers = [Worker(work_fn) for _ in range(n)]
for w in workers:
    w.send(config)
ready = wait_any(workers, timeout=1.0)
for w in ready:
    result = w.recv()
```

### Future (local + remote)

```python
# Same API, workers can be remote
local_workers = [Worker(work_fn) for _ in range(4)]
remote_workers = [
    RemoteWorker(host="node2", port=10000),
    RemoteWorker(host="node3", port=10000),
]
workers = local_workers + remote_workers

for i, w in enumerate(workers):
    w.send({"rank": i, "world_size": len(workers)})
```

### Key insight from Heinrich

> "if you replace the UDS with a tcp socket you have the beginning of a distributed system here with much simpler semantics than, say, Ray"

miniray already has `Worker` (local) and `RemoteWorker` (TCP). What's missing:
- **Deployment**: Use Layer 1 to get WorkerServer running on remote nodes
- **Multi-node NCCL**: `setup_nccl_env()` needs MASTER_ADDR coordination

### Why Worker is a class (legitimate state)

Worker owns:
- `pid` - child process ID
- `r`, `w` - file handles for socketpair
- Lifecycle: needs `wait()` for cleanup

This is the "resource ownership" case from CLASSES_VS_FUNCTIONAL.

### Files to change

- `miniray/remote_worker.py` - refinement
- `miniray/cluster.py` - use Layer 1 for deployment

---

## Layer 3: Pipeline RL (Future)

**Goal:** Coordinated training + inference with in-flight weight sync.

### What miniray needs for PipelineRL-style training

| Capability | miniray today | Need to add |
|------------|---------------|-------------|
| Worker spawn | fork() + socketpair | ✓ |
| Remote workers | RemoteWorker via TCP | Layer 1 integration |
| Weight sync | `load_state_dict()` blocking | NCCL broadcast (non-blocking) |
| Version tracking | None | `weight_version` per sample |
| Multi-node NCCL | Local only | Cross-node group init |
| Backpressure | None | `max_lag` + streams |

### Architecture (from PipelineRL)

```
Trainer                    Inference (vLLM)              Actor
───────                    ────────────────              ─────

train_step()
    │
    ├── HTTP /receive_weight_update ──▶ pause
    │                                      │
    ├── NCCL broadcast ───────────────────▶ recv
    │                                      │
    │                                      └──▶ resume
    │
    └── write(WeightUpdateSuccess) ──────────────────▶  read version
                                                              │
                                           ◀── HTTP /generate ┘
                                                 (stamped with version)
```

### Key patterns to adopt

1. **In-flight sync**: HTTP handshake → NCCL broadcast → continue (no pause)
2. **Version stamping**: Every sample tagged with `model_version`
3. **Staleness control**: `max_lag` parameter
4. **Stream-based IPC**: File/Redis JSONL queues

### Functions to add (not classes)

```python
# Weight sync
def broadcast_weights(model, worker_group, version: int) -> None
def receive_weights(model, trainer_group) -> int  # returns version

# Version tracking (explicit state dict, like nmoe's zero2_state)
def make_version_state() -> dict
def stamp_sample(sample: dict, version_state: dict) -> dict
def check_staleness(sample: dict, current_version: int, max_lag: int) -> bool
```

### Open questions

- [ ] vLLM extension vs nano-inference?
- [ ] DeepSpeed ZeRO-3 vs FSDP?
- [ ] Redis vs files for streams?

---

## Migration Path

### Phase 1: Layer 1
- [ ] Extract `acquire_node()` into `bifrost/provision.py`
- [ ] Delete legacy job manager API (`JobInfo`, `get_job()`, etc.)
- [ ] Refactor `JobHandle` → `JobInfo` (frozen) + functions in `job.py`
- [ ] Refactor `ServerHandle` → `ServerInfo` (frozen) + functions in `server.py`
- [ ] Migrate tito test and examples/rl to use new API
- [ ] Deprecate kerbal

### Phase 2: Layer 2
- [ ] Integrate miniray with Layer 1 for deployment
- [ ] Multi-node NCCL setup
- [ ] Test with `examples/rl/base_config.py`

### Phase 3: Layer 3 (future)
- [ ] Add weight broadcast functions to miniray
- [ ] Version tracking state
- [ ] Stream backend
- [ ] Integrate with Miles

---

## Decisions

### 1. Functions vs Classes for Layer 1

**Decision:** Functions with explicit state dict (nmoe style)

```python
# Yes: explicit state threading
state = make_job_state()
start_job(deploy, job, state)
wait_for_completion(state)

# No: class with methods for namespacing
handle = job.start()
handle.wait()
```

Rationale: State is visible at call sites. No hidden mutations. Easier to test.

### 2. Layer 2: Extend miniray

**Decision:** Yes, extend miniray (not new package)

- Already has correct Heinrich pattern
- Worker class is legitimate (owns pid, file handles)
- Just needs Layer 1 integration

### 3. Layer 3: Build using PipelineRL patterns

**Decision:** Build, don't fork

- Copy patterns: HTTP + NCCL weight sync, version stamping, streams
- Skip: Hydra, their vLLM fork specifics
- Integrate with our miniray + rollouts

---

## References

### Code Style (why functions over classes)
- `~/research/docs/code_style/CLASSES_VS_FUNCTIONAL.md` - when to use classes vs functions
- `~/research/docs/code_style/multiprocessing_heinrich.md` - Heinrich's fork+socketpair pattern
- `~/research/docs/code_style/casey_muratori_semantic_compression.md` - "don't abstract until 2+ uses"

### nmoe (functional style reference)
- `/tmp/nmoe/nmoe/train.py:36-160` - `train()` function with explicit state threading
- `/tmp/nmoe/nmoe/opt.py:146-227` - `build_optimizer()` returns tuple, not class
- `/tmp/nmoe/nmoe/opt.py:289-331` - `step()` takes explicit state dict, mutates it

### PipelineRL (weight sync patterns for Layer 3)
- `/tmp/pipelinerl/pipelinerl/vllm1.py:55-97` - `WorkerExtension` for in-flight weight updates
- `/tmp/pipelinerl/pipelinerl/vllm1.py:100-121` - `WeightUpdateManager` HTTP+NCCL handshake
- `/tmp/pipelinerl/pipelinerl/actor.py:164-197` - version stamping on samples

### miniray (Heinrich pattern implementation)
- `miniray/worker.py:206-252` - fork + socketpair Worker class
- `miniray/worker.py:289-308` - send/recv over socketpair
- `miniray/remote_worker.py` - TCP version of same pattern

### Current duplication (what we're compressing)
- `rollouts/tests/test_tito_correctness.py:56-105` - `acquire_node_runpod_only()`
- `rollouts/tests/test_tito_correctness.py:40-41` - kerbal imports (should be bifrost)
- `rollouts/examples/rl/base_config.py:36-37` - kerbal imports (should be bifrost)
- `rollouts/examples/rl/base_config.py:107-114` - `start_tmux_session()` dance

### bifrost (what we're changing)
- `bifrost/bifrost/types.py:140-161` - Legacy `JobInfo` class (v1 job manager API) - delete entirely
- `bifrost/bifrost/client.py` - Add `submit()` returning `JobInfo`, add `serve()` returning `ServerInfo`
- `bifrost/bifrost/client.py` - Delete legacy job manager methods: `get_job()`, `get_all_jobs()`, `submit_job()`, `get_job_status()`

## Files

**Modify (Layer 1):**

bifrost changes:
- `bifrost/bifrost/types.py`:
  - Delete legacy `JobInfo` class (lines 140-161, v1 job manager API)
  - Add new frozen `JobInfo` dataclass (name, tmux_session, log_file, workspace)
  - Add new frozen `ServerInfo` dataclass (name, tmux_session, log_file, port, health_endpoint)
- `bifrost/bifrost/provision.py` (new):
  - `acquire_node(ssh, node_id, provision) -> tuple[BifrostClient, ClientGPUInstance | None]`
- `bifrost/bifrost/job.py` (new):
  - `job_status(session, job) -> str`
  - `job_wait(session, job, timeout) -> int`
  - `job_logs(session, job, tail) -> str`
  - `job_stream_logs(session, job) -> Iterator[str]`
  - `job_kill(session, job) -> None`
- `bifrost/bifrost/server.py` (new):
  - `server_is_healthy(session, server) -> bool`
  - `server_wait_until_healthy(session, server, timeout) -> bool`
  - `server_logs(session, server, tail) -> str`
  - `server_stop(session, server) -> None`
- `bifrost/bifrost/client.py`:
  - `submit()` returns `JobInfo` not `JobHandle`
  - `serve()` returns `ServerInfo` not `ServerHandle`
  - Delete `get_job()`, `get_all_jobs()`, `submit_job()` (v1 job manager API)
- Deprecate `kerbal/` - functionality absorbed

**Modify (Layer 2):**
- `miniray/remote_worker.py`
- `miniray/cluster.py`

**Modify (Layer 3, future):**
- `miniray/weight_sync.py` (new)
- `rollouts/rollouts/training/`
