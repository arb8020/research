# Bifrost v2: Unified Remote Execution

> **Status:** Design document (not yet implemented)
> **Date:** 2024-12-17
> **Related:** broker, miniray, kerbal (to be absorbed)

---

## Motivation

The current infrastructure has four libraries with unclear boundaries:

| Library | Current Role | Problem |
|---------|-------------|---------|
| **broker** | Provision GPU machines | Leaks software concerns (Docker image selection) |
| **bifrost** | SSH connection + file transfer | Too low-level, no process abstraction |
| **kerbal** | Environment setup + job lifecycle | Depends entirely on bifrost, should be merged |
| **miniray** | Worker communication pattern | Orthogonal, stays separate |

**Key insight:** Bifrost and kerbal are always used together. Kerbal adds no value without bifrost. Merging them eliminates the artificial boundary and creates a cleaner API.

---

## Design Principles

From our code style guides:

1. **Write usage code first** - Design the API by writing how we want to use it
2. **Continuous granularity** - Every higher-level function can be replaced by lower-level calls
3. **Don't reuse until 2+ examples** - Let structure emerge from compression
4. **Classes for resources, functions for computation** - Frozen dataclasses for data
5. **Minimize stateful components** - Clear ownership of state

---

## Usage Code First

### Basic Usage

```python
from bifrost import connect, ProcessSpec

# Connect to a machine (broker gave us MachineHandle)
session = connect(machine)

# Transfer code
workspace = session.push("./my_project")

# Run a command
result = session.exec("nvidia-smi")
print(result.stdout)

# Run with structured process spec
spec = ProcessSpec(
    command="python",
    args=["train.py", "--lr", "0.001"],
    cwd=workspace,
    env={"CUDA_VISIBLE_DEVICES": "0,1", "HF_TOKEN": token},
)
result = session.run(spec)
```

### Persistent Servers

```python
from bifrost import connect, ProcessSpec
from bifrost.inference import sglang

# Build command using preset
spec = sglang.build_spec(
    model="Qwen/Qwen2.5-7B-Instruct",
    port=30000,
    gpu_ids=[0, 1],
)

# Run as persistent server
server = session.serve(
    spec,
    port=30000,
    health_endpoint="/health",
    health_timeout=600,
)
print(f"Server ready at {server.url}")

# Use the server...
# ...

# Clean shutdown
server.stop()
```

### Environment Setup

```python
from bifrost import connect

session = connect(machine)
workspace = session.push("./my_project")

# Setup Python environment
env = session.setup_python_env(
    workspace,
    requirements=["torch>=2.0", "sglang[all]"],
    python_version=">=3.10",
)

# Run with that environment
spec = ProcessSpec(
    command=env.python,  # /workspace/.venv/bin/python
    args=["train.py"],
    cwd=workspace,
    env=env.env_vars,  # Includes PATH, PYTHONPATH
)
result = session.run(spec)
```

### Background Jobs

```python
from bifrost import connect, ProcessSpec

session = connect(machine)
workspace = session.push("./my_project")

# Submit long-running job (runs in tmux, survives disconnection)
job = session.submit(
    ProcessSpec(
        command="python",
        args=["train.py", "--epochs", "100"],
        cwd=workspace,
        env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    ),
    name="training-run-001",
)

print(f"Job submitted: {job.name}")
print(f"Logs: {job.log_file}")

# Check status later (even from new session)
job = session.get_job("training-run-001")
print(job.status())  # "running" | "completed" | "failed"
print(job.logs(tail=50))

# Cancel if needed
job.kill()
```

### Continuous Granularity in Action

```python
# Layer 0: Raw SSH (full control)
result = session.exec("cd /workspace && CUDA_VISIBLE_DEVICES=0 python train.py")

# Layer 1: Structured execution (ProcessSpec)
result = session.run(ProcessSpec(
    command="python",
    args=["train.py"],
    cwd="/workspace",
    env={"CUDA_VISIBLE_DEVICES": "0"},
))

# Layer 2: Persistent process (tmux wrapper)
server = session.serve(spec, port=30000, health_endpoint="/health")

# Layer 3: With environment setup
env = session.setup_python_env(workspace, requirements=["torch"])
result = session.run(ProcessSpec(command=env.python, args=["train.py"], ...))

# Layer 4: ML presets
spec = sglang.build_spec(model="...", port=30000)
server = session.serve(spec, port=30000, ...)
```

Each layer builds on the one below. No holes - you can always drop down.

---

## What is a Session?

A Session is **a live SSH connection to a remote machine**. Nothing more.

```python
@dataclass
class Session:
    """A live SSH connection to a remote machine."""
    _ssh: paramiko.SSHClient
    _sftp: paramiko.SFTPClient | None
    host: str
    port: int
    user: str

    def close(self) -> None:
        """Close the connection."""
        if self._sftp:
            self._sftp.close()
        self._ssh.close()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *args) -> None:
        self.close()
```

**Why Session exists:**
- SSH handshake is slow (~100-500ms)
- Reusing one connection for multiple operations is faster
- It's a resource that needs cleanup (context manager)

**What Session is NOT:**
- Not a namespace for methods (though it has methods for convenience)
- Not a place to cache env vars or other state
- Not a god object

**Current bifrost already does this** - `BifrostClient` holds a persistent `_ssh_client` and reuses it via `_get_ssh_client()`. The class is just badly named.

---

## Environment Variables: Per-Command, Not Per-Session

Each `exec()` call spawns a **new shell process** on the remote:

```python
session.exec("export FOO=bar")
session.exec("echo $FOO")  # Empty! Different shell process
```

This is because SSH `exec_command()` opens a new channel each time. The shell is **non-login, non-interactive** - it doesn't source `~/.bashrc` or `~/.bash_profile`.

**Design decision:** Env vars are passed explicitly each time via `ProcessSpec.env`. No caching on Session.

```python
env = {"CUDA_VISIBLE_DEVICES": "0,1", "HF_TOKEN": token}

# Pass explicitly each time
session.run(ProcessSpec(command="python", args=["train.py"], env=env))
session.run(ProcessSpec(command="python", args=["eval.py"], env=env))
```

This is explicit and avoids hidden state.

---

## Core Types

### Frozen Dataclasses (Immutable Data)

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
    env: frozenset[tuple[str, str]] = frozenset()
    gpu_ids: tuple[int, ...] | None = None

    def with_env(self, **kwargs) -> "ProcessSpec":
        """Return new spec with additional env vars."""
        new_env = dict(self.env) | kwargs
        return replace(self, env=frozenset(new_env.items()))

    def with_cwd(self, cwd: str) -> "ProcessSpec":
        """Return new spec with different cwd."""
        return replace(self, cwd=cwd)


@dataclass(frozen=True)
class ExecResult:
    """Result of command execution - immutable."""
    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def check(self) -> "ExecResult":
        """Raise if command failed."""
        if not self.success:
            raise RuntimeError(f"Command failed ({self.exit_code}): {self.stderr}")
        return self


@dataclass(frozen=True)
class PythonEnvState:
    """Result of environment setup - immutable."""
    python: str          # /workspace/.venv/bin/python
    bin_dir: str         # /workspace/.venv/bin
    workspace: str       # /workspace
    env_vars: frozenset[tuple[str, str]]  # PATH, PYTHONPATH, etc.

    def to_env_dict(self) -> dict[str, str]:
        return dict(self.env_vars)
```

### Classes (Stateful Resources)

```python
class Session:
    """SSH session - a live connection to reuse."""
    # See "What is a Session?" section above
    ...

class ServerHandle:
    """Handle to a running server process."""
    session: Session
    name: str
    port: int
    url: str
    log_file: str

    def is_alive(self) -> bool: ...
    def is_healthy(self) -> bool: ...
    def wait_until_healthy(self, timeout: float = 300) -> bool: ...
    def logs(self, tail: int = 100) -> str: ...
    def stop(self) -> None: ...


class JobHandle:
    """Handle to a background job."""
    session: Session
    name: str
    log_file: str

    def status(self) -> Literal["running", "completed", "failed", "unknown"]: ...
    def is_alive(self) -> bool: ...
    def logs(self, tail: int = 100) -> str: ...
    def stream_logs(self, timeout: float | None = None) -> Iterator[str]: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...
```

---

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BIFROST LAYERS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 4: ML Presets (bifrost.inference)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  sglang.build_spec(model, port, ...)  →  ProcessSpec                │   │
│  │  vllm.build_spec(model, port, ...)    →  ProcessSpec                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  Layer 3: Environment Setup                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  session.setup_python_env(workspace, requirements)  →  PythonEnvState│  │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  Layer 2: Persistent Processes                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  session.serve(spec, port, ...)   →  ServerHandle                   │   │
│  │  session.submit(spec, ...)        →  JobHandle                      │   │
│  │  session.get_job(name)            →  JobHandle | None               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  Layer 1: Structured Execution                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  session.run(spec: ProcessSpec)   →  ExecResult                     │   │
│  │  session.push(local_path)         →  workspace path                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  Layer 0: Raw SSH                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  session.exec(cmd: str)           →  ExecResult                     │   │
│  │  session.exec_stream(cmd)         →  Iterator[str]                  │   │
│  │  session.upload(local, remote)                                      │   │
│  │  session.download(remote, local)                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Shell Backend: Nushell Integration

### The Problem

Building shell commands via string concatenation is error-prone:

```python
# Current approach - shell escaping hell
def build_command(spec: ProcessSpec) -> str:
    env_parts = []
    for k, v in spec.env.items():
        escaped = v.replace("'", "'\\''")  # Hope we got this right
        env_parts.append(f"{k}='{escaped}'")
    env_prefix = "export " + " ".join(env_parts) + " && " if env_parts else ""
    return f"cd {spec.cwd} && {env_prefix}{spec.command} {' '.join(spec.args)}"
```

### The Solution

Nushell treats shell data as structured. We can generate nushell commands from ProcessSpec without escaping:

```python
class ShellBackend(Protocol):
    def build_command(self, spec: ProcessSpec) -> str: ...

class BashBackend:
    def build_command(self, spec: ProcessSpec) -> str:
        parts = []
        if spec.cwd:
            parts.append(f"cd {shlex.quote(spec.cwd)}")
        for k, v in spec.env:
            parts.append(f"export {k}={shlex.quote(v)}")
        cmd = spec.command
        if spec.args:
            cmd += " " + " ".join(shlex.quote(a) for a in spec.args)
        parts.append(cmd)
        return " && ".join(parts)

class NushellBackend:
    def build_command(self, spec: ProcessSpec) -> str:
        env_record = ", ".join(f"{k}: {json.dumps(v)}" for k, v in spec.env)
        nu_script = f"""
with-env {{ {env_record} }} {{
    cd {json.dumps(spec.cwd or ".")}
    ^{spec.command} {" ".join(json.dumps(a) for a in spec.args)}
}}
"""
        return f"nu -c {shlex.quote(nu_script)}"
```

### Nushell also helps with introspection

```python
# With bash - parse text in Python
def get_gpu_info_bash(session: Session) -> list[dict]:
    result = session.exec("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader")
    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split(", ")
        gpus.append({"index": int(parts[0]), "name": parts[1], ...})
    return gpus

# With nushell - structured data throughout
def get_gpu_info_nushell(session: Session) -> list[dict]:
    result = session.exec("""
        nu -c 'nvidia-smi --query-gpu=index,name,memory.total --format=csv | from csv | to json'
    """)
    return json.loads(result.stdout)
```

---

## File Structure

```
bifrost/
├── __init__.py              # Public API exports
├── session.py               # Session class (Layers 0-2)
├── types.py                 # ProcessSpec, ExecResult, PythonEnvState
├── handles.py               # ServerHandle, JobHandle
├── shell.py                 # ShellBackend, BashBackend, NushellBackend
├── transfer.py              # File transfer helpers (git sync, rsync)
├── environment.py           # setup_python_env implementation
├── introspection.py         # GPU info, hardware queries
└── inference/               # ML presets (Layer 4)
    ├── __init__.py
    ├── sglang.py
    └── vllm.py
```

### What Gets Deleted

After merge, kerbal becomes unnecessary:

```
kerbal/                      # DELETE - absorbed into bifrost
├── python_env.py            # → bifrost/environment.py
├── server.py                # → bifrost/session.py (serve)
├── job.py                   # → bifrost/session.py (submit)
├── tmux.py                  # → bifrost/session.py (internal)
├── job_monitor.py           # → bifrost/handles.py (JobHandle.stream_logs)
└── inference/               # → bifrost/inference/
```

---

## Integration with Broker

Broker's responsibility after the merge:

```python
# Broker returns hardware handle - no software concerns
machine = broker.acquire(
    gpu=GPURequest(type="A100", count=4),
    capabilities=["docker"],  # Optional
)

@dataclass(frozen=True)
class MachineHandle:
    host: str
    port: int
    user: str
    ssh_key_path: str
    gpu_type: str
    gpu_count: int
    cuda_driver: str

# Bifrost takes it from there
session = bifrost.connect(machine)
```

Broker no longer picks Docker images. Software stack is controlled by bifrost (via `setup_python_env`, Docker commands, or nix).

---

## What's NOT in Bifrost (Miniray's Domain)

Bifrost handles: **"Run processes on remote machines"**

Miniray handles: **"Processes talking to each other"**

| Concern | Owner | Status |
|---------|-------|--------|
| Process scope (wd, env, args) | Bifrost | ✅ `ProcessSpec` |
| Where process lives (host) | Broker | ✅ `MachineHandle` |
| Start process remotely | Bifrost | ✅ `session.run()`, `session.submit()` |
| Monitor process | Bifrost | ✅ `JobHandle`, `ServerHandle` |
| Data handoff between processes | Miniray | ❌ Not designed yet |
| Object references | Miniray | ❌ Not designed yet |
| Distributed scheduler | Neither | ❌ Not planned |

### What Ray does that we don't (yet)

| Ray Feature | Our Status |
|-------------|-----------|
| Object Store (Plasma) | None - no shared memory / object refs |
| Distributed Scheduler | None - no automatic task placement |
| Actor Model | None - no stateful remote objects |
| Fault Tolerance | Partial - tmux survives, no auto-restart |
| Task Graph | None - no DAG of dependencies |

These would be miniray's domain if we need them.

---

## Open Questions

1. **Async support?** Current bifrost has both sync and async clients. Keep both?

2. **Nushell installation:** Auto-install if missing, or just fall back to bash?

3. **Job persistence format:** Where to store job metadata? `~/.bifrost/jobs/`?

4. **Health check flexibility:** HTTP-only, or support TCP/custom commands?

---

## Migration Path

### Phase 1: Add New API to Bifrost
- Add `ProcessSpec` and types
- Add `session.run(spec)` method
- Add `session.serve()` and `session.submit()`
- Keep old kerbal working

### Phase 2: Migrate Kerbal Features
- Move `setup_python_env` to bifrost
- Move inference presets to `bifrost/inference/`
- Kerbal becomes thin wrapper (deprecation path)

### Phase 3: Update Consumers
- Update deploy scripts to use new API
- Update integration tests

### Phase 4: Delete Kerbal
- Remove kerbal package

---

## Appendix A: Actual Usage Patterns

Analysis of how broker/bifrost/kerbal are actually used in the codebase.

### Pattern 1: Simple Deploy + Stream (SFT)

**File:** `rollouts/examples/sft/base_config.py`

```python
# Current
bifrost = BifrostClient(gpu.ssh_connection_string(), ssh_key_path)
bifrost.push(workspace_path=workspace, bootstrap_cmd=bootstrap)
for line in bifrost.exec_stream(cmd):
    print(line, end="")

# Proposed
session = connect(gpu)
workspace = session.push("~/.bifrost/workspaces/rollouts", bootstrap=bootstrap)
for line in session.exec_stream(cmd):
    print(line, end="")
```

### Pattern 2: Advanced Deploy + Monitor (RL)

**File:** `rollouts/examples/rl/base_config.py`

```python
# Current - requires kerbal
from kerbal.tmux import start_tmux_session
from kerbal.job_monitor import stream_log_until_complete

session, err = start_tmux_session(client=bifrost, session_name="rl-training", ...)
success, exit_code, err = stream_log_until_complete(bifrost, monitor_config, on_line=queue_line)

# Proposed - unified in bifrost
job = session.submit(ProcessSpec(...), name="rl-training")
for line in job.stream_logs(timeout=7200):
    queue_line(line)
```

### Pattern 3: Complex Orchestration (Outlier Features)

**File:** `dev/outlier-features/deploy.py`

```python
# Current - manual tmux, custom polling
tmux_cmd = f"tmux new-session -d -s outlier-analysis '{cmd}'"
bifrost_client.exec(tmux_cmd)
for i in range(max_iterations):
    result = bifrost_client.exec(check_cmd)
    if "COMPLETE" in result.stdout:
        break

# Proposed
job = session.submit(ProcessSpec(...), name="outlier-analysis")
exit_code = job.wait(timeout=2700)
```

### Key Insight: Kerbal is Just "Bifrost + tmux"

Kerbal adds:
- `start_tmux_session()` → wrap command in tmux
- `stream_log_until_complete()` → tail log file
- `setup_python_env()` → create venv

None require a separate library.

---

## Appendix B: Original Questions Mapping

From the discussion about "what is a unix process":

| Question | Owner | Implementation |
|----------|-------|----------------|
| 1.1 Process scope (wd, env, args) | Bifrost | `ProcessSpec` (env passed each time, not cached) |
| 1.2 Where it lives (host, perms) | Broker | `MachineHandle` |
| 1.3 Reflection (GPU, bandwidth) | Bifrost | `introspection.py` + nushell |
| 2. Start process remotely | Bifrost | `session.run()`, `session.serve()` |
| 2.1 Head node death | Bifrost | tmux survives, `session.get_job()` reconnects |
| 3. Talk to it (liveness) | Bifrost | `ServerHandle.is_healthy()` |
| 4. Coherent data handoff | Miniray | Not yet designed |
| 5. Do this remotely | Miniray | Not yet designed |

---

## Appendix C: SSH exec_command() Behavior

Each `session.exec()` spawns a **new shell process**:

```python
# This is how paramiko works
stdin, stdout, stderr = ssh_client.exec_command("export FOO=bar && echo $FOO")
# FOO exists in this shell ^

stdin, stdout, stderr = ssh_client.exec_command("echo $FOO")
# FOO is gone - different shell process ^
```

The shell is **non-login, non-interactive**:
- Does NOT source `~/.bashrc`
- Does NOT source `~/.bash_profile`
- Only gets minimal environment from sshd

This is why env vars must be passed explicitly each time, not cached on Session.
