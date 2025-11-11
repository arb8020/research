# deploy - Environment Setup & Orchestration

Environment setup and deployment orchestration for remote execution.

This package provides the missing layer between `bifrost` (SSH primitives) and your application code, handling the complexity of "SSH connection → working Python environment".

## Architecture

```
broker/    - GPU provisioning (abstracts RunPod, Vast, etc.)
bifrost/   - SSH primitives (connect, exec, push, download)
deploy/    - Environment setup + orchestration (THIS PACKAGE)
miniray/   - Multi-node coordination (future)
```

## Quick Start

### High-level API (simplest)

```python
from bifrost import BifrostClient
from deploy import deploy_and_run

bifrost = BifrostClient("root@host:22", "~/.ssh/id_ed25519")

# Deploy project and run command (one function call)
result = deploy_and_run(
    bifrost,
    local_path="dev/speedrun",
    extra="dev-speedrun",
    command="python train.py",
)

if result.success:
    print("Training complete!")
```

### Mid-level API (more control)

```python
from deploy import deploy_project, run_in_project

# Deploy project (push code + setup environment)
workspace = deploy_project(
    bifrost,
    local_path="dev/integration_training",
    extra="dev-integration-training",
)

# Run multiple commands in the deployed environment
run_in_project(bifrost, workspace, "python train.py configs/01_baseline.py")
run_in_project(bifrost, workspace, "python evaluate.py")
```

### Low-level API (full control)

```python
from deploy.api import push_code, bootstrap_env, start_tmux_session, sync_results

# Step by step control
workspace = push_code(bifrost, "dev/speedrun")
bootstrap_env(bifrost, workspace, "dev-speedrun")

# Start training in tmux (detached)
start_tmux_session(
    bifrost,
    session_name="training",
    command="python train.py",
    workspace=workspace,
    log_file="training.log"
)

# Later: sync results back
sync_results(bifrost, f"{workspace}/results", "./local_results")
```

## Environment Backends

The `deploy` package uses pluggable backends for environment setup. This follows Casey Muratori's principle of **decoupling** - the deployment API doesn't care HOW the environment is set up.

### UvBackend (Production)

Current production backend using UV for Python environment management:

```python
from deploy.backends import UvBackend

backend = UvBackend()
workspace = deploy_project(bifrost, "dev/speedrun", "dev-speedrun", backend=backend)
```

**What it does:**
1. Installs UV if not present
2. Ensures UV is in PATH
3. Runs `uv sync --extra <extra>` to install dependencies
4. Verifies Python venv is working

### NixBackend (Future - Stub)

For reproducible environments using Nix:

```python
from deploy.backends import NixBackend

# Future usage (not yet implemented)
backend = NixBackend(flake_ref=".#speedrun")
workspace = deploy_project(bifrost, "dev/speedrun", "dev-speedrun", backend=backend)
```

**Planned benefits:**
- Fully reproducible (pins everything, including system packages)
- Declarative environment definition
- Handles non-Python dependencies (CUDA, system libs)

See `deploy/backends/nix.py` for implementation plan.

### DockerBackend (Future - Stub)

For containerized environments:

```python
from deploy.backends import DockerBackend

# Future usage (not yet implemented)
backend = DockerBackend(image="my-research:latest", gpu=True)
workspace = deploy_project(bifrost, "dev/speedrun", "dev-speedrun", backend=backend)
```

**Planned benefits:**
- Industry-standard containers
- Complete isolation from host
- Easy environment sharing (push/pull images)

See `deploy/backends/docker.py` for implementation plan.

## Design Principles

This package follows three code style guides from `docs/code_style/`:

### 1. Casey Muratori (API Design)

From `code_reuse_casey_muratori.md`:

- **Granularity:** Small, composable functions (push_code, bootstrap_env, start_tmux)
- **Redundancy:** Multiple API levels (high/mid/low) for different use cases
- **Decoupling:** Backends are protocols, not concrete implementations
- **No retention:** Immediate mode - no hidden state between calls

### 2. Tiger Style (Safety)

From `tiger_style_safety.md`:

- **Functions < 70 lines:** Every function is under 70 lines
- **Assert preconditions:** Every function asserts its inputs
- **Explicit control flow:** No magic, no hidden behavior
- **Fail fast:** Errors are detected immediately

### 3. Ray-Ready Design

From `ray_design.txt`:

- **Use Protocols, Not Concrete Classes:** `EnvBackend` is a Protocol
- **Dependency Injection:** Pass backends to functions, don't create them
- **Serializable Configuration:** Backends are simple, data-driven

## Example: Simplified deploy.py

Before (400+ lines of duplicated code):

```python
# dev/speedrun/deploy.py (OLD)
def main():
    # 100 lines of GPU provisioning code
    # 100 lines of code pushing + bootstrap
    # 100 lines of tmux session management
    # 100 lines of result syncing
    ...
```

After (150 lines using deploy API):

```python
# dev/speedrun/deploy.py (NEW)
from deploy import deploy_and_run
from bifrost import BifrostClient

def main():
    args = parse_args()
    config = load_config(args.config)

    bifrost = BifrostClient(args.ssh_connection, args.ssh_key)

    result = deploy_and_run(
        bifrost,
        local_path="dev/speedrun",
        extra="dev-speedrun",
        command=f"python train.py {args.config}",
        detached=args.detached,
    )
```

## Multi-Node (Future)

For distributed training or multi-service coordination:

```python
# Explicit multi-node orchestration
nodes = [
    BifrostClient("root@host1:22"),
    BifrostClient("root@host2:22"),
]

# Deploy to all nodes
for node in nodes:
    deploy_project(node, "dev/integration_training", "dev-integration-training")

# Start distributed training with explicit RANK
for rank, node in enumerate(nodes):
    run_in_project(
        node,
        workspace,
        f"torchrun --rank={rank} --world-size={len(nodes)} train.py"
    )
```

Future coordinator abstraction (like miniray) can be built on top of this once we have real multi-node use cases.

## Testing

```bash
# Run tests (when we add them)
uv run pytest deploy/tests/

# Type check
uv run ty deploy/
```

## Contributing

When adding a new backend:

1. Create `deploy/backends/your_backend.py`
2. Implement the `EnvBackend` protocol
3. Add comprehensive docstrings explaining the approach
4. Update this README with usage example

See existing backends for examples.
