# Midas/Kerbal Refactor - Handoff Document

## Current Problem

**midas** is mixing two concerns:
1. Infrastructure setup ("Is uv installed?")
2. Script execution setup ("Install deps for THIS script")

This violates separation of concerns. We need to split these.

## Core Insight

We're not "setting up a project" - we're **running a script that needs dependencies**.

Different scripts on different nodes might need different deps:
- Node 1: `train.py` needs `["torch", "wandb", "accelerate"]`
- Node 2: `infer.py` needs `["vllm", "sglang"]`

Same codebase, different runtime dependencies.

## Proposed Separation

### midas/ - Infrastructure Bootstrap
**Concern:** "Can we run Python tooling at all?"

Responsibilities:
- Check if `uv` exists
- Install `uv` if missing
- Ensure `uv` is in PATH
- Verify basic Python availability

**Does NOT:**
- Generate pyproject.toml
- Run `uv sync`
- Install project dependencies
- Know about YOUR scripts

**Example API:**
```python
from midas import ensure_uv

# Just makes sure uv is installed and available
ensure_uv(client)
```

### kerbal/ - Script Execution Orchestration
**Concern:** "Run THIS script with THESE dependencies"

Responsibilities:
- Generate pyproject.toml from DependencyConfig
- Run `uv sync --extra X` to install deps
- Execute scripts in the venv
- Manage environment variables for execution

**Example API:**
```python
from kerbal import setup_script_deps, run_script
from midas import DependencyConfig  # DependencyConfig stays in midas - it's data

deps = DependencyConfig(
    project_name="training-env",
    dependencies=["torch>=2.0"],
    extras={
        "training": ["wandb", "accelerate"],
        "inference": ["vllm", "sglang"],
    }
)

# Setup dependencies for training
setup_script_deps(client, workspace, deps, install_extras=["training"])

# Run the script in that environment
run_script(client, workspace, "train.py", env_vars={"CUDA_VISIBLE_DEVICES": "0,1"})
```

## Architecture

```
┌─────────────────────────────────────┐
│  User Code                          │
│  - train.py needs torch, wandb      │
│  - infer.py needs vllm              │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  kerbal/ (script execution)         │
│  - Generate pyproject.toml          │
│  - Run uv sync --extra X            │
│  - Execute script in venv           │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  midas/ (infrastructure)            │
│  - Ensure uv exists                 │
│  - Verify Python available          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  bifrost/ (SSH primitives)          │
│  - exec(), push(), download()       │
└─────────────────────────────────────┘
```

## What Moves Where

### Stays in midas/
- `DependencyConfig` dataclass (it's data, not execution)
- `ensure_uv()` function (infrastructure check)
- Basic Python verification

### Moves to kerbal/
- `pyproject_gen.py` → `kerbal/python_env.py`
- `UvBackend.bootstrap()` split:
  - uv installation → `midas.ensure_uv()`
  - pyproject + uv sync → `kerbal.setup_script_deps()`
- `UvBackend.run_in_env()` → `kerbal.run_script()`

### New kerbal/ API

```python
# kerbal/python_env.py

def setup_script_deps(
    client: BifrostClient,
    workspace: str,
    dependencies: DependencyConfig,
    install_extras: list[str] | None = None,
) -> None:
    """Setup Python dependencies for script execution.

    Generates pyproject.toml and runs uv sync.

    Args:
        client: BifrostClient instance
        workspace: Remote workspace path
        dependencies: DependencyConfig specifying deps
        install_extras: List of extra groups to install (e.g., ["training", "dev"])

    Example:
        deps = DependencyConfig(
            project_name="training",
            dependencies=["torch>=2.0"],
            extras={"training": ["wandb"], "inference": ["vllm"]},
        )
        setup_script_deps(client, workspace, deps, install_extras=["training"])
    """
    ...

def run_script(
    client: BifrostClient,
    workspace: str,
    script: str,
    env_vars: dict[str, str] | None = None,
) -> CommandResult:
    """Run a Python script in the uv venv.

    Args:
        client: BifrostClient instance
        workspace: Remote workspace path
        script: Python script to run (e.g., "train.py --epochs 100")
        env_vars: Environment variables to export

    Returns:
        CommandResult with exit_code, stdout, stderr

    Example:
        result = run_script(
            client, workspace, "train.py --epochs 100",
            env_vars={"CUDA_VISIBLE_DEVICES": "0,1"}
        )
    """
    ...
```

### New midas/ API

```python
# midas/uv.py (simplified)

def ensure_uv(client: BifrostClient) -> None:
    """Ensure uv is installed and available.

    Checks if uv exists, installs if missing, ensures it's in PATH.
    Pure infrastructure - no knowledge of project dependencies.

    Args:
        client: BifrostClient instance

    Example:
        ensure_uv(client)  # Now uv is ready to use
    """
    ...

def verify_python(client: BifrostClient) -> bool:
    """Verify Python is available on remote.

    Returns:
        True if python3 exists and is runnable
    """
    ...
```

## Migration Path

### Phase 1: Create new kerbal/ functions (non-breaking)
1. Add `kerbal/python_env.py` with new API
2. Move pyproject generation to kerbal
3. Keep old `UvBackend` working for now

### Phase 2: Simplify midas/ (non-breaking)
1. Add `midas/uv.py` with `ensure_uv()`
2. Keep `UvBackend` as a wrapper around new APIs

### Phase 3: Update callers
1. Update dev/integration_training to use new API
2. Update other experiments

### Phase 4: Remove old code
1. Delete `UvBackend` class
2. Clean up midas/backends/

## Benefits

1. **Clear separation**: Infrastructure vs. execution
2. **Flexible deps**: Different nodes can have different extras
3. **No "project" assumption**: Just "run this script with these deps"
4. **Composable**: Mix and match midas + kerbal functions
5. **Explicit**: `install_extras=["training", "dev"]` vs. mysterious `extra="dev-speedrun"`

## Key Design Decisions

### Why DependencyConfig stays in midas?
It's **data**, not execution logic. It describes dependencies, doesn't install them.

### Why pyproject generation moves to kerbal?
It's execution orchestration - "prepare to run THIS script". Not infrastructure.

### Why split install_extras into a list?
Because different nodes might need multiple extra groups:
```python
# Node 1: Training with dev tools
install_extras=["training", "dev"]

# Node 2: Just inference
install_extras=["inference"]
```

### Why keep uv installation in midas?
It's infrastructure - "ensure the Python package manager exists". Like ensuring `git` or `make` is installed.

## Questions to Resolve

1. Should `run_script()` automatically call `ensure_uv()` or require caller to do it?
   - **Proposal**: Caller does it explicitly. More granular control.

2. Should we support multiple venvs per workspace?
   - **Proposal**: Not yet. One `.venv` per workspace. Keep it simple.

3. Should `setup_script_deps()` be idempotent (skip if pyproject exists)?
   - **Proposal**: Always regenerate. Dependencies might have changed.

## Implementation Notes

- All functions < 70 lines (Tiger Style)
- Tuple returns for errors (not exceptions)
- Assert preconditions
- Use TYPE_CHECKING for bifrost imports

## Example End-to-End Usage

```python
from bifrost import BifrostClient
from midas import ensure_uv, DependencyConfig
from kerbal import push_code, setup_script_deps, start_tmux_session

# Connect
client = BifrostClient("root@gpu:22", ssh_key_path="~/.ssh/id_rsa")

# Push code
workspace = push_code(client, "dev/integration_training")

# Ensure infrastructure
ensure_uv(client)

# Setup dependencies for THIS run
deps = DependencyConfig(
    project_name="training",
    dependencies=["torch>=2.0", "transformers"],
    extras={"training": ["wandb", "accelerate"]},
)
setup_script_deps(client, workspace, deps, install_extras=["training"])

# Run training in tmux
start_tmux_session(
    client, "training",
    "cd dev/integration_training && .venv/bin/python train.py",
    env_vars={"CUDA_VISIBLE_DEVICES": "0,1"}
)
```

Clean, explicit, composable!
