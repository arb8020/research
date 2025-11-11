# Deploy Package - Context Handoff

## What Was Just Built

Created `deploy/` package to handle "SSH connection → working Python environment". Separates bifrost (dumb SSH) from environment orchestration.

**Status:** Core structure done, UvBackend working. Need to add GPU checking and env var support.

## Architecture

```
bifrost/   - SSH primitives (exec, push, download)
deploy/    - Environment setup + orchestration (THIS PACKAGE)
  - backends/uv.py (✅ production ready)
  - backends/nix.py, docker.py (stubs for future)
  - api.py (needs GPU + env var functions added)
```

## Files to Read

### Code Style (MUST READ - in ~/research/docs/code_style/)
1. **`casey_muratori.md`** - API design principles (granular, redundant, decoupled, no retention)
2. **`tiger_style_safety.md`** - Functions < 70 lines, assert preconditions, explicit control flow
3. **`my_notes.md`** - Tuple returns for errors, no try/except in internal code
4. **`ray_design.txt`** - Use protocols, dependency injection

### Implementation Files
1. **`deploy/env.py`** - EnvBackend protocol definition
2. **`deploy/api.py`** - Current high/mid/low level functions (where to add GPU/env var support)
3. **`deploy/backends/uv.py`** - Example of proper backend implementation (< 70 lines per function)
4. **`deploy/README.md`** - Full docs on what exists

### For Context on Original Problem
- `dev/integration_training/deploy.py:178` - See old bootstrap approach (duplicated 4x)
- `grep -r "uv sync --extra" dev/` - See how bootstrap is currently done
- `grep -r "CUDA_VISIBLE_DEVICES" dev/` - See how env vars currently handled

## What Needs Implementation

User wants to add to `deploy/api.py`:

### 1. GPU Checking Functions
```python
check_gpus_available(bifrost, gpu_ids, memory_threshold_mb) -> tuple[bool, str]
wait_for_gpus(bifrost, gpu_ids, timeout_sec) -> bool
```

**Implementation notes:**
- Use `bifrost.exec("nvidia-smi ...")` to check GPU state
- Parse nvidia-smi output (CSV format)
- Check memory usage and utilization
- Tiger Style: < 70 lines per function, assert preconditions

### 2. Environment Variable Support
Add `env_vars: dict[str, str] | None = None` parameter to:
- `run_in_env()`
- `start_tmux_session()`
- `deploy_and_run()`

**Implementation notes:**
- Build `export KEY='VALUE' && ...` prefix
- Pass through to bifrost.exec()
- Common vars: HF_TOKEN, CUDA_VISIBLE_DEVICES, WANDB_API_KEY

## Key Constraints

### Design Principles (from code style docs)
- **Casey:** Granular functions, each < 70 lines. Provide both high-level and low-level APIs
- **Tiger:** Assert preconditions. Explicit control flow. No hidden magic
- **Tuple returns:** Use `(result, error)` not exceptions for internal code
- **Protocols:** EnvBackend is a protocol - don't couple to concrete implementations

### Bifrost Boundary
**bifrost does:** exec(), push(), download_files()
**bifrost does NOT:** Know about Python, tmux, GPUs, or env vars

**deploy does:** Everything between SSH and running code
- Environment setup (uv, venv)
- GPU checking
- Process management (tmux)
- Env vars

### Code Quality
- Functions < 70 lines (Tiger Style)
- Docstrings with examples
- Type hints
- Assert preconditions
- No try/except unless wrapping external calls (see `my_notes.md`)

## Implementation Strategy

1. Read the 4 code style docs (30 min)
2. Read `deploy/api.py` to understand current patterns (10 min)
3. Add GPU checking functions following UvBackend style (< 70 lines each)
4. Add env_vars parameter to existing functions (careful not to break current API)
5. Update examples in `deploy/examples/simple_deployment.py`
6. Test with: `uv run python -c "from deploy.api import check_gpus_available; print('✅')"`

## User's Mental Model

> "I go from 'node that may not have uv' to 'node with all my code + packages' and easily run things in tmux"

User understands:
- bifrost = SSH primitives
- deploy = orchestration (env setup, GPU checks, tmux, env vars)

User wants simple high-level API but with escape hatches for control.

## Grep Keywords for Context

```bash
# See old bootstrap patterns:
grep -r "uv sync" dev/*/deploy.py

# See env var usage:
grep -r "CUDA_VISIBLE_DEVICES" dev/

# See GPU checking attempts:
grep -r "nvidia-smi" dev/

# See tmux patterns:
grep -r "tmux new-session" dev/
```

## Questions to Ask User (if needed)

1. GPU checking: Should it check utilization % too, or just memory?
2. wait_for_gpus: How often to poll? (suggest 30 sec intervals)
3. Env vars: Any special handling for secrets? (e.g., HF_TOKEN)

## Notes

- There's a pre-existing issue with rollouts/ package (PROJECT_ROOT error) - ignore it
- Lock file regenerated successfully with deploy/ included
- User might rename package later but wants it functional first
