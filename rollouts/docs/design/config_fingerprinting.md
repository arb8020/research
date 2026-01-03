# Config Fingerprinting for Reproducibility

**DRI:** chiraag
**Status:** Proposed → Ready for Implementation
**Priority:** Soon
**Inspired by:** [nmoe](https://github.com/Noumena-Network/nmoe) config fingerprinting

## Problem

When reviewing evaluation results, sessions, or training runs, it's hard to know exactly what configuration was used:

1. **What environment/tools?** Which tools were available? What sandbox settings?
2. **What model/endpoint?** Provider, model name, temperature, etc.
3. **What code version?** Has the score function, environment, or agent logic changed?
4. **Are these runs comparable?** Same config = same fingerprint → fair comparison

Currently `EvalReport.config` stores ad-hoc config, sessions store some header info, but there's no:
- Deterministic hash for quick "are these equivalent?" checks
- Code version tracking (git sha)
- Function source tracking (detect score_fn changes even if name stays same)
- Dataset identity

## Scope

**All three domains:**
- Evals (`EvalReport`)
- Sessions (interactive rollouts)
- Training/RL runs (future)

## Design Decisions (from interview)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tool capture | **Introspect at runtime** | Capture from first sample's env (see open question below) |
| Function identity | **Name only** | Just qualified name - git SHA tracks code changes |
| Lambdas | **Require named functions** | Error if `score_fn` is a lambda - can't identify reliably |
| Dirty git | **Error by default** | Refuse to run eval with uncommitted changes (--allow-dirty to override) |
| Dataset identity | **Path + checksum** | Include dataset path and content hash |
| Structure | **Shared core + adapters** | Core utils in `fingerprint.py`, domain-specific wrappers |
| Logging | **Silent** | Only in report.json/session files, no console spam |
| Mismatch behavior | **Warn** | Print warning on resume if config changed, but continue |
| run_config.handle_stop | **Include** | Hash the stop handler using callable_identity |

### Open: Tool Declaration Without Instantiation

Current problem: To get tools, we need to call `environment_factory()` which may be expensive (Docker, sandbox setup).

**Needs more design work.** Options to explore:
1. Class-level `TOOLS` attribute on Environment subclasses
2. `@with_tools([...])` decorator on factory functions  
3. Explicit `tools: list[str]` field in EvalConfig
4. Capture from first sample's env (current plan, but adds latency)

For now: capture tools from first sample's environment during eval. Revisit if this becomes a bottleneck.

## Architecture

```
rollouts/fingerprint.py          # Core utilities
├── canonical_json()             # Deterministic JSON encoding
├── git_info()                   # SHA + dirty status  
├── require_clean_git()          # Error if uncommitted changes
├── callable_name()              # Module.qualname for functions
├── file_checksum()              # Fast content hash for datasets
└── fingerprint_short()          # Short display format helper

rollouts/evaluation.py           # Uses fingerprint_eval()
rollouts/sessions.py             # Uses fingerprint_session()
rollouts/training/...            # Future: fingerprint_training()
```

## Core Module: `rollouts/fingerprint.py`

```python
"""Config fingerprinting for reproducibility.

Provides stable hashes of configuration to enable:
- Quick comparison of eval/session configs ("are these equivalent?")
- Code version tracking (git sha + dirty status)
- Function identity (detect changes to score_fn even if name unchanged)
- Dataset identity (path + content checksum)

Design: Shared core utilities, domain-specific wrappers.
"""
import hashlib
import inspect
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# ── Core Utilities ──────────────────────────────────────────────────────────


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding for hashing.
    
    - Sorted keys
    - No whitespace
    - ASCII only
    - Handles non-JSON types via str()
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def hash_str(s: str) -> str:
    """SHA-256 hash of string, first 16 chars."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def git_info() -> tuple[str, bool]:
    """Get current git hash (8 chars) and dirty status.
    
    Returns ("unknown", False) if not in a git repo or git unavailable.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()[:8]
        
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip())
        
        return sha, dirty
    except Exception:
        return "unknown", False


def require_clean_git(*, allow_dirty: bool = False) -> tuple[str, bool]:
    """Get git info, erroring if there are uncommitted changes.
    
    Args:
        allow_dirty: If True, warn instead of error on dirty state
        
    Returns:
        (git_sha, git_dirty) tuple
        
    Raises:
        RuntimeError: If git is dirty and allow_dirty=False
    """
    sha, dirty = git_info()
    
    if dirty and not allow_dirty:
        raise RuntimeError(
            "Uncommitted changes detected. Eval results may not be reproducible.\n"
            "Options:\n"
            "  1. Commit your changes: git commit -am 'wip'\n"
            "  2. Override with --allow-dirty flag\n"
            f"Git SHA: {sha} (dirty)"
        )
    
    if dirty and allow_dirty:
        import sys
        print(
            f"⚠ Running with uncommitted changes (git_dirty=True). "
            f"Results may not be reproducible.",
            file=sys.stderr
        )
    
    return sha, dirty


def callable_name(fn: Callable, *, allow_lambda: bool = False) -> str:
    """Get qualified name for a callable.
    
    Returns module.qualname (e.g., 'evals.gsm8k.score_fn').
    
    Note: We don't hash source code. Git SHA is the authoritative code version.
    If you have uncommitted changes, the eval will error by default.
    
    Raises:
        ValueError: If fn is a lambda and allow_lambda=False
    """
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
    name = f"{module}.{qualname}" if module else qualname
    
    # Reject lambdas for critical functions (score_fn, prepare_messages)
    if not allow_lambda and getattr(fn, "__name__", "") == "<lambda>":
        raise ValueError(
            f"Lambdas cannot be fingerprinted reliably. "
            f"Define a named function instead of: {name}"
        )
    
    return name


def file_checksum(path: Path, max_bytes: int = 1024 * 1024) -> str:
    """Fast checksum of file contents.
    
    Hashes first max_bytes (default 1MB) for speed on large files.
    Returns "missing" if file doesn't exist.
    """
    try:
        with open(path, "rb") as f:
            content = f.read(max_bytes)
        return hashlib.sha256(content).hexdigest()[:16]
    except (OSError, IOError):
        return "missing"


# ── Fingerprint Helpers ─────────────────────────────────────────────────────

# Note: No Fingerprint class. Just return dicts and use helper functions.
# "Don't abstract until you've done something twice."

def fingerprint_short(fp: dict) -> str:
    """Short format: 'abc123@def456*' (dirty marked with *)"""
    dirty = "*" if fp.get("git_dirty") else ""
    config_hash = fp.get("config_hash", "????????")[:8]
    git_sha = fp.get("git_sha", "unknown")
    return f"{config_hash}@{git_sha}{dirty}"


# ── Eval Fingerprint ────────────────────────────────────────────────────────


def fingerprint_eval(
    config: "EvalConfig",
    tools: list[str] | None = None,
    dataset_path: Path | None = None,
    allow_dirty: bool = False,
) -> dict:
    """Compute fingerprint for an evaluation run.
    
    Args:
        config: EvalConfig with endpoint, score_fn, etc.
        tools: Actual tool names from environment (introspected at runtime)
        dataset_path: Path to dataset file for checksum
        allow_dirty: If False (default), error on uncommitted git changes
    
    Includes in hash:
        - endpoint (provider, model, params - minus api_key)
        - score_fn name
        - prepare_messages name
        - environment_factory name (if present)
        - handle_stop name (if present)
        - max_samples
        - dataset checksum (if path provided)
        
    Raises:
        RuntimeError: If git has uncommitted changes and allow_dirty=False
    """
    from dataclasses import asdict
    
    # Require clean git by default
    git_sha, git_dirty = require_clean_git(allow_dirty=allow_dirty)
    
    cfg = {}
    
    # Endpoint (sanitize API key)
    endpoint_dict = asdict(config.endpoint)
    endpoint_dict.pop("api_key", None)
    cfg["endpoint"] = endpoint_dict
    
    # Function identities (names only - git SHA tracks code)
    cfg["score_fn"] = callable_name(config.score_fn)
    cfg["prepare_messages"] = callable_name(config.prepare_messages)
    
    if config.environment_factory:
        cfg["environment_factory"] = callable_name(config.environment_factory, allow_lambda=True)
    
    # Execution params that affect results
    cfg["max_samples"] = config.max_samples
    
    # Stop handler (if present in run_config)
    if config.run_config and config.run_config.handle_stop:
        cfg["handle_stop"] = callable_name(config.run_config.handle_stop, allow_lambda=True)
    
    # Dataset checksum
    if dataset_path:
        cfg["dataset_checksum"] = file_checksum(dataset_path)
        cfg["dataset_path"] = str(dataset_path)
    
    # Compute hash
    config_hash = hash_str(canonical_json(cfg))
    
    # Metadata for display
    metadata = {
        "endpoint_summary": f"{config.endpoint.provider}/{config.endpoint.model}",
        "tools": tools or [],
        "dataset_path": str(dataset_path) if dataset_path else None,
    }
    
    return {
        "config_hash": config_hash,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "domain": "eval",
        **metadata,
    }


# ── Session Fingerprint ─────────────────────────────────────────────────────


def fingerprint_session(
    model: str,
    env: str,
    tools: list[str],
    system_prompt: str | None = None,
) -> dict:
    """Compute fingerprint for an interactive session.
    
    Args:
        model: Model identifier (e.g., "anthropic/claude-sonnet-4-5-20250929")
        env: Environment name (e.g., "coding", "git", "none")
        tools: List of tool names available
        system_prompt: Custom system prompt (hashed, not stored)
    """
    cfg = {
        "model": model,
        "env": env,
        "tools": sorted(tools),
    }
    
    if system_prompt:
        cfg["system_prompt_hash"] = hash_str(system_prompt)
    
    config_hash = hash_str(canonical_json(cfg))
    git_sha, git_dirty = git_info()
    
    metadata = {
        "model": model,
        "env": env,
        "tools": sorted(tools),
    }
    
    return {
        "config_hash": config_hash,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "domain": "session",
        **metadata,
    }


# ── Comparison Utilities ────────────────────────────────────────────────────


def compare_fingerprints(a: dict, b: dict) -> dict[str, Any]:
    """Compare two fingerprints for reproducibility analysis."""
    return {
        "same_config": a.get("config_hash") == b.get("config_hash"),
        "same_code": a.get("git_sha") == b.get("git_sha"),
        "a": fingerprint_short(a),
        "b": fingerprint_short(b),
    }


def warn_if_changed(saved: dict, current: dict, context: str = "") -> None:
    """Print warning if fingerprint changed (for resume scenarios).
    
    Called when resuming a session or continuing an eval.
    """
    if not saved:
        return  # No saved fingerprint (legacy data)
    
    saved_hash = saved.get("config_hash", "")
    current_hash = current.get("config_hash", "")
    if saved_hash and current_hash and saved_hash != current_hash:
        import sys
        print(f"⚠ Config changed since {context} was created", file=sys.stderr)
        print(f"  Saved:   {fingerprint_short(saved)}", file=sys.stderr)
        print(f"  Current: {fingerprint_short(current)}", file=sys.stderr)
    
    saved_git = saved.get("git_sha", "")
    current_git = current.get("git_sha", "")
    if saved_git and current_git and saved_git != current_git and saved_hash == current_hash:
        import sys
        print(f"ℹ Code version changed: {saved_git} → {current_git}", file=sys.stderr)
```

## CLI Behavior

```bash
# Default: errors if git is dirty
$ rollouts eval --dataset gsm8k.jsonl
Error: Uncommitted changes detected. Eval results may not be reproducible.
Options:
  1. Commit your changes: git commit -am 'wip'
  2. Override with --allow-dirty flag
Git SHA: abc12345 (dirty)

# Override to run anyway (fingerprint shows git_dirty=True)
$ rollouts eval --dataset gsm8k.jsonl --allow-dirty
⚠ Running with uncommitted changes (git_dirty=True). Results may not be reproducible.
Starting evaluation...

# Clean git state: runs normally
$ git commit -am 'ready for eval'
$ rollouts eval --dataset gsm8k.jsonl
Starting evaluation...
# fingerprint: {"config_hash": "...", "git_sha": "abc12345", "git_dirty": false, ...}
```

For **sessions**, git dirty is allowed by default (interactive work often has uncommitted changes), but fingerprint still records the state.

## Integration: Evaluation

```python
# evaluation.py

@dataclass
class EvalReport:
    eval_name: str
    dataset_path: str
    total_samples: int
    summary_metrics: dict[str, float]
    sample_results: list[Sample]
    config: dict[str, Any]
    fingerprint: dict[str, Any] | None = None  # NEW
    timestamp: str = field(default_factory=...)


async def evaluate(dataset, config: EvalConfig) -> EvalReport:
    from rollouts.fingerprint import fingerprint_eval
    
    # Introspect tools from first environment (if factory provided)
    tools = None
    if config.environment_factory:
        # Create one env to get tool list
        sample_env = await config.environment_factory({})
        tools = [t.name for t in sample_env.get_tools()]
    
    # Compute fingerprint
    fp = fingerprint_eval(config, tools=tools, dataset_path=dataset_path)
    
    # ... run evaluation ...
    
    report = EvalReport(
        ...,
        fingerprint=fp,  # Already a dict
    )
```

## Integration: Sessions

```python
# sessions.py

def create_session(..., model: str, env: str, tools: list[str]) -> Session:
    from rollouts.fingerprint import fingerprint_session
    
    fp = fingerprint_session(model, env, tools)
    
    header = SessionHeader(
        ...,
        fingerprint=fp,  # Already a dict
    )
```

On resume:

```python
# cli.py

def resume_session(session: Session, current_config: CLIConfig):
    from rollouts.fingerprint import fingerprint_session, warn_if_changed
    
    current_fp = fingerprint_session(
        model=current_config.model,
        env=current_config.env,
        tools=current_config.tools,
    )
    
    saved_fp = session.header.get("fingerprint", {})
    warn_if_changed(saved_fp, current_fp, context="session")
```

## Output Format

### report.json (Eval)

```json
{
  "eval_name": "swebench_lite",
  "fingerprint": {
    "config_hash": "a1b2c3d4e5f6g7h8",
    "git_sha": "def45678",
    "git_dirty": false,
    "domain": "eval",
    "endpoint_summary": "anthropic/claude-sonnet-4-5-20250929",
    "tools": ["bash", "write", "read", "edit"],
    "dataset_path": "/data/swebench_lite.jsonl"
  }
}
```

### Session JSONL Header

```jsonl
{"type": "session", "id": "abc123", "fingerprint": {"config_hash": "a1b2c3d4", "git_sha": "def45678", "git_dirty": false, "domain": "session", "model": "anthropic/claude-sonnet-4-5-20250929", "env": "coding", "tools": ["read", "write", "edit", "bash"]}}
```

## Implementation Plan

### Phase 1: Core module
- [ ] Create `rollouts/fingerprint.py` with core utilities
- [ ] `canonical_json()`, `hash_str()`, `git_info()`, `require_clean_git()`
- [ ] `callable_name()` with lambda rejection
- [ ] `file_checksum()` for datasets
- [ ] `fingerprint_short()` helper

### Phase 2: Eval integration
- [ ] Add `fingerprint` field to `EvalReport`
- [ ] Introspect tools from environment factory
- [ ] Compute fingerprint in `evaluate()`
- [ ] Include in `report.json` output

### Phase 3: Session integration
- [ ] Add `fingerprint` field to `SessionHeader`
- [ ] Compute fingerprint in `create_session()`
- [ ] Add `warn_if_changed()` on resume
- [ ] Show in session picker (optional)

### Phase 4: Training (future)
- [ ] `fingerprint_training()` for RL/training runs
- [ ] Integrate with training loop

## Files Changed

- `rollouts/fingerprint.py` (new)
- `rollouts/evaluation.py`
- `rollouts/dtypes.py` (EvalReport)
- `rollouts/sessions.py`
- `rollouts/cli.py` (resume warning)

## References

- [nmoe/config.py](https://github.com/Noumena-Network/nmoe/blob/main/nmoe/config.py) - `fingerprint()` function
- [nmoe/checkpoint.py](https://github.com/Noumena-Network/nmoe/blob/main/nmoe/checkpoint.py) - `build_states()` with `config_fingerprint`
- [nmoe/train.py](https://github.com/Noumena-Network/nmoe/blob/main/nmoe/train.py) - Resume validation
