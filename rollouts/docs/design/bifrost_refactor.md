# Bifrost Refactor: Applying Semantic Compression

> **Status**: Notes / Future Work
> **Date**: 2025-01-14

## Context

Bifrost is 5500 lines. Applying Casey Muratori's semantic compression and API design principles, it could likely be ~500 lines for what we actually use.

## Current State

```
bifrost/
├── client.py          1093 lines  ← Core, but bundled
├── deploy.py           869 lines  ← Deployment orchestration
├── async_client.py     747 lines  ← Async variant (duplication)
├── cli.py              518 lines  ← CLI wrapper (separate concern)
├── git_sync.py         400 lines  ← Git operations
├── types.py            337 lines  ← Dataclasses
├── job.py              301 lines  ← tmux job management
├── job_manager.py      290 lines  ← Job tracking
├── validation.py       290 lines  ← Validation (overkill?)
├── remote_fs.py        213 lines  ← File operations
├── provision.py        176 lines  ← Broker integration
├── server.py           160 lines  ← Server mode
├── __init__.py          77 lines  ← Exports
                       ─────
                       5471 lines
```

## Analysis: Casey's Five Characteristics

### 1. Granularity - Can I break it down?

**Problem:** Operations are bundled, can't access lower levels easily.

```python
# Current: bundled
client.push()  # git bundle + upload + unbundle + symlink

# Want: granular options
bundle = create_git_bundle(local_dir)
client.upload(bundle, remote_path)
client.exec_raw(f"git clone {bundle}")
# OR
client.rsync(local_dir, remote_dir)  # Skip git entirely
```

```python
# Current: bundled
client.exec("python train.py")  # cd + env + command

# Want: granular options
client.exec_raw("ls -la")  # No wrapping
client.exec("cmd", working_dir=..., env=...)  # Full version
```

### 2. Redundancy - Multiple ways to do same thing?

- `exec()` vs `run_detached()` - good
- No raw SSH command option - missing
- No rsync alternative to git push - missing

### 3. Coupling - A implies B?

- `push()` couples: git bundle creation + upload + unbundle + symlink
- `exec()` couples: cd + env setup + command
- Can't push without git machinery
- Can't exec without working_dir setup

### 4. Retention - Mirrored state?

- `_last_workspace` - minimal
- `_ssh_client` - connection caching
- This is fine, not problematic

### 5. Flow Control - Who calls who?

- Good: Always call Bifrost, no callbacks
- No inheritance required
- Clean here

## Proposed Compressed Structure

```python
# Layer 0: Raw SSH (~100 lines)
class SSHSession:
    def connect(self, host, port, user, key_path): ...
    def exec_raw(self, command) -> ExecResult: ...
    def upload(self, local, remote): ...
    def download(self, remote, local): ...
    def close(self): ...

# Layer 1: Convenience (~100 lines)
def exec_with_env(session, cmd, working_dir=None, env=None): ...
def upload_dir(session, local_dir, remote_dir): ...

# Layer 2: Git sync (~100 lines)
def create_bundle(local_dir) -> Path: ...
def push_bundle(session, bundle, remote_workspace): ...
def git_push(session, local_dir, remote_workspace): ...  # Combines above

# Layer 3: Job management (~100 lines)
def run_detached(session, cmd, job_id) -> JobInfo: ...
def job_status(session, job) -> str: ...
def job_logs(session, job) -> Iterator[str]: ...
def job_kill(session, job): ...

# Layer 4: High-level client (~100 lines)
class BifrostClient:
    """Convenience wrapper that combines layers 0-3."""
    def __init__(self, ssh_connection, ssh_key_path): ...
    def push(self): ...  # Uses Layer 2
    def exec(self, cmd): ...  # Uses Layer 1
    def run_detached(self, cmd): ...  # Uses Layer 3
```

**Total: ~500 lines** for equivalent functionality.

Each layer is independently usable. User can:
- Use just `SSHSession` for raw access
- Use `git_push()` without BifrostClient
- Use `BifrostClient` for convenience (most common)

## What to Delete

| Component | Lines | Action |
|-----------|-------|--------|
| `async_client.py` | 747 | Delete or generate from sync |
| `cli.py` | 518 | Separate package |
| `validation.py` | 290 | Inline critical checks only |
| `deploy.py` | 869 | Simplify significantly |
| `job_manager.py` | 290 | Merge into job.py |
| `provision.py` | 176 | Move to broker |
| `server.py` | 160 | Delete (unused?) |

## Casey's Checklist Applied

- [ ] Any retain-mode construct has immediate-mode equivalent
- [ ] No callbacks or inheritance required ✓
- [ ] No required proprietary data types ✓
- [ ] Every non-atomic function can be replaced by 2-4 finer-grained calls
- [ ] Data structures are transparent ✓
- [ ] Don't have to use their resource management ✓
- [ ] Don't have to use their file format ✓

## Action Items

1. **Write usage code first** - What do we actually call?
   ```python
   client = BifrostClient(ssh_conn, key)
   client.push()
   result = client.exec("rollouts eval dataset.jsonl")
   ```

2. **Identify minimum granularity needed**
   - Do we ever need raw SSH without BifrostClient? (Probably not)
   - Do we ever need non-git push? (Maybe - rsync is simpler)
   - Do we ever need exec without working_dir? (Maybe)

3. **Rebuild from usage, not from existing code**

## References

- `docs/code_style/casey_muratori_semantic_compression.md`
- `docs/code_style/code_reuse_casey_muratori.md`
