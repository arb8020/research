# Infrastructure Pain Points Triage

> Analysis of the 8 complaints from GRPO RL training runs, with root causes and fixes aligned with code style principles.

---

## Summary Table

| # | Issue | Severity | Root Cause | Status |
|---|-------|----------|------------|--------|
| 1 | No easy job cancel | Medium | Missing CLI command | **Fixed** (has `--cancel`) |
| 2 | Log streaming crashes monitor | High | Server sends huge batches, client ignores `has_more` | Partially fixed |
| 3 | Can't tell if remote code is updated | Medium | No version/hash visibility | Not implemented |
| 4 | Old log files pollute new runs | Medium | Same log dir reused across runs | Not implemented |
| 5 | No way to SSH + run command easily | Low | `broker ssh` only prints command | Not implemented |
| 6 | Monitor crashes kill visibility | High | JSON parse errors crash TUI | Not implemented |
| 7 | Multiple stale pods accumulating | Medium | No indication which pods are active | Partially addressed |
| 8 | grpo.py endpoint bug (base_url vs api_base) | High | Silent 404s, no health check | Not implemented |

---

## Detailed Analysis

### 1. No Easy Job Cancel ✅ FIXED

**Complaint:** "took a while to realize we needed it"

**Current State:** `rollouts monitor --cancel <run_id>` exists in `monitor_cli.py:525-574`

**Code Location:**
```python
# rollouts/rollouts/tui/monitor_cli.py:525
def _cancel_job(run_id: str) -> int:
    """Cancel a running job by killing its tmux session."""
    ...
```

**Root Cause:** Feature existed but wasn't discoverable. The help text mentions it, but users didn't find it.

**Fix (Already Applied):**
- Added `--cancel` flag to monitor CLI
- Kills tmux session `bifrost-job-rl-training` via SSH
- Removes job from `~/.rollouts/jobs.json`

**Code Style Alignment:**
- ✅ **Tiger Style:** Clear function with assertions
- ✅ **Usage-first:** `rollouts monitor --cancel run_20250127-143052`

---

### 2. Log Streaming Crashes Monitor 🔶 PARTIALLY FIXED

**Complaint:** "LogsServer sends huge batches, crashes the client. Fixed pagination server-side, but client still doesn't handle `has_more` properly"

**Current State:**
- Server: `miniray/logs_server.py:90-110` implements pagination with `has_more`
- Client: `monitor_cli.py:385-420` sync loop - **does NOT handle `has_more`**

**Root Cause - Server Side (FIXED):**
```python
# miniray/logs_server.py:90-110
elif cmd == "tail":
    max_lines = msg.get("max_lines", 1000)  # Pagination limit
    max_bytes = msg.get("max_bytes", 512 * 1024)  # 512KB per response
    ...
    has_more = len(content) == max_bytes or f.read(1) != ""
    send({"lines": lines, "offset": new_offset, "has_more": has_more})
```

**Root Cause - Client Side (NOT FIXED):**
```python
# rollouts/rollouts/tui/monitor_cli.py:385-420 (sync_loop)
def sync_loop() -> None:
    while not stop_sync.is_set():
        ...
        worker.send({"cmd": "tail", "file": filename, "offset": offset})
        result = worker.recv()  # <-- Only fetches ONCE, ignores has_more
        new_lines = result.get("lines", [])
        offsets[filename] = result.get("offset", offset)  # <-- May be incomplete!
```

**The Bug:**
When `has_more=true`, the client should immediately fetch the next chunk. Currently it waits 2 seconds for the next sync cycle, and if the file grows rapidly (sglang.log during training), the offset gets out of sync.

**Fix Needed:**
```python
# In sync_loop, after receiving result:
while result.get("has_more"):
    # Fetch next chunk immediately
    worker.send({"cmd": "tail", "file": filename, "offset": result["offset"]})
    result = worker.recv()
    new_lines.extend(result.get("lines", []))
    # Update offset to final position
    new_offset = result.get("offset", new_offset)
```

**Code Style Alignment:**
- 🔄 **Tiger Style:** Add assertion that `has_more` is handled
- 🔄 **Push Ifs Up:** Centralize the pagination logic in the parent sync loop
- 🔄 **SSA:** Don't reuse `result` variable for multiple fetches

---

### 3. Can't Tell if Remote Code is Updated 🔴 NOT IMPLEMENTED

**Complaint:** "Spent time debugging wondering if fixes were deployed. Would be nice to have a version/hash shown"

**Current State:** No version tracking in deployment

**Root Cause:** 
- `rollouts/run.py:_deploy_and_submit()` pushes code via `bifrost.push()`
- No git hash or version info is captured or displayed
- The `broker ssh` command doesn't show what's deployed

**Proposed Fix:**

1. **Capture git info at deploy time:**
```python
# rollouts/rollouts/run.py, in _deploy_and_submit
def _get_deploy_info() -> dict:
    """Get git hash and dirty status for deployment tracking."""
    import subprocess
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        is_dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True
        ).stdout.strip() != ""
        return {"git_hash": git_hash, "dirty": is_dirty}
    except subprocess.CalledProcessError:
        return {"git_hash": "unknown", "dirty": False}
```

2. **Store in run metadata:**
```python
# In _deploy_and_submit, after workspace push
deploy_info = _get_deploy_info()
log("deploy_done", workspace=workspace, **deploy_info)
# Write to remote output dir
bifrost.exec(f'echo "{json.dumps(deploy_info)}" > {remote_output_dir}/deploy_info.json')
```

3. **Display in monitor:**
```python
# In monitor_cli.py _run_attached, after connection
print(f"Remote code: {deploy_info['git_hash']}{'*' if deploy_info['dirty'] else ''}")
```

**Code Style Alignment:**
- ✅ **No Magic Constants:** Use explicit git commands, not shell scripts
- ✅ **Single Assignment:** Each deployment gets its own info dict
- ✅ **Make Invisible Visible:** Git hash is explicit in UI

---

### 4. Old Log Files Pollute New Runs 🔴 NOT IMPLEMENTED

**Complaint:** "sglang.log from failed runs persists and gets streamed again. Should either auto-clean between runs or have separate log dirs per session"

**Current State:**
- `rollouts/run.py:run_remote()` creates `remote_output_dir = f"{workspace}/rollouts/results/rl/{run_name}"`
- Each run has its own directory - **this should work**
- But `sglang.log` might be written to workspace root if inference engine logs there

**Root Cause Investigation Needed:**

1. **Check where sglang.log is written:**
```python
# rollouts/rollouts/training/weight_sync.py:SGLangEngine.launch()
# Logs go to: self.output_dir / "sglang.log"
# output_dir is passed from grpo_train: output_dir
```

2. **The issue:** If `output_dir` is reused or if the inference engine is shared across runs, old logs persist.

**Proposed Fix:**

Option A: **Explicit cleanup on run start** (safer)
```python
# In _deploy_and_submit, before starting training
bifrost.exec(f"mkdir -p {remote_output_dir}")
# Clean any existing logs from previous failed runs
bifrost.exec(f"rm -f {remote_output_dir}/*.log {remote_output_dir}/*.jsonl")
```

Option B: **Session-scoped log dirs** (cleaner)
```python
# Instead of results/rl/{run_name}, use results/rl/{run_name}/{session_id}
# But this complicates the monitor's job of finding logs
```

**Recommended: Option A** - Clean on start, fail fast if cleanup fails.

**Code Style Alignment:**
- ✅ **State Invariants Positively:** "Log dir is clean before run starts"
- ✅ **Assertions Everywhere:** Assert no old logs exist after cleanup
- ✅ **Tiger Style:** Explicit cleanup step, not hidden side effect

---

### 5. No Way to SSH + Run Command Easily 🔴 NOT IMPLEMENTED

**Complaint:** "broker ssh just prints the command string, doesn't actually SSH. Had to manually copy/paste"

**Current State:**
```python
# broker/broker/cli.py:ssh command
@app.command()
def ssh(...):
    async def _ssh_async() -> None:
        ...
        # Output full SSH command with key path (copy-pastable)
        print(instance._instance.ssh_connection_string(ssh_key_path=ssh_key, full_command=True))
```

**Root Cause:** The `broker ssh` command intentionally only prints the command. This is a design choice (security? flexibility?), but it's annoying.

**Proposed Fix:**

Add `--exec` flag to actually run SSH:
```python
@app.command()
def ssh(
    ctx: typer.Context,
    instance_id: str = typer.Argument(...),
    provider: str | None = typer.Argument(None),
    exec: bool = typer.Option(False, "--exec", "-e", help="Actually execute SSH (not just print)"),
    command: str | None = typer.Option(None, "--command", "-c", help="Command to run via SSH"),
) -> None:
    """Get SSH connection string or execute SSH."""
    ...
    if exec or command:
        import subprocess
        ssh_cmd = instance._instance.ssh_connection_string(ssh_key_path=ssh_key, full_command=True)
        if command:
            ssh_cmd += f' "{command}"'
        subprocess.run(ssh_cmd, shell=True)
    else:
        print(instance._instance.ssh_connection_string(...))
```

**Usage:**
```bash
broker ssh abc123 --exec                    # Interactive SSH session
broker ssh abc123 --exec --command "nvidia-smi"  # Run command
```

**Code Style Alignment:**
- ✅ **Casey Granularity:** Low-level `ssh_connection_string()` + high-level `--exec`
- ✅ **Usage-first:** `broker ssh abc123 --exec` is what users want

---

### 6. Monitor Crashes Kill Visibility 🔴 NOT IMPLEMENTED

**Complaint:** "When monitor crashes (JSON parse), we lose all insight into what's happening on remote. Job might be running fine but we can't see it"

**Current State:**
- `monitor_cli.py:sync_loop()` catches `EOFError, BrokenPipeError, ConnectionResetError`
- Writes "[monitor] LogsServer connection lost" to training.log
- **But:** JSON parse errors in `worker.recv()` are not caught

**Root Cause:**
```python
# miniray/__init__.py (RemoteWorker)
def recv(self) -> dict:
    line = self._r.readline()
    if not line:
        raise EOFError("Connection closed")
    return json.loads(line)  # <-- Can raise JSONDecodeError, crashes caller
```

**Proposed Fix:**

1. **Wrap JSON parse in monitor:**
```python
# monitor_cli.py:sync_loop
def sync_loop() -> None:
    while not stop_sync.is_set():
        try:
            worker.send({"cmd": "list"})
            resp = _safe_recv(worker)  # <-- Wrapper with error handling
            ...
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            ...

def _safe_recv(worker) -> dict | None:
    """Receive with JSON error handling."""
    try:
        return worker.recv()
    except json.JSONDecodeError as e:
        _log("sync_json_error", error=str(e))
        # Write to local training.log so user sees it
        with open(local_sync_dir / "training.log", "a") as f:
            f.write(f"[monitor] JSON parse error: {e}\n")
        return None
```

2. **Add error boundary in TUI:**
```python
# rlmon/app.py:update
def update(model: Model, msg: object) -> tuple[Model, Cmd]:
    try:
        match msg:
            ...
    except Exception as e:
        # Log error but don't crash TUI
        logger.error(f"Update error: {e}")
        return model, Cmd.none()
```

**Code Style Alignment:**
- ✅ **Error Handling Decision Tree:** JSON parse = operational failure → return error, don't crash
- ✅ **Tiger Style:** Assert invariants, but handle operational failures gracefully
- ✅ **Minimize Stateful Components:** Monitor is stateless, can restart cleanly

---

### 7. Multiple Stale Pods Accumulating 🔶 PARTIALLY ADDRESSED

**Complaint:** "Had 3 pods running, easy to lose track. broker list shows them but no indication which are actively being used"

**Current State:**
- `broker list` shows all instances
- `rollouts monitor --runs --probe` shows jobs and probes LogsServer
- **But:** No integration between them - can't see which instances have active jobs

**Proposed Fix:**

Enhance `broker list` to show job association:
```python
# broker/broker/cli.py:list_instances
@app.command(name="list")
def list_instances(ctx: typer.Context) -> None:
    ...
    # Load rollouts jobs
    try:
        from rollouts.jobs import list_jobs
        jobs = list_jobs()
        job_by_node = {}
        for job in jobs:
            for node in job.nodes:
                job_by_node[f"{node.provider}:{node.node_id}"] = job
    except ImportError:
        jobs = []
        job_by_node = {}

    for instance in instances:
        node_key = f"{instance.provider}:{instance.id}"
        job = job_by_node.get(node_key)
        if job:
            status = f"[green]active: {job.job_id}[/green]"
        else:
            status = "[yellow]idle[/yellow]"
        table.add_row(..., status)
```

**Code Style Alignment:**
- ✅ **Make Invisible Visible:** Instance status is explicit
- ✅ **No Magic Constants:** Explicit job→instance mapping

---

### 8. grpo.py Endpoint Bug (base_url vs api_base) 🔴 NOT IMPLEMENTED

**Complaint:** "Silent 404s that fill logs. Should have been caught earlier, maybe with a health check that validates the endpoint responds"

**Current State:**
```python
# rollouts/rollouts/training/grpo.py:263
endpoint = Endpoint(
    model=f"openai/{config.model.name}",
    base_url=inference_engine.api_base,  # <-- api_base includes /v1
    api_format="openai-completions",
    ...
)
```

```python
# rollouts/rollouts/training/weight_sync.py:352-360
@property
def api_base(self) -> str:
    return f"http://localhost:{self.port}/v1"  # <-- Has /v1

@property
def base_url(self) -> str:
    return f"http://localhost:{self.port}"  # <-- No /v1
```

**The Bug:**
- `Endpoint.base_url` expects the base URL WITHOUT `/v1` (it adds it internally)
- But `grpo.py` passes `inference_engine.api_base` which INCLUDES `/v1`
- Result: URLs like `http://localhost:30000/v1/v1/completions` → 404

**Root Cause:** Naming confusion. `api_base` sounds like it should be the base for API calls, but `Endpoint` expects the server base.

**Proposed Fix:**

1. **Fix the call site:**
```python
# rollouts/rollouts/training/grpo.py:263
endpoint = Endpoint(
    model=f"openai/{config.model.name}",
    base_url=inference_engine.base_url,  # <-- Use base_url, not api_base
    api_format="openai-completions",
    ...
)
```

2. **Add health check validation:**
```python
# In grpo_train, after creating endpoint
async def _validate_endpoint(endpoint: Endpoint) -> None:
    """Validate endpoint responds before starting training."""
    import httpx
    health_url = f"{endpoint.base_url}/health"  # Or /v1/models
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(health_url, timeout=5.0)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Inference endpoint not ready: {e}")

# Call before training loop
await _validate_endpoint(endpoint)
```

3. **Better naming (optional refactor):**
```python
# weight_sync.py - rename for clarity
@property
def server_url(self) -> str:  # Was: base_url
    """Server base URL without /v1 (e.g., http://localhost:30000)"""
    return f"http://localhost:{self.port}"

@property  
def openai_base_url(self) -> str:  # Was: api_base
    """OpenAI-compatible API base URL with /v1."""
    return f"{self.server_url}/v1"
```

**Code Style Alignment:**
- ✅ **No Magic Constants:** Explicit URL construction
- ✅ **Assertions Everywhere:** Health check asserts endpoint is ready
- ✅ **Tiger Style:** Fail fast with clear error message

---

## Implementation Priority

### P0 (Critical - Fix This Week)
1. **Issue 8:** Fix `base_url` vs `api_base` bug + add health check
2. **Issue 2:** Handle `has_more` in monitor client

### P1 (High - Fix Next Week)
3. **Issue 6:** Add error boundaries to monitor (JSON parse handling)
4. **Issue 4:** Clean log dir on run start
5. **Issue 3:** Add git hash tracking to deployments

### P2 (Medium - Nice to Have)
6. **Issue 5:** Add `--exec` flag to `broker ssh`
7. **Issue 7:** Show job association in `broker list`

---

## Code Style Checklist for Fixes

For each fix, verify:

- [ ] **Usage-first:** Did I write the usage code first?
- [ ] **Tiger Style:** Are there assertions checking preconditions?
- [ ] **Push Ifs Up:** Is control flow centralized?
- [ ] **SSA:** Are intermediate values named and visible?
- [ ] **No Magic Constants:** Are all literals named?
- [ ] **Make Invisible Visible:** Are assumptions explicit?
- [ ] **Error Handling:** Is the decision tree applied correctly?

---

## Files to Modify

| Issue | Files |
|-------|-------|
| 2 | `rollouts/rollouts/tui/monitor_cli.py` (sync_loop) |
| 3 | `rollouts/rollouts/run.py`, `monitor_cli.py` |
| 4 | `rollouts/rollouts/run.py` (_deploy_and_submit) |
| 5 | `broker/broker/cli.py` (ssh command) |
| 6 | `rollouts/rollouts/tui/monitor_cli.py`, `rlmon/app.py` |
| 7 | `broker/broker/cli.py` (list_instances) |
| 8 | `rollouts/rollouts/training/grpo.py`, `weight_sync.py` |
