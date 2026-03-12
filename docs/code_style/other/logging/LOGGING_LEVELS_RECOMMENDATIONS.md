# Logging Levels: INFO vs DEBUG Recommendations

## Principle
**INFO** = What users want to see in stdout (high-level progress, results, important status)  
**DEBUG** = Technical details useful for debugging but noisy for normal operation

---

## Categories

### ✅ INFO (Keep as INFO)

**Operation Start/Completion**
- `"deploying sglang server..."` - User wants to know operation started
- `"deployment complete"` / `"code deployed successfully"` - Important completion
- `"server is ready!"` - Critical status user needs to know
- `"bootstrap completed successfully"` - Important milestone

**Configuration Summary**
- `"model: {model}"`, `"gpus: {gpus}"`, `"port: {port}"` - User wants to confirm what's running
- `"endpoint: {provider} @ {api_base}"` - Confirms configuration
- `"running {n} samples with max concurrency: {m}"` - High-level operation summary

**Final Results/Summaries**
- `"summary:"`, `"total samples: {n}"`, `"completed: {n}"` - User wants to see results
- `"mean reward: {x}"` - Important metric
- `"successfully terminated {n} instance(s)"` - Operation result

**Important Status Changes**
- `"instance {id} ready"` - Critical status
- `"instance {id} provisioning started"` - Important milestone
- `"job completed (exit code: {code})"` - Final status

---

### 🔧 DEBUG (Should be DEBUG)

**Step-by-Step Progress Within Operations**
- `"step {i}/{n}: {cmd}"` - Detailed progress within bootstrap
- `"installing: {package}"` - Individual package installation (too verbose)
- `"installed successfully"` for individual packages - Too granular
- `"installing {n} git package(s)..."` - Detailed progress
- `"installing: {pkg_url}"` - Per-package detail

**Polling/Waiting Details**
- `"waiting for ssh..."` - Polling detail (user already knows instance is provisioning)
- `"waiting for instance {id} to reach running..."` - Internal polling
- `"gpus not ready ({err}). retrying..."` - Polling retry detail
- `"waiting for ssh details..."` - Internal state polling
- `"waiting for direct ssh (may take 5-15 min)"` - Polling detail

**Internal State/Technical Details**
- `"file exists"` / `"file does not exist"` - Internal state check
- `"workspace updated to: {hash}"` - Technical detail (user doesn't need hash)
- `"ssh ready: {ip}:{port} (took {elapsed}s)"` - Technical detail (redundant if we said "ready")
- `"ssh connectivity confirmed!"` - Redundant confirmation
- `"instance is active with ssh details assigned. waiting 10s..."` - Too detailed
- `"direct ssh: {ip}:{port}"` - Technical detail

**Detailed Progress Within Larger Operations**
- `"loaded {n} samples from {file}"` - Per-file detail (keep total as INFO)
- `"loading {annotation_file}..."` - Per-file detail
- `"pushing code from {local_path}..."` - Detailed step
- `"syncing results from {remote_path}..."` - Detailed step

**Package Installation Details**
- `"already satisfied: {packages}"` - Too verbose
- `"installed: {packages}"` - Too verbose for individual packages
- `"installed with --no-deps"` - Technical detail
- `"found {tool} at {path}"` - Internal verification

**Redundant/Verbose Confirmations**
- `"all bootstrap steps completed successfully"` - Redundant if we already said "bootstrap completed"
- `"ssh should be ready!"` - Redundant confirmation
- `"instance {id} is running"` - Redundant if we already said "ready" or "provisioning started"

**CLI Helper Messages**
- `"found keys at:"`, `"no ssh keys found"` - Error context (could be INFO for errors, DEBUG for normal flow)
- `"expected format: KEY=VALUE"` - Error context (INFO is fine for errors)

---

## Specific Recommendations by File

### bifrost/bifrost/git_sync.py
- ✅ INFO: `"all bootstrap steps completed successfully"` (keep - important milestone)
- 🔧 DEBUG: `"step {i}/{n}: {cmd}"` (detailed progress)
- 🔧 DEBUG: `"workspace updated to: {hash}"` (technical detail)

### bifrost/bifrost/deploy.py
- ✅ INFO: `"code deployed successfully"` (keep - important completion)
- ✅ INFO: `"code deployed to: {path}"` (keep - important result)

### bifrost/bifrost/remote_fs.py
- 🔧 DEBUG: `"file exists"` / `"file does not exist"` (internal state)

### broker/broker/providers/*.py
- ✅ INFO: `"instance {id} is running"` (keep - important status)
- 🔧 DEBUG: `"waiting for instance {id} to reach running..."` (polling detail)
- 🔧 DEBUG: `"waiting for ssh details..."` (polling detail)
- 🔧 DEBUG: `"ssh ready: {ip}:{port}"` (technical detail)
- 🔧 DEBUG: `"ssh connectivity confirmed!"` (redundant)
- 🔧 DEBUG: `"instance is active with ssh details assigned. waiting 10s..."` (too detailed)

### kerbal/kerbal/python_env.py
- ✅ INFO: `"python environment ready"` (keep - important milestone)
- 🔧 DEBUG: `"setting up python environment"` (could be INFO or DEBUG - borderline)
- 🔧 DEBUG: `"installing: {packages}"` (detailed progress)
- 🔧 DEBUG: `"installed: {packages}"` (detailed progress)
- 🔧 DEBUG: `"already satisfied: {packages}"` (too verbose)
- 🔧 DEBUG: `"installing {n} git package(s)..."` (detailed progress)
- 🔧 DEBUG: `"installed successfully"` (too granular)
- 🔧 DEBUG: `"found {tool} at {path}"` (internal verification)

### kerbal/kerbal/gpu.py
- ✅ INFO: `"gpus {ids} are available"` (keep - important status)
- 🔧 DEBUG: `"waiting for gpus {ids}..."` (polling detail)
- 🔧 DEBUG: `"gpus not ready ({err}). retrying..."` (polling retry)

### kerbal/kerbal/transfer.py
- 🔧 DEBUG: `"pushing code from {path}..."` (detailed step)
- ✅ INFO: `"code pushed to {workspace}"` (keep - important completion)
- 🔧 DEBUG: `"syncing results from {path}..."` (detailed step)
- ✅ INFO: `"results synced to {path}"` (keep - important completion)

### rollouts/rollouts/run_eval.py
- 🔧 DEBUG: `"loading {annotation_file}..."` (per-file detail)
- 🔧 DEBUG: `"loaded {n} samples from {file}"` (per-file detail)
- ✅ INFO: `"total: {n} samples"` (keep - important summary)
- ✅ INFO: `"endpoint: {provider} @ {api_base}"` (keep - configuration confirmation)
- ✅ INFO: `"model: {model}"` (keep - configuration confirmation)
- ✅ INFO: `"running {n} samples..."` (keep - high-level operation)
- 🔧 DEBUG: `"writing incremental results to: {path}"` (detailed step)
- ✅ INFO: `"jsonl results written to: {path}"` (keep - important completion)
- ✅ INFO: `"summary:"` and all summary lines (keep - user wants results)

### rollouts/rollouts/deploy.py
- ✅ INFO: `"deploying sglang server..."` (keep - operation start)
- ✅ INFO: `"model: {model}"`, `"gpus: {gpus}"`, `"port: {port}"` (keep - configuration)
- ✅ INFO: `"server is ready!"` (keep - critical status)
- ✅ INFO: `"bootstrap completed successfully"` (keep - important milestone)
- ✅ INFO: `"server started in tmux session: {name}"` (keep - important status)
- 🔧 DEBUG: `"server is ready (took ~{n}s)"` (redundant if we already said "ready")

---

## Edge Cases / Borderline

**Could be either INFO or DEBUG:**
- `"running {n} bootstrap step(s)..."` - Borderline. If bootstrap is quick, INFO is fine. If it's long-running, DEBUG might be better.
- `"monitoring job: {name}"` - INFO is fine (user wants to know what's being monitored)
- `"setting up python environment"` - Could be INFO (user wants to know what's happening) or DEBUG (too detailed)

**Context-dependent:**
- Error messages should generally be INFO or WARNING (users need to see errors)
- Configuration display: INFO (users want to confirm what's running)
- Per-item progress in loops: DEBUG (too verbose)
- Final summaries: INFO (users want results)

---

## Summary

**Move to DEBUG:**
1. Step-by-step progress within operations
2. Polling/waiting details
3. Internal state checks
4. Technical implementation details
5. Redundant confirmations
6. Per-item progress in loops

**Keep as INFO:**
1. Operation start/completion
2. Configuration summaries
3. Final results/summaries
4. Important status changes
5. High-level progress indicators

